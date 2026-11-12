#!/usr/bin/env python3
"""The chief of the autolab: one agent that walks an improvement trajectory (propose -> run -> verdict -> check on other data -> review -> accept to a branch).

    python3 autolab_chief.py            one tick (cron every 30 minutes; does nothing unless a human wrote the flag ~/.config/autolab/CHIEF with `шеф вкл`)
    python3 autolab_chief.py --status   state, limits, last actions
    python3 autolab_chief.py --force    a tick now, ignoring the minimum gap between ticks
    python3 autolab_chief.py selftest

The model only PLANS (a JSON list of actions with reasons). Everything that decides what is allowed lives in this file and in autolab.py:
the limits (actions per tick, active hypotheses, rechecks per hypothesis, queue depth, dollars per day / week), the acceptance policy
(`accept_policy`) and the dry-run switch (`chief_dry_run`, true by default). An accept of the chief goes only to the branch `auto/<name>` on GitHub
(never --force); `main` of the host and `master` of GitHub get the change only after a human `принять <имя>` in Telegram. Delete the branch to roll back.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import autolab as al  # noqa: E402
import autolab_ctl as c  # noqa: E402
import autolab_status as astat  # noqa: E402

UID = "chief"
ACTIONS = ("propose", "check", "rerun", "review", "accept", "reject", "wait", "note", "parallel")
EXTRA_FORBIDDEN = ("Dockerfile", "harness/mode.py")                  # kept by the human on top of the autolab's own patch rules
EXTRA_FORBIDDEN_PREFIX = ("profiles/mode_",)
DECIDED = ("accepted", "rejected", "auto_rejected")


# ------------------------------------------------------------------ state
def state_file() -> Path:
    return al.DRAFTS / "chief.json"


def plan_file() -> Path:
    return al.status_root() / "CHIEF_PLAN.md"


def _state() -> dict:
    day = time.strftime("%Y-%m-%d")
    try:
        st = json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    if st.get("day") != day:
        st.update(day=day, actions_today=0, ticks_today=0, spent_notice="")
    st.setdefault("last_tick", 0)
    st.setdefault("digest_day", "")
    st.setdefault("log", [])
    st.setdefault("notes", [])
    return st


def _save(st: dict) -> None:
    al.DRAFTS.mkdir(parents=True, exist_ok=True)
    st["log"] = st["log"][-60:]
    st["notes"] = st["notes"][-20:]
    state_file().write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")


def is_on() -> bool:
    return al.chief_flag().exists()


def dry_run() -> bool:
    return bool(al.limits_cfg().get("chief_dry_run", True))


def _say(text: str) -> None:
    try:
        al.outbox_put(al.allowed_chat_ids(), text)
    except Exception as ex:  # noqa: BLE001 - the chat must not break a tick
        al.log(f"chief: chat message not sent: {type(ex).__name__}: {ex}")


def _chat() -> str:
    ids = al.allowed_chat_ids()
    return str(ids[0]) if ids else ""


# ------------------------------------------------------------------ hypotheses: families, best arm, checks
def _names() -> list[str]:
    return sorted(f.name[:-10] for f in al.DRAFTS.glob("*.meta.json"))


def family(name: str) -> list[str]:
    base = al.family_base(name)
    return sorted({n for n in _names() if al.family_base(n) == base or al.meta(n).get("check_of") == base} | {base})


def _arm_stats(name: str, arm: str | None = None) -> tuple[str, dict] | None:
    """(arm, points_diff_vs_first) of the arm that carries the verdict (accept first, else the largest mean)."""
    m = al.meta(name)
    res, ver = m.get("result") or {}, m.get("verdict") or {}
    best = None
    for a, v in res.items():
        d = v.get("points_diff_vs_first")
        if not d or (arm and a != arm):
            continue
        key = (ver.get(a) == "accept", d.get("mean", 0))
        if best is None or key > best[0]:
            best = (key, a, d)
    return (best[1], best[2]) if best else None


def _budget(name: str) -> int:
    try:
        return int(al.spec_of(name).get("budget") or 0)
    except (TypeError, ValueError):
        return 0


def _set_of(name: str) -> str:
    return str(al.spec_of(name).get("set") or "")


def checks_of(base: str) -> list[str]:
    return [n for n in family(base) if al.meta(n).get("check_of") == al.family_base(base) and n != base]


def rechecks(base: str) -> int:
    """Runs spent on a hypothesis after its first experiment: checks on other data and repeats."""
    b = al.family_base(base)
    return sum(1 for n in family(b) if n != b)


# ------------------------------------------------------------------ the acceptance policy (code, not prompt)
def accept_policy(name: str) -> tuple[bool, list[str]]:
    """(ok, reasons the accept to `auto/<name>` is refused). The model may recommend; this decides."""
    base = al.family_base(name)
    m = al.meta(base)
    why: list[str] = []
    if not (al.DRAFTS / f"{base}.yaml").exists():
        return False, [f"no hypothesis {base}"]
    dec = al.decision_of(base)
    if dec in DECIDED:
        why.append(f"already decided ({al.DECISION_RU.get(dec, dec)})")
    if m.get("in_stack"):
        why.append(f"already on the stack {STACK}: only a human `принять стек` goes further")
    if al.stack_active():
        sig = al.stack_sig()
        for k in ("stack_at_apply", "stack_at_submit"):
            if k in m and m[k] != sig:
                why.append(f"measured on another state of the stack ({m[k] or 'main'} instead of {sig}): `rerun` it from scratch on the current stack")
                break
        else:
            if "stack_at_apply" not in m and "stack_at_submit" not in m:
                why.append("measured before the stack existed: `rerun` it from scratch on the current stack")
    if m.get("stale"):
        why.append("the hypothesis is stale (another hypothesis of the same type was accepted after it): re-run it from scratch")
    ver = m.get("verdict") or {}
    arms = [a for a, v in ver.items() if v == "accept"]
    if not arms:
        why.append("the lab verdict on the dev set is not accept (for a binding budget this is the in-budget verdict)")
    st = _arm_stats(base, arms[0] if arms else None)
    if arms and not (st and st[1].get("mean", 0) > 0):
        why.append("the accepted arm has no positive measured difference")
    # (2) a check on a fresh or narrow set
    chk = [n for n in checks_of(base) if al.meta(n).get("verdict")]
    good_chk = False
    for n in chk:
        cv = al.meta(n).get("verdict") or {}
        a = arms[0] if arms and arms[0] in cv else (next(iter(cv)) if cv else None)
        cs = _arm_stats(n, a)
        if a is None or cs is None:
            continue
        if cv.get(a) == "reject":
            why.append(f"the check {n} on {_set_of(n)} says reject")
        elif cs[1].get("mean", 0) > 0 and _set_of(n) != _set_of(base):
            good_chk = True
    if not good_chk:
        why.append("no check on a fresh or narrow set with a verdict that is not reject and a positive difference (`check`)")
    # (3) a second budget without a loss beyond the threshold
    b0 = _budget(base)
    second = False
    for n in [base] + chk:
        bn = _budget(n)
        if b0 and bn and abs(bn - b0) / b0 >= 0.2 and al.meta(n).get("verdict"):
            cv = al.meta(n).get("verdict") or {}
            a = arms[0] if arms and arms[0] in cv else (next(iter(cv)) if cv else None)
            cs = _arm_stats(n, a)
            if cs and cv.get(a) != "reject" and cs[1].get("mean", 0) >= -float(cs[1].get("threshold") or 0):
                second = True
    if not second:
        why.append(f"no second budget (at least 20% away from {b0}) without a loss beyond the threshold (`check` with `budget`)")
    # (4) the reviewer
    rec = m.get("review_recommendation")
    if not rec or rec == "не указана":
        why.append("no review yet (`review`)")
    elif rec == "отклонить":
        why.append("the reviewer recommends reject")
    # (6) the patch and the cells: the accept preflight of the human flow plus the files the human keeps
    info = None
    try:
        info, w = _preflight(base)
    except Exception as ex:  # noqa: BLE001
        w = f"preflight failed: {type(ex).__name__}: {ex}"
    if not info:
        why.append(w)
    else:
        bad = [f for f in info.get("files", []) if f in EXTRA_FORBIDDEN or f.startswith(EXTRA_FORBIDDEN_PREFIX)]
        if bad:
            why.append(f"the patch touches {bad}: these stay with the human")
    return (not why), why


# ------------------------------------------------------------------ the stack auto/stack: accepted hypotheses accumulate on one branch
# A hypothesis that passed the policy is added to the stack (host: worktree _stack on the branch auto/stack; GitHub: the branch auto/stack, appended to,
# never --force). New hypotheses are prepared and measured on top of the stack (autolab.cmd_apply / cmd_submit). The whole stack is re-measured against
# main from time to time; when it loses, the culprit is found by leave-one-out arms and reverted. Only a human `принять стек` reaches main / master.
STACK = al.STACK_BRANCH


def _cfg() -> dict:
    return al.limits_cfg()


def _patch_dir() -> Path:
    p = al.DRAFTS / "stack_patches"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _git(repo: Path, *a, env: dict | None = None):
    return c._git(repo, *a, env=env)


def _apply_patch(repo: Path, pf: Path, msg: str, reverse: bool = False, env: dict | None = None) -> str:
    """"" on success, else the reason (the tree is restored on failure)."""
    rv = ["-R"] if reverse else []
    if _git(repo, "apply", "--check", *rv, str(pf), env=env).returncode:
        return "the patch does not apply (" + ("it cannot be taken out: later items touch the same lines" if reverse else "the stack moved") + ")"
    if _git(repo, "apply", "--index", *rv, str(pf), env=env).returncode or _git(repo, "commit", "-m", msg, env=env).returncode:
        _git(repo, "reset", "--hard", "HEAD", env=env)
        return "could not commit the patch"
    return ""


def _items(st: dict | None = None) -> list[dict]:
    return [i for i in (st or al.stack_load())["items"] if i.get("active", True)]


def _item_env(base: str, arm: str) -> dict:
    arms = al.spec_of(base).get("arms") or {}
    ref = next(iter(arms), None)
    renv = (arms.get(ref) or {}).get("env") or {}
    return {k: v for k, v in ((arms.get(arm) or {}).get("env") or {}).items() if renv.get(k) != v}


def _preflight(base: str):
    return c.accept_preflight(base, against=(STACK if al.stack_active() else "HEAD"))


def stack_ensure() -> str:
    """Create the branch auto/stack and its worktree from main of the host copy ("" or the reason)."""
    repo, wt = al.repo_root(), al.stack_wt()
    if wt.exists():
        return ""
    if repo.name != "autolab_repo":
        return "the stack lives only in the host copy ~/harness-lab/autolab_repo"
    if _git(repo, "status", "--porcelain").stdout.strip():
        return "the host copy has uncommitted changes"
    if _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != "main":
        return "the host copy is not on main"
    _git(repo, "worktree", "prune")
    _git(repo, "branch", "-D", STACK)
    wt.parent.mkdir(parents=True, exist_ok=True)
    r = _git(repo, "worktree", "add", "-b", STACK, str(wt), "HEAD")
    if r.returncode:
        return "worktree: " + (r.stderr or r.stdout)[-200:]
    st = al.stack_load()
    st["base"] = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    al.stack_save(st)
    return ""


def _clone_env() -> tuple[Path, dict, str, str] | str:
    cfg = _cfg()
    key = c.deploy_key()
    if not key.exists():
        return "no deploy key (~/.config/autolab/deploy_key)"
    remote = os.environ.get("AUTOLAB_PUSH_REMOTE") or cfg["push_remote"]
    br = cfg["push_branch"]
    clone = Path(os.environ.get("AUTOLAB_PUSH_CLONE_AUTO") or al.LAB / "push_repo_auto")
    env = dict(os.environ, GIT_SSH_COMMAND=f"ssh -i {key} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes", GIT_TERMINAL_PROMPT="0")
    if not (clone / ".git").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", remote, str(clone)], capture_output=True, text=True, env=env)
        if r.returncode:
            return "clone failed: " + al.scrub((r.stderr or r.stdout))[-200:]
    r = _git(clone, "fetch", "origin", env=env, )
    if r.returncode:
        return "git fetch failed: " + al.scrub((r.stderr or r.stdout))[-200:]
    return clone, env, remote, br


def push_stack() -> tuple[bool, str]:
    """Append the operations not yet on GitHub (add / revert, in order) to the branch auto/stack of the push clone and push it (never --force)."""
    st = al.stack_load()
    ops = st.setdefault("ops", [])
    done = int(st.get("pushed_ops", 0))
    if done >= len(ops):
        return True, "GitHub is up to date"
    ce = _clone_env()
    if isinstance(ce, str):
        return False, ce + " (the stack is on the host; GitHub gets it at the next push)"
    clone, env, remote, br = ce
    has = _git(clone, "rev-parse", "--verify", "-q", f"origin/{STACK}", env=env).returncode == 0
    if not has:
        done = 0
    start = f"origin/{STACK}" if has else f"origin/{br}"
    for step in (("checkout", "-B", STACK, start), ("reset", "--hard", start)):
        r = _git(clone, *step, env=env)
        if r.returncode:
            return False, f"git {step[0]} failed: " + al.scrub((r.stderr or r.stdout))[-200:]
    for op in ops[done:]:
        pf = _patch_dir() / f"{op['name']}.patch"
        if not pf.exists() or not pf.stat().st_size:
            continue
        err = _apply_patch(clone, pf, f"autolab chief stack: {op['op']} {op['name']}", reverse=(op["op"] == "revert"), env=env)
        if err:
            _git(clone, "reset", "--hard", start, env=env)
            _git(clone, "checkout", br, env=env)
            return False, f"GitHub {STACK}: {op['op']} {op['name']}: {err}"
    r = _git(clone, "push", "origin", STACK, env=env)
    _git(clone, "checkout", br, env=env)
    if r.returncode:
        return False, "push failed: " + al.scrub((r.stderr or r.stdout))[-240:]
    st["pushed_ops"] = len(ops)
    al.stack_save(st)
    return True, f"pushed to {STACK} on GitHub"


def stack_add(name: str) -> tuple[bool, str]:
    """Policy -> the patch on the stack worktree (host) -> GitHub auto/stack. The hypothesis stays undecided: a human decides with `принять стек`."""
    base = al.family_base(name)
    ok, why = accept_policy(base)
    if not ok:
        return False, "refused by the policy: " + "; ".join(why)
    st = al.stack_load()
    if st.get("blocked"):
        return False, "the stack is blocked: " + st["blocked"]
    if (st.get("check") or {}).get("state") == "running" or (st.get("ablation") or {}).get("state") == "running":
        return False, "a check of the stack is running: wait for it"
    info, w = _preflight(base)
    if not info:
        return False, w
    err = stack_ensure()
    if err:
        return False, err
    st = al.stack_load()
    pf = _patch_dir() / f"{base}.patch"
    pf.write_bytes(b"")
    if not info.get("no_code"):
        patch = _git(al.repo_root(), "diff", f"{STACK}...{info['branch']}").stdout
        pf.write_bytes(patch.replace("\r\n", "\n").encode("utf-8"))
        err = _apply_patch(al.stack_wt(), pf, f"autolab chief: {base} ({info['commit']})")
        if err:
            return False, "on the stack: " + err
    ver = al.meta(base).get("verdict") or {}
    arm = next((a for a, v in ver.items() if v == "accept"), "")
    st["items"].append({"name": base, "arm": arm, "env": _item_env(base, arm), "commit": info.get("commit") or "", "no_code": bool(info.get("no_code")),
                        "cells": info.get("cells") or [], "active": True, "added": time.strftime("%Y-%m-%d %H:%M")})
    st.setdefault("ops", []).append({"op": "add", "name": base})
    st["check"], st["ablation"] = None, None
    st["history"].append(f"{time.strftime('%Y-%m-%d %H:%M')} add {base}")
    al.stack_save(st)
    al.set_meta(base, auto_branch=STACK, in_stack=True, auto_accepted_at=time.time())
    pushed, pmsg = push_stack()
    return True, f"on the stack {STACK} (item {len(_items(st))}, env {st['items'][-1]['env'] or '-'}); {pmsg}; main only after a human `принять стек`"


def stack_revert(name: str, why: str) -> tuple[bool, str]:
    st = al.stack_load()
    it = next((i for i in st["items"] if i["name"] == name and i.get("active", True)), None)
    if not it:
        return False, f"{name} is not on the stack"
    pf = _patch_dir() / f"{name}.patch"
    if not it.get("no_code") and pf.exists() and pf.stat().st_size:
        err = _apply_patch(al.stack_wt(), pf, f"autolab chief: revert {name} ({why[:80]})", reverse=True)
        if err:
            return False, f"{name} cannot be taken out of the stack: {err}"
    it["active"], it["reverted"] = False, why
    st.setdefault("ops", []).append({"op": "revert", "name": name})
    st["check"], st["ablation"] = None, None
    st["checked_n"] = min(int(st.get("checked_n", 0)), len(st["ops"]) - 1)
    st["history"].append(f"{time.strftime('%Y-%m-%d %H:%M')} revert {name}: {why[:120]}")
    al.stack_save(st)
    al.set_meta(name, auto_branch=None, in_stack=False, stack_reverted=why[:300])
    pushed, pmsg = push_stack()
    return True, f"{name} taken out of the stack ({why[:100]}); {pmsg}"


def stack_reset(why: str) -> str:
    """Forget the stack (after the human accept, or by hand): worktree, local branch and the branch on GitHub go; the patches stay in autolab/ for the record."""
    repo, wt = al.repo_root(), al.stack_wt()
    st = al.stack_load()
    names = [i["name"] for i in _items(st)]
    if wt.exists():
        _git(repo, "worktree", "remove", "--force", str(wt))
    _git(repo, "branch", "-D", STACK)
    ce = _clone_env()
    remote_msg = ""
    if not isinstance(ce, str):
        clone, env, remote, br = ce
        if _git(clone, "rev-parse", "--verify", "-q", f"origin/{STACK}", env=env).returncode == 0:
            r = _git(clone, "push", "origin", "--delete", STACK, env=env)
            remote_msg = " Ветка на GitHub удалена." if not r.returncode else " Ветку на GitHub удалить не вышло."
    hist = st.get("history", [])[-30:] + [f"{time.strftime('%Y-%m-%d %H:%M')} reset: {why}"]
    for n in names:
        al.set_meta(n, auto_branch=None, in_stack=False)
    al.stack_save({"items": [], "check": None, "ablation": None, "history": hist, "seq": st.get("seq", 0)})
    return f"🗑 Стек сброшен ({why}); в нём были: {', '.join(names) or 'ничего'}.{remote_msg}"


# --- checks of the stack
def _set_for_stack() -> str:
    sets = al.check_sets()
    want = _cfg().get("chief_stack_set")
    if want in sets:
        return want
    return next(iter(sorted(sets)), "")


def _start_stack_exp(kind: str, arms: dict, why: str) -> tuple[bool, str]:
    cfg = _cfg()
    ds = _set_for_stack()
    if not ds:
        return False, "no set for the stack check"
    if al.queue_depth() >= cfg["max_queue_depth"]:
        return False, "queue is full"
    st = al.stack_load()
    st["seq"] = int(st.get("seq", 0)) + 1
    name = f"stk_{st['seq']}" + ("_abl" if kind == "ablation" else "")
    al.stack_save(st)
    al.write_draft({"name": name, "hypothesis": f"(stack {kind}) {why}", "set": ds, "model": "auto", "repeats": int(cfg.get("chief_stack_repeats", 2)),
                    "budget": int(cfg.get("chief_stack_budget", 30000)), "arms": arms, "target_cells": [], "target_mode": cfg.get("goal_mode") or "accuracy"})
    al.set_meta(name, stage="proposed", sim_verdict="pass", stack_exp=True, stack_kind=kind, stack_sig_at=al.stack_sig())
    try:
        al.cmd_submit(name)
    except SystemExit as ex:
        return False, f"not submitted: {al.scrub(str(ex))[:160]}"
    c._spawn(["await", name], "await")
    st = al.stack_load()
    rec = {"name": name, "sig": al.stack_sig(), "state": "running", "started": time.time()}
    st["check" if kind == "check" else "ablation"] = rec
    al.stack_save(st)
    return True, name


def _outcome(name: str) -> tuple[str, dict]:
    """(state, verdicts): 'running' while there is no verdict, else 'done'; 'lost' when it vanished from the queue without a verdict for 6 hours."""
    m = al.meta(name)
    if m.get("verdict"):
        return "done", m["verdict"]
    q = any((al.EXP / "queue" / k / f"{name}.yaml").exists() for k in ("pending", "running"))
    if not q and time.time() - float(m.get("queued_at") or 0) > 6 * 3600:
        return "lost", {}
    return "running", {}


def stack_housekeeping(dry: bool) -> list[str]:
    """No model: advance the checks of the stack. Returns lines for the chat."""
    out: list[str] = []
    st = al.stack_load()
    if not _items(st):
        return out
    cfg = _cfg()
    sig = al.stack_sig()
    chk, abl = st.get("check"), st.get("ablation")

    def say(t: str):
        out.append(t)

    # 1. a running check finished
    if chk and chk.get("state") == "running":
        state, ver = _outcome(chk["name"])
        if state == "running":
            return out
        stt = _arm_stats(chk["name"], "stack")
        mean = stt[1].get("mean", 0) if stt else 0
        v = ver.get("stack")
        good = state == "done" and v in ("accept", "clarify") and mean > 0
        chk.update(state="passed" if good else "failed", mean=mean, verdict=v or state)
        if good:
            st["checked_n"] = len(st.get("ops", []))
        st["history"].append(f"{time.strftime('%Y-%m-%d %H:%M')} check {chk['name']}: {chk['state']} ({v}, {mean:+.1f})")
        al.stack_save(st)
        say(f"стек {chk['name']}: общая проверка {'пройдена ✅' if good else 'ПРОВАЛЕНА ❌'} (вердикт {v or state}, разность {mean:+.1f})")
        chk = st["check"]
    # 2. a running ablation finished
    if abl and abl.get("state") == "running":
        state, ver = _outcome(abl["name"])
        if state == "running":
            return out
        cand = []
        for a, v in ver.items():
            s_ = _arm_stats(abl["name"], a)
            if a.startswith("no_") and v == "accept" and s_:
                cand.append((s_[1].get("mean", 0), a[3:]))
        abl["state"] = "done"
        al.stack_save(st)
        if cand:
            culprit = max(cand)[1]
            good, msg = (True, f"dry-run: would take {culprit} out") if dry else stack_revert(culprit, f"culprit of the failed stack check {chk['name'] if chk else ''} (ablation {abl['name']})")
            say(f"стек: виновник {culprit}: {msg}")
        else:
            st = al.stack_load()
            st["blocked"] = f"the stack check failed and no single item explains it (ablation {abl['name']}): look at it by hand; `стек сбросить` drops the stack"
            al.stack_save(st)
            say("стек: общая проверка провалена, а по одной гипотезе виновника нет (взаимодействие?). Стек заблокирован, жду вас: `стек`, `стек сбросить`.")
        return out
    st = al.stack_load()
    chk, abl = st.get("check"), st.get("ablation")
    if st.get("blocked"):
        return out
    # 3. a failed check: the culprit
    if chk and chk.get("state") == "failed" and chk.get("sig") == sig and not abl:
        its = _items(st)
        if len(its) == 1:
            good, msg = (True, "dry-run: would take it out") if dry else stack_revert(its[0]["name"], f"the stack check {chk['name']} failed with it alone")
            say(f"стек: единственная гипотеза {its[0]['name']} провалила общую проверку: {msg}")
            return out
        envs = [i for i in its if i.get("env")]
        if len(envs) < 2:
            st["blocked"] = "the stack check failed, and the items without switches cannot be taken out one by one: decide by hand"
            al.stack_save(st)
            say("стек: проверка провалена, а гипотезы без переключателей по одной не отключить. Стек заблокирован, жду вас.")
            return out
        senv = al.stack_env()
        arms = {"full": {"env": senv}}
        for i in envs:
            arms["no_" + i["name"]] = {"env": {k: v for k, v in senv.items() if k not in i["env"]}}
        if dry:
            say(f"стек: dry-run: поставил бы поиск виновника ({len(arms) - 1} вариантов без одной гипотезы)")
            return out
        ok, nm = _start_stack_exp("ablation", arms, f"leave-one-out after the failed check {chk['name']}")
        say(f"стек: поиск виновника {nm if ok else '(не поставлен: ' + nm + ')'}")
        return out
    # 4. time for a check
    unchecked = len(st.get("ops", [])) - int(st.get("checked_n", 0))
    lu = time_left_h()
    closing = lu is not None and lu <= float(cfg.get("chief_final_check_h", 3)) and unchecked > 0
    if (not chk or chk.get("sig") != sig) and (unchecked >= int(cfg.get("chief_stack_check_every", 2)) or closing):
        arms = {"main": {}, "stack": {"env": al.stack_env()}}
        if dry:
            say(f"стек: dry-run: поставил бы общий перепрогон стека против main ({len(_items(st))} гипотез)")
            return out
        ok, nm = _start_stack_exp("check", arms, f"the whole stack ({sig}) against main with the stack switches off")
        say(f"стек: общий перепрогон {nm if ok else '(не поставлен: ' + nm + ')'}")
    return out


# --- the human accept of the whole stack
def accept_stack_preflight() -> tuple[dict | None, str]:
    st = al.stack_load()
    its = _items(st)
    if not its or not al.stack_wt().exists():
        return None, "в стеке ничего нет (`стек`)"
    if st.get("blocked"):
        return None, "стек заблокирован: " + st["blocked"]
    chk = st.get("check")
    if not chk or chk.get("state") != "passed" or chk.get("sig") != al.stack_sig():
        return None, "общей проверки для текущего состава стека нет (или она не пройдена): шеф поставит её сам, когда накопится " + str(_cfg().get("chief_stack_check_every", 2)) + " гипотез; состояние: `стек`"
    for k in ("pending", "running"):
        if any(f.stem.startswith("stk_") for f in (al.EXP / "queue" / k).glob("stk_*.yaml")):
            return None, "проверка стека ещё идёт"
    repo = al.repo_root()
    if repo.name != "autolab_repo":
        return None, "accept works only in the host copy ~/harness-lab/autolab_repo"
    if _git(repo, "status", "--porcelain").stdout.strip():
        return None, "the host copy has uncommitted changes"
    if _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != "main":
        return None, "the host copy is not on main"
    root = al.status_root()
    if not astat.index(root):
        return None, "схема статусов (status/index.json) не найдена на хосте"
    cells: list[str] = []
    for i in its:
        try:
            plan = al.status_plan(i["name"], accept=True)
            astat.dry_run(root, plan)
        except ValueError as ex:
            return None, f"статус ячеек гипотезы {i['name']} не обновить: {ex}"
        cells += [x for x in (i.get("cells") or []) if x not in cells]
    head = _git(al.stack_wt(), "rev-parse", "--short", "HEAD").stdout.strip()
    stat = _git(repo, "diff", "--stat", f"HEAD...{STACK}").stdout.strip() or "кода нет: только запись результатов в схему статусов"
    return {"names": [i["name"] for i in its], "commit": head, "branch": STACK, "cells": cells, "stat": stat,
            "check": f"{chk['name']}: вердикт {chk.get('verdict')}, разность {chk.get('mean', 0):+.1f}", "no_code": all(i.get("no_code") for i in its)}, ""


def accept_stack_do(uid, info: dict) -> str:
    info2, why = accept_stack_preflight()
    if not info2 or info2["commit"] != info["commit"] or info2["names"] != info["names"]:
        return f"accept refused at the last check: {why or 'the stack changed since the button was shown'}"
    repo = al.repo_root()
    before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    merged = "кода для слияния нет"
    if not info2["no_code"]:
        snap = al.snapshot_protected()
        r = _git(repo, "merge", "--no-ff", "-m", f"autolab accept stack ({', '.join(info2['names'])})", STACK)
        if r.returncode:
            _git(repo, "merge", "--abort")
            return "слияние не удалось и отменено:\n" + (r.stdout + r.stderr)[-500:]
        if al.snapshot_protected() != snap:
            _git(repo, "reset", "--hard", before)
            return "слияние изменило защищённые файлы: откат"
        merged = f"влито в main хостовой копии ({_git(repo, 'rev-parse', '--short', 'HEAD').stdout.strip()})"
    stat_txt, stale_all = [], []
    for nm in info2["names"]:
        try:
            plan = al.status_plan(nm, accept=True)
            astat.apply(al.status_root(), plan)
            for it in plan:
                astat.events_add(al.EXP, {"name": nm, "item": it})
            al.record_journal(nm, accepted="да (в составе стека, код влит)")
        except ValueError as ex:
            stat_txt.append(f"{nm}: схема НЕ обновлена: {ex}")
        al.set_meta(nm, stage="accepted", merged_by=str(uid), merged_at=time.strftime("%Y-%m-%d %H:%M:%S"), in_stack=False, auto_branch=None)
        stale_all += al.set_decision(nm, "accepted", "", str(uid))
    pinfo = {"name": "stack", "branch": STACK, "commit": info2["commit"], "no_code": info2["no_code"], "cells": info2["cells"]}
    pushed = c.push_master(pinfo, before)
    reset = stack_reset("принят в main")
    return f"{merged}; схема: {', '.join(info2['cells'])}; {pushed}" + ("\n" + "; ".join(stat_txt) if stat_txt else "") + \
        (f"\n♻️ Устарели отложенные гипотезы тех же типов: {', '.join(sorted(set(stale_all)))}" if stale_all else "") + "\n" + reset


def stack_text() -> str:
    st = al.stack_load()
    its = _items(st)
    chk, abl = st.get("check"), st.get("ablation")
    lines = [f"📚 Стек {STACK}: " + (f"{len(its)} гипотез" if its else "пуст") + (f" · ЗАБЛОКИРОВАН: {st['blocked']}" if st.get("blocked") else "")]
    for n, i in enumerate(its, 1):
        lines.append(f"{n}. {i['name']} — ячейки {', '.join(i.get('cells') or []) or '-'}; переключатели {i.get('env') or '—'}; {'без кода' if i.get('no_code') else 'код ' + i.get('commit', '')}")
    if chk:
        lines.append(f"Общая проверка: {chk['name']} — {chk['state']}" + (f", разность {chk.get('mean', 0):+.1f}" if chk.get("mean") is not None and chk["state"] != "running" else "")
                     + ("" if chk.get("sig") == al.stack_sig() else " (для другого состава: устарела)"))
    elif its:
        lines.append(f"Общей проверки ещё нет (каждые {_cfg().get('chief_stack_check_every', 2)} добавления)")
    if abl:
        lines.append(f"Поиск виновника: {abl['name']} — {abl['state']}")
    lines.append("Последнее: " + ("; ".join(st.get("history", [])[-4:]) or "—"))
    lines.append("В main и на master стек попадает только после вашего `принять стек` (нужна пройденная общая проверка). `стек сбросить` — выбросить стек без принятия.")
    return "\n".join(lines)


def auto_accept(name: str) -> tuple[bool, str]:
    return stack_add(name)


# ------------------------------------------------------------------ time: the deadline of the experiments
def _ts(text: str) -> float | None:
    try:
        return time.mktime(time.strptime(str(text).strip(), "%Y-%m-%d %H:%M"))
    except (ValueError, TypeError):
        return None


def deadlines() -> tuple[float | None, float | None]:
    cfg = al.limits_cfg()
    return _ts(cfg.get("chief_experiments_until")), _ts(cfg.get("chief_final_deadline"))


def median_run_min() -> float:
    """Median minutes an experiment takes on this host (manifest -> cells of the last 25 results); 15 when there is no history."""
    rows = []
    for d in al.EXP.glob("results/*"):
        m, f = d / "manifest.json", d / "cells.jsonl"
        if m.exists() and f.exists():
            rows.append((f.stat().st_mtime, (f.stat().st_mtime - m.stat().st_mtime) / 60))
    xs = sorted(x for _, x in sorted(rows)[-25:] if 0.5 < x < 240)
    return xs[len(xs) // 2] if xs else 15.0


def time_left_h(which: int = 0) -> float | None:
    t = deadlines()[which]
    return None if t is None else (t - time.time()) / 3600


def time_text() -> str:
    until, final = deadlines()
    if until is None:
        return "TIME: no deadline is set (chief_experiments_until)."
    med = median_run_min()
    lu = (until - time.time()) / 3600
    out = (f"TIME: the experiments end {time.strftime('%Y-%m-%d %H:%M', time.localtime(until))} host time ({lu:.1f} h left)"
           + (f"; everything ends {time.strftime('%Y-%m-%d %H:%M', time.localtime(final))} ({(final - time.time()) / 3600:.1f} h left): the hours after the experiments are for the final runs on main and the hand-in" if final else "")
           + f". An experiment takes about {med:.0f} min here (median of the last runs), the whole path of a hypothesis (verdict, review, a check on other data, a second budget) several hours with the queue. ")
    if lu <= 0:
        return out + "PHASE: over. Start nothing."
    if lu <= float(al.limits_cfg().get("chief_final_check_h", 3)) + 1:
        return out + "PHASE: closing. No new proposals; finish what is in flight, get the stack through a whole-stack check so the human can `принять стек`."
    if lu <= 10:
        return out + "PHASE: converging. Prefer confirming the best candidates (a check on other data, a second budget) and stacking them over new ideas."
    return out + "PHASE: exploring. Keep the queue full; breadth is still worth it."


def _time_refusal(kind: str) -> str:
    """"" or why a new run is not started any more (not enough time to finish it before the experiments end)."""
    lu = time_left_h()
    if lu is None:
        return ""
    need = max(3 * median_run_min(), 20) / 60
    if lu < need:
        return f"not enough time: {max(lu, 0) * 60:.0f} min before the end of the experiments, a run takes about {median_run_min():.0f} min (need {need * 60:.0f})"
    return ""


# ------------------------------------------------------------------ the context of a tick
def _read(p: Path, limit: int) -> str:
    try:
        return p.read_text(encoding="utf-8")[:limit]
    except OSError:
        return ""


def role_text() -> str:
    for f in (HERE.parent / "autolab_chief.md", al.LAB / "harness" / "autolab_chief.md"):
        t = _read(f, 20000)
        if t:
            return t
    return "You are the chief of the harness autolab. (the role file autolab_chief.md is missing: plan conservatively, prefer wait)"


def parallel_text() -> str:
    """The parallelism of the runs and what could show it is too much: aborts of the last 12 hours, the gateway abuse cooldown."""
    bad = []
    for f in al.EXP.glob("results/*/ABORTED.txt"):
        if time.time() - f.stat().st_mtime < 12 * 3600:
            bad.append(f"{f.parent.name}: {_read(f, 120).strip()}")
    ab = ""
    try:
        gw = json.loads((al.LAB / "dispatcher_state.json").read_text(encoding="utf-8")).get("gateway") or {}
        ab = f"; gateway abuse cooldown {gw.get('abuse')} s, day {gw.get('day_pct')}%, week {gw.get('week_pct')}%"
    except (OSError, ValueError):
        pass
    return (f"PARALLEL RUNS: {al.parallel_runs_now()} runs of one experiment at once (action `parallel` with n 1..{al.PARALLEL_MAX} changes it for experiments queued from now on){ab}. "
            + ("Aborts in the last 12 h: " + " | ".join(bad[:4]) + ". If they look like gateway 429 / abuse / broken cells from load, lower it; raise it again only after clean experiments." if bad else "No aborts in the last 12 h."))


def _l4() -> str:
    for f in (HERE.parent / "harness-gap.md", HERE.parent.parent / "harness-gap.md", al.LAB / "harness-gap.md"):
        t = _read(f, 400000)
        i = t.find("## L4")
        if i >= 0:
            j = t.find("\n## ", i + 5)
            return t[i:(j if j > 0 else i + 3000)][:3000]
    return "(harness-gap.md is not available on this host)"


def _lab_failure(b: str) -> str:
    blk = al.done_block(b)
    if blk is not None and blk.get("status") != "ok" and not (al.meta(b).get("verdict") or al.meta(b).get("result")):
        return f" (LAB {blk.get('status')}: {str(blk.get('reason') or '')[:140]}; it holds nothing; do not rerun it the same way)"
    return ""


def hyp_table(max_rows: int = 14) -> str:
    rows = []
    bases = sorted({al.family_base(n) for n in _names()}, key=lambda b: -al.meta(b).get("queued_at", 0))
    for b in bases:
        m = al.meta(b)
        dec = al.decision_of(b)
        if dec in DECIDED and time.time() - m.get("decided_at", 0) > 2 * 86400:
            continue
        stt = _arm_stats(b)
        ver = ",".join(f"{a}:{v}" for a, v in (m.get("verdict") or {}).items()) or "-"
        diff = f"{stt[0]} {stt[1].get('mean', 0):+.1f} CI {stt[1].get('ci95')} thr {stt[1].get('threshold')}" if stt else "-"
        chk = "; ".join(f"{n} set {_set_of(n)} budget {_budget(n)} verdict {','.join((al.meta(n).get('verdict') or {}).values()) or 'running'}"
                        + (f" diff {_arm_stats(n)[1].get('mean', 0):+.1f}" if _arm_stats(n) else "") for n in family(b) if n != b) or "none"
        pol = ""
        if (m.get("verdict") or {}) and any(v == "accept" for v in m["verdict"].values()) and not dec:
            try:
                ok, why = accept_policy(b)
                pol = "ACCEPT ALLOWED" if ok else "accept refused: " + "; ".join(why)
            except Exception as ex:  # noqa: BLE001
                pol = f"policy error {type(ex).__name__}"
        rows.append(f"- {b} [{al.mode_of(b)}] cells {','.join(al.targets_of(b)) or '-'}; set {_set_of(b)} budget {_budget(b)} repeats {al.spec_of(b).get('repeats')}; "
                    f"state {c._state_ru(b)}{_lab_failure(b)}; verdict {ver}; diff {diff}; review {m.get('review_recommendation') or 'none'} "
                    f"(data: {str(m.get('review_data_set') or '-')}); decision {dec or 'open'}{' STALE' if m.get('stale') else ''}"
                    f"{'; ON THE STACK' if m.get('in_stack') else ''}; checks/repeats: {chk}" + (f"; {pol}" if pol else ""))
        if len(rows) >= max_rows:
            break
    return "\n".join(rows) or "(no hypotheses yet)"


def recent_journal(n: int = 24) -> str:
    root = al.status_root()
    out = []
    try:
        for e in astat.index(root)["entries"]:
            for r in astat.journal_rows(root, f"{e['block']}/{e['type']}", 2):
                out.append((r.get("date", ""), f"{e['block']}/{e['type']}: {r['name']} {r['verdict']} {r['result'][:140]}"))
    except Exception:  # noqa: BLE001
        return "(journal unavailable)"
    seen, lines = set(), []
    for d, t in sorted(out, reverse=True):
        key = t.split(": ", 1)[1].split(" ", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {d} {t}")
        if len(lines) >= n:
            break
    return "\n".join(lines) or "(empty)"


def queue_text() -> str:
    rows = []
    for k in ("running", "pending"):
        for f in sorted((al.EXP / "queue" / k).glob("*.yaml")):
            rows.append(f"{k}: {f.stem}")
    return "; ".join(rows) or "empty"


def fresh_reports(n: int = 2) -> str:
    fs = sorted(al.DRAFTS.glob("*.verdict.md"), key=lambda f: -f.stat().st_mtime)[:n]
    return "\n".join(f"### {f.stem}\n{_read(f, 900)}" for f in fs) or "(none)"


def build_context() -> str:
    cfg = al.limits_cfg()
    root = al.status_root()
    st = _state()
    parts = [
        time_text(),
        parallel_text(),
        f"NOW {time.strftime('%Y-%m-%d %H:%M')} host time. goal_mode (focus) = {cfg.get('goal_mode') or 'alternate'}; dry_run = {cfg.get('chief_dry_run', True)}.",
        f"LIMITS: max_actions per tick {cfg['chief_max_actions']}; active hypotheses {len({al.family_base(v) for v in al.active_types().values()})}/{cfg['chief_max_active']}; "
        f"queue depth {al.queue_depth()}/{cfg['max_queue_depth']}; rechecks per hypothesis {cfg['chief_max_rechecks']}; Claude dollars today {al.day_usd():.2f}/{cfg['usd_per_day']}, "
        f"this week {al.week_usd():.2f}/{cfg['usd_per_week']}; gateway requests this week {al.gw_week_used()}; actions today {st['actions_today']}.",
        "RATING by mode (scheme of statuses):\n" + "\n".join(astat.prompt_block(root, 5, 3, m) for m in astat.MODES) if astat.index(root) else "(no scheme)",
        "PRIORITY.md (head):\n" + _read(root / "PRIORITY.md", 1800),
        "HYPOTHESES (open and decided in the last 2 days):\n" + hyp_table(),
        "RECENT JOURNAL ROWS of the cells:\n" + recent_journal(),
        "QUEUE (stand experiments are not yours): " + queue_text(),
        "FRESH VERDICTS:\n" + fresh_reports(),
        "SETS for checks on other data (dev/fresh, never holdout): " + "; ".join(f"{k}: {str(v)[:90]}" for k, v in list(al.check_sets().items())[:30]),
        "STACK (accepted hypotheses accumulate here; new hypotheses are measured on top of it; only a human `принять стек` reaches main):\n" + stack_text(),
        "harness-gap.md L4:\n" + _l4(),
        "YOUR PLAN FROM THE PREVIOUS TICK (CHIEF_PLAN.md):\n" + (_read(plan_file(), 3500) or "(none yet: write it)"),
        "YOUR NOTES: " + (" | ".join(st["notes"]) or "(none)"),
    ]
    return "\n\n".join(parts)


# ------------------------------------------------------------------ plan parsing and validation
def parse_plan(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    dec = json.JSONDecoder()
    i = text.find("{")
    while i >= 0:
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict) and isinstance(obj.get("actions"), list):
                return obj
        except ValueError:
            pass
        i = text.find("{", i + 1)
    return None


def validate_action(a) -> str | None:
    if not isinstance(a, dict) or a.get("action") not in ACTIONS:
        return f"unknown action {a.get('action') if isinstance(a, dict) else a!r}"
    kind = a["action"]
    if not str(a.get("why") or "").strip():
        return "every action needs a `why`"
    if kind in ("check", "rerun", "review", "accept", "reject"):
        nm = c._safe_name(a.get("name") or "")
        if not nm or not (al.DRAFTS / f"{nm}.yaml").exists():
            return f"no hypothesis {a.get('name')!r}"
    if kind == "propose":
        root = al.status_root()
        cell = str(a.get("cell") or "")
        if astat.index(root):
            ok, msg = astat.check_target(root, cell)
            if not ok:
                return msg
        if len(str(a.get("idea") or "")) < 20:
            return "propose needs an `idea` (a few sentences)"
        if a.get("mode") not in (None, "accuracy", "economy", "both"):
            return "mode must be accuracy, economy or both"
    if kind == "parallel":
        n = a.get("n")
        if not isinstance(n, int) or not 1 <= n <= al.PARALLEL_MAX:
            return f"n must be an integer 1..{al.PARALLEL_MAX}"
    if kind == "rerun":
        n = a.get("repeats")
        if not isinstance(n, int) or not 2 <= n <= al.limits_cfg()["max_rerun_repeats"]:
            return f"repeats must be 2..{al.limits_cfg()['max_rerun_repeats']}"
    if kind == "check":
        b = a.get("budget")
        if b is not None and (not isinstance(b, int) or not 10000 <= b <= 100000):
            return "budget must be an integer 10000..100000"
        if a.get("set") and a["set"] not in al.check_sets():
            return f"no set {a['set']} (holdout is never used)"
    return None


def _gate_ok() -> tuple[bool, str]:
    try:
        import dispatcher
        return dispatcher.exp_gate(dispatcher.load_cfg(), dispatcher.fractions(al.sr.limits()))
    except Exception as ex:  # noqa: BLE001
        return False, f"gateway limits unreadable ({type(ex).__name__})"


# ------------------------------------------------------------------ executing one action
def execute(a: dict, dry: bool, st: dict, gate: tuple[bool, str]) -> str:
    kind = a["action"]
    cfg = al.limits_cfg()
    if kind in ("wait", "note"):
        if kind == "note" and not dry:
            st["notes"].append(str(a.get("text") or a.get("why"))[:300])
        return "ok"
    if kind == "parallel":
        old = al.parallel_runs_now()
        if dry:
            return f"dry-run: would set parallel runs {old} -> {a['n']}"
        (al.DRAFTS / "parallel.json").write_text(json.dumps({"n": a["n"], "ts": time.strftime("%Y-%m-%d %H:%M"), "why": str(a.get("why"))[:200]}), encoding="utf-8")
        return f"parallel runs {old} -> {al.parallel_runs_now()} (for experiments queued from now on)"
    name = al.family_base(c._safe_name(a.get("name") or "") or "") if a.get("name") else ""
    if kind in ("propose", "check", "rerun"):
        tr = _time_refusal(kind)
        if tr:
            return "refused: " + tr
        if al.queue_depth() >= cfg["max_queue_depth"]:
            return f"refused: queue is full ({al.queue_depth()}/{cfg['max_queue_depth']})"
        if not gate[0] and not dry:
            return f"refused: gateway gate: {gate[1]}"
    if kind == "propose":
        if len({al.family_base(v) for v in al.active_types().values()}) >= cfg["chief_max_active"]:
            return f"refused: {cfg['chief_max_active']} hypotheses are already active"
        taken = al.active_types()
        cell = a["cell"]
        conf = al.type_conflicts([cell], taken)
        if conf:
            return "refused: task types are held by " + ", ".join(sorted(set(conf.values())))
        if dry:
            return f"dry-run: would propose for {cell} [{a.get('mode') or cfg.get('goal_mode') or 'accuracy'}]"
        mode = a.get("mode") if a.get("mode") in ("accuracy", "economy") else (cfg.get("goal_mode") or "accuracy")
        before = set(_names())
        try:
            al.cmd_cycle(1, "chief: " + str(a["idea"])[:500], None, False, [cell], True, mode, {cell: mode})
        except SystemExit as ex:
            return f"failed: {al.scrub(str(ex))[:200]}"
        except Exception as ex:  # noqa: BLE001
            return f"failed: {type(ex).__name__}: {al.scrub(str(ex))[:200]}"
        new = [n for n in sorted(set(_names()) - before) if al.meta(n).get("stage") in ("queued", "verdict")]
        for nm in new:
            c._spawn(["await", nm], "await")
        return ("queued " + ", ".join(new)) if new else "nothing passed the validation, the simulator and the dedupe"
    if kind in ("check", "rerun") and rechecks(name) >= cfg["chief_max_rechecks"]:
        return f"refused: {name} already had {rechecks(name)} checks/repeats (limit {cfg['chief_max_rechecks']})"
    if kind == "check":
        ds = a.get("set") or al.meta(name).get("review_data_set")
        if not ds or ds not in al.check_sets():
            return "refused: no set named and the reviewer named none"
        args = {"name": name, "set": ds}
        if a.get("budget"):
            args["budget"] = int(a["budget"])
        if dry:
            return f"dry-run: would check {name} on {ds}" + (f" at {args['budget']}" if "budget" in args else "")
        return c.do_check(UID, {"args": args})
    if kind == "rerun":
        if dry:
            return f"dry-run: would rerun {name} x{a['repeats']}"
        return c.do_rerun(UID, {"args": {"name": name, "repeats": a["repeats"]}})
    if kind == "review":
        if not (al.meta(name).get("verdict") or al.meta(name).get("result")):
            return "refused: no verdict to review yet"
        if dry:
            return f"dry-run: would order a review of {name}"
        return c.do_review(UID, {"args": {"name": name, "chat": _chat()}})
    if kind == "accept":
        ok, why = accept_policy(name)
        if dry:
            return ("dry-run: the policy would ALLOW adding it to the stack " + STACK) if ok else "dry-run: the policy REFUSES: " + "; ".join(why)
        good, msg = auto_accept(name)
        return msg if good else "refused: " + msg
    if kind == "reject":
        cur = al.decision_of(name)
        if cur:
            return f"refused: {name} is already decided ({cur})"
        if dry:
            return f"dry-run: would reject {name}"
        return c.decide(UID, [name], "rejected")["text"]
    return "unknown"


# ------------------------------------------------------------------ the tick
def _plan_md(plan: str, st: dict, dry: bool) -> str:
    head = f"# CHIEF_PLAN (updated {time.strftime('%Y-%m-%d %H:%M')}{', DRY-RUN' if dry else ''})\n\n"
    notes = ("\n\n## Notes of the chief\n" + "\n".join(f"- {n}" for n in st["notes"])) if st["notes"] else ""
    return head + plan.strip() + notes + "\n"


def _limit_stop(st: dict, why: str) -> str:
    if st.get("spent_notice") != st["day"]:
        st["spent_notice"] = st["day"]
        _say(f"🧭 Шеф остановился: {why}. Продолжит сам завтра/в новой неделе или когда вы поднимете лимит (autolab.yaml: usd_per_day / usd_per_week).")
    return f"chief: stopped: {why}"


ACTION_RU = {"parallel": ("⚙️", "Параллельность прогонов"), "propose": ("💡", "Предложить гипотезу"), "check": ("🔬", "Проверка на других данных"), "rerun": ("🔁", "Повтор"), "review": ("📝", "Ревью"),
             "accept": ("✅", "В стек"), "reject": ("❌", "Отклонить"), "wait": ("⏳", "Ждать"), "note": ("🗒", "Заметка")}
RESULT_RU = ((r"^dry-run: would order a review of .*", "заказал бы ревью"),
             (r"^dry-run: would check \S+ on (\S+)( at (\d+))?", lambda m: f"поставил бы проверку на {m.group(1)}" + (f", бюджет {m.group(3)}" if m.group(3) else "")),
             (r"^dry-run: would rerun \S+ x(\d+)", lambda m: f"поставил бы повтор ×{m.group(1)}"), (r"^dry-run: would reject .*", "отклонил бы"), (r"^dry-run: would set parallel runs (\d+) -> (\d+)", lambda m: f"поставил бы параллельность {m.group(1)} → {m.group(2)}"),
             (r"^dry-run: would propose for (\S+)", lambda m: f"предложил бы гипотезу для {m.group(1)}"), (r"^dry-run: the policy would ALLOW.*", "политика разрешила бы добавить в стек"),
             (r"^dry-run: the policy REFUSES: ", "политика отказала бы: "), (r"^refused: ", "отказ: "), (r"^invalid: ", "неверное действие: "), (r"^failed: ", "ошибка: "),
             (r"^queued ", "поставлено в очередь: "), (r"^ok$", "ок"))


def _short(text: str, limit: int = 260) -> str:
    """The first sentences that fit in `limit` characters (never cut in the middle of a number)."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    k = max(cut.rfind(". "), cut.rfind("; "))
    return cut[:k + 1] if k > limit // 3 else cut.rsplit(" ", 1)[0] + "…"


def _fmt_action(n: int, desc: str, nm: str, why: str, res: str) -> str:
    icon, label = ACTION_RU.get(desc, ("•", desc))
    for pat, rp in RESULT_RU:
        if re.match(pat, res):
            res = re.sub(pat, rp, res, count=1)
            break
    return f"{n}. {icon} {label}" + (f"\n   {nm}" if nm else "") + f"\n   Почему: {_short(why)}\n   Итог: {_short(res, 220)}"


INSTANT = ("reject", "accept", "note", "wait")           # done at once: the next thought may follow in the same tick
BLOCKING = ("propose", "check", "rerun", "review")        # something has to run (a review, an experiment): the result wakes the next tick


def _world_sig() -> str:
    """What a tick would want to know changed: hypotheses (stage, verdict, review, decision, stack), the queue, the stack checks. The chief's own actions are
    folded in at the end of its tick, so only what happens later (a verdict, a review, a free slot) wakes it before the heartbeat."""
    import hashlib
    parts = []
    for n in _names():
        m = al.meta(n)
        parts.append((n, m.get("stage"), bool(m.get("verdict")), m.get("review_recommendation"), al.decision_of(n), bool(m.get("in_stack")), bool(m.get("stale"))))
    q = sorted(f.stem for k in ("pending", "running") for f in (al.EXP / "queue" / k).glob("*.yaml"))
    sk = al.stack_load()
    parts.append((tuple(q), (sk.get("check") or {}).get("state"), (sk.get("ablation") or {}).get("state"), bool(sk.get("blocked")), len(_items(sk))))
    return hashlib.md5(repr(parts).encode()).hexdigest()


def _round(st: dict, dry: bool, gate: tuple[bool, str], lines: list[str]) -> tuple[str, dict | None, int, bool, bool]:
    """One model call and its actions. Returns (error, plan, executed, blocking_started, replan): replan when the world changed at once (a reject, an accept)."""
    cfg = al.limits_cfg()
    try:
        raw = al.llm(role_text(), build_context(), int(cfg["chief_max_tokens"]), "chief")
    except SystemExit as ex:
        msg = al.scrub(str(ex))
        if "limit" in msg and ("USD" in msg):
            return _limit_stop(st, msg[:160]), None, 0, False, False
        out = f"chief: the model did not answer: {msg[:200]}"
        al.log(out)
        return out, None, 0, False, False
    plan = parse_plan(raw)
    if not plan:
        al.log("chief: the answer was not a JSON plan")
        return "chief: the answer was not a JSON plan: nothing done", None, 0, False, False
    done, blocking, replan = 0, False, False
    for a in plan["actions"][: int(cfg["chief_max_actions"])]:
        err = validate_action(a)
        if err:
            res = "invalid: " + err
        else:
            try:
                res = execute(a, dry, st, gate)
            except SystemExit as ex:
                res = f"failed: {al.scrub(str(ex))[:200]}"
            except Exception as ex:  # noqa: BLE001
                res = f"failed: {type(ex).__name__}: {al.scrub(str(ex))[:200]}"
        okres = not err and not res.startswith(("refused", "invalid", "failed"))
        kind = a.get("action") if isinstance(a, dict) else "?"
        if okres and not dry and kind not in ("wait", "note"):
            done += 1
            if kind in BLOCKING:
                blocking = True
            if kind in ("accept", "reject"):
                replan = True
        nm = (a.get("name") or a.get("cell") or "") if isinstance(a, dict) else ""
        why = str(a.get("why") if isinstance(a, dict) else "")
        c.journal(UID, "chief", action=kind, name=nm, why=why[:200], result=res[:200], dry=dry)
        lines.append(_fmt_action(len(lines) + 1, kind, nm, why, res))
        st["log"].append({"ts": time.strftime("%Y-%m-%d %H:%M"), "action": kind, "name": nm, "result": res[:120], "dry": dry})
        if okres and not dry and kind == "accept":      # the stack changed: the rest of this plan was made for the old one
            break
    return "", plan, done, blocking, replan


def cmd_chief(force: bool = False) -> str:
    cfg = al.limits_cfg()
    st = _state()
    if not is_on():
        return "chief: off (a human switches it on with `шеф вкл`)"
    if al.STOP_FILE.exists() or (al.EXP / "queue" / "STOP").exists():
        return "chief: stop flag: nothing done"
    sig = _world_sig()
    if not force:
        gap = time.time() - st["last_tick"]
        if gap < cfg["chief_event_gap_min"] * 60:
            return "chief: too early for the next tick"
        if gap < cfg["chief_min_gap_min"] * 60 and sig == st.get("sig"):
            return "chief: nothing changed since the last tick (heartbeat in " + str(int(cfg["chief_min_gap_min"] - gap / 60)) + " min)"
    if al.day_usd() >= cfg["usd_per_day"] * 0.97:
        out = _limit_stop(st, f"суточный лимит Claude {al.day_usd():.2f} из {cfg['usd_per_day']} $")
        _save(st)
        return out
    if al.week_usd() >= cfg["usd_per_week"] * 0.97:
        out = _limit_stop(st, f"недельный лимит Claude {al.week_usd():.2f} из {cfg['usd_per_week']} $")
        _save(st)
        return out
    dry = dry_run()
    st["last_tick"] = time.time()
    lines: list[str] = []
    lu = time_left_h()
    if lu is not None and lu <= 0:
        for t in (stack_housekeeping(dry) if _items() else []):
            lines.append("• " + t)
        if st.get("over_notice") != "1":
            st["over_notice"] = "1"
            _say("🧭 Время экспериментов вышло: шеф ничего нового не запускает. " + ("Стек: " + str(len(_items())) + " гипотез. Принять: `принять стек` (если общая проверка пройдена), затем финальные прогоны на main." if _items() else "Стек пуст."))
        if lines:
            _say("🧭 Шеф · стек\n" + "\n".join(lines))
        st["sig"] = _world_sig()
        _save(st)
        return "chief: the time of the experiments is over: nothing started"
    try:
        for t in stack_housekeeping(dry):
            lines.append("• " + t)
            c.journal(UID, "chief", action="stack", name="", why=t[:200], result="", dry=dry)
    except Exception as ex:  # noqa: BLE001 - a broken stack must not stop the planning
        lines.append(f"• стек: ошибка {type(ex).__name__}: {al.scrub(str(ex))[:160]}")
    gate = _gate_ok()
    done_all, rounds, last_plan, err = 0, 0, None, ""
    while rounds < int(cfg["chief_max_rounds"]):
        rounds += 1
        err, plan, done, blocking, replan = _round(st, dry, gate, lines)
        if plan:
            last_plan = plan
        done_all += done
        if err or blocking or not replan or dry:       # an experiment or a review is running (its result wakes the next tick), or nothing changed at once
            break
        gate = _gate_ok()
    st["actions_today"] += done_all
    st["ticks_today"] += 1
    if last_plan and last_plan.get("plan"):
        try:
            plan_file().write_text(_plan_md(str(last_plan["plan"]), st, dry), encoding="utf-8")
        except OSError as ex:
            al.log(f"chief: CHIEF_PLAN.md not written: {ex}")
    st["sig"] = _world_sig()
    _save(st)
    if lines:
        _say(f"🧭 Шеф · {'dry-run, ничего не запущено' if dry else 'боевой режим'} · {len(lines)} {'действие' if len(lines) == 1 else 'действия' if 2 <= len(lines) <= 4 else 'действий'}\n\n" + "\n\n".join(lines))
    _maybe_digest(st)
    return err or f"chief: {len(lines)} actions ({'dry-run' if dry else 'executed ' + str(done_all)}), {rounds} round(s)"


# ------------------------------------------------------------------ views for the bot
def status_text() -> str:
    cfg = al.limits_cfg()
    st = _state()
    last = st["log"][-5:]
    return (f"🧭 Шеф: {'включён ✅' if is_on() else 'выключен'}" + (" · режим dry-run (ничего не запускает)" if dry_run() else " · боевой режим") + "\n"
            f"Тиков сегодня: {st['ticks_today']}, действий исполнено: {st['actions_today']}. Следующий тик не раньше, чем через {cfg['chief_min_gap_min']} мин после предыдущего.\n"
            f"Claude: сегодня {al.day_usd():.2f} из {cfg['usd_per_day']} $, за неделю {al.week_usd():.2f} из {cfg['usd_per_week']} $.\n"
            f"Лимиты: действий за тик {cfg['chief_max_actions']}, активных гипотез {cfg['chief_max_active']}, перепроверок на гипотезу {cfg['chief_max_rechecks']}.\n"
            f"Стек: {len(_items())} гипотез (`стек`).\n"
            "Последние действия:\n" + ("\n".join(f"• {r['ts']} {r['action']} {r['name']} → {r['result']}" + (" [dry]" if r.get("dry") else "") for r in last) or "• нет")
            + "\nКоманды: шеф вкл (спросит подтверждение), шеф выкл, шеф план, шеф дайджест")


def plan_text() -> str:
    return _read(plan_file(), 3800) or "План шефа ещё не написан: он появится после первого тика."


def digest_text() -> str:
    st = _state()
    today = [r for r in st["log"] if r["ts"].startswith(st["day"])]
    acc = [n for n in _names() if al.meta(n).get("auto_branch") and time.strftime("%Y-%m-%d", time.localtime(al.meta(n).get("auto_accepted_at", 0))) == st["day"]]
    return (f"🧭 Дайджест шефа за {st['day']}: тиков {st['ticks_today']}, действий в журнале {len(today)}, исполнено {st['actions_today']}"
            + (" (dry-run)" if dry_run() else "") + f"\nClaude: {al.day_usd():.2f} из {al.limits_cfg()['usd_per_day']} $ за сутки, {al.week_usd():.2f} из {al.limits_cfg()['usd_per_week']} $ за неделю.\n"
            + ("Добавлено в стек: " + ", ".join(acc) + " (в main только после `принять стек`)\n" if acc else "")
            + "\n".join(f"• {r['ts'][11:]} {r['action']} {r['name']} → {r['result']}" for r in today[-12:]))


def _maybe_digest(st: dict) -> None:
    if time.localtime().tm_hour >= int(al.limits_cfg().get("chief_digest_hour", 21)) and st.get("digest_day") != st["day"]:
        st["digest_day"] = st["day"]
        _save(st)
        _say(digest_text())


# ------------------------------------------------------------------ selftest
def selftest() -> None:
    import tempfile
    import yaml
    tmp = Path(tempfile.mkdtemp())
    c.CFG_DIR = tmp / "cfg"
    al.CFG_DIR = c.CFG_DIR
    al.EXP = tmp / "exp"
    al.DRAFTS = al.EXP / "autolab"
    al.USAGE = al.DRAFTS / "usage.jsonl"
    al.STOP_FILE = tmp / "AUTOLAB_STOP"
    al.LAB = tmp / "lab"
    for k in ("pending", "done", "running"):
        (al.EXP / "queue" / k).mkdir(parents=True)
    (al.EXP / "sets").mkdir()
    al.DRAFTS.mkdir(parents=True)
    c.CFG_DIR.mkdir(parents=True, exist_ok=True)
    (c.CFG_DIR / "telegram_allow.json").write_text('{"user_ids": [42], "chat_ids": [42]}')
    os.environ["AUTOLAB_STATUS_ROOT"] = str(tmp / "status")
    astat.mini(tmp / "status")
    for n, body in (("mixed100_dev", "role: dev\ntasks: {math: 4}\nseed: 1\n"), ("dev_math", "role: dev\ntasks: {math: 4}\nseed: 2\n"), ("hold_math", "role: holdout\ntasks: {math: 4}\nseed: 3\n")):
        (al.EXP / "sets" / f"{n}.yaml").write_text(body)
    spawned = []
    c._spawn = lambda argv, tag: spawned.append(argv) or "started (mock)"
    gate_ok = (True, "ok")
    globals()["_gate_ok"] = lambda: gate_ok

    # a hypothesis with a lab accept on the dev set, a review, no check yet
    d = {"name": "ab_win", "hypothesis": "h", "set": "mixed100_dev", "model": "auto", "repeats": 2, "budget": 30000,
         "arms": {"a": {}, "b": {"env": {"HARNESS_SORTER": "fifo"}}}, "target_cells": ["blk/a"], "why_cell": "w"}
    al.write_draft(d)
    res_ok = lambda mean, thr=7.0: {"a": {}, "b": {"points_diff_vs_first": {"mean": mean, "ci95": [1.0, mean * 2], "threshold": thr}}}
    al.set_meta("ab_win", stage="verdict", sim_verdict="pass", verdict={"b": "accept"}, result=res_ok(12.0), review_recommendation="принять", target_cells=["blk/a"], queued_at=time.time())
    info = {"name": "ab_win", "no_code": True, "branch": "", "commit": "", "files": [], "stat": "", "verdict": {"b": "accept"}, "review": "принять", "cells": ["blk/a"],
            "plan": [{"cell": "blk/a", "evidence": "ab_win: accept", "experiments": "ab_win", "date": "2026-11-11", "mode": "accuracy"}]}
    c.accept_preflight = lambda name, against="HEAD": (dict(info), "")
    ok, why = accept_policy("ab_win")
    assert not ok and any("no check on a fresh or narrow set" in w for w in why) and any("second budget" in w for w in why), why      # no check on other data -> refused

    # a check on another set at the same budget: still no second budget
    def add_check(name, ds, budget, verdict, mean, thr=7.0):
        al.write_draft(dict(d, name=name, set=ds, budget=budget))
        al.set_meta(name, stage="verdict", check_of="ab_win", verdict={"b": verdict}, result=res_ok(mean, thr))
    add_check("ab_win_fresh", "dev_math", 30000, "clarify", 3.0)
    ok, why = accept_policy("ab_win")
    assert not ok and not any("no check on a fresh" in w for w in why) and any("second budget" in w for w in why), why
    # a check that says reject refuses
    add_check("ab_win_fresh2", "dev_math", 30000, "reject", -9.0)
    ok, why = accept_policy("ab_win")
    assert any("says reject" in w for w in why), why
    # reject check replaced by a good second budget
    for f in ("ab_win_fresh2.yaml", "ab_win_fresh2.meta.json"):
        (al.DRAFTS / f).unlink()
    add_check("ab_win_fresh3", "dev_math", 50000, "clarify", -2.0, 7.0)      # loss within the threshold counts as "no loss beyond"
    ok, why = accept_policy("ab_win")
    assert ok, why
    # (4) the reviewer, (5) stale, (6) forbidden files
    al.set_meta("ab_win", review_recommendation="отклонить")
    assert any("recommends reject" in w for w in accept_policy("ab_win")[1])
    al.set_meta("ab_win", review_recommendation=None)
    assert any("no review yet" in w for w in accept_policy("ab_win")[1])
    al.set_meta("ab_win", review_recommendation="принять", stale=True)
    assert any("stale" in w for w in accept_policy("ab_win")[1])
    al.set_meta("ab_win", stale=False)
    for bad in ("Dockerfile", "profiles/mode_economy.json", "harness/mode.py"):
        info["files"] = [bad]
        assert any("stay with the human" in w for w in accept_policy("ab_win")[1]), bad
    info["files"] = ["harness/sorter.py"]
    assert accept_policy("ab_win")[0]
    c.accept_preflight = lambda name, against="HEAD": (None, "the patch breaks the autolab rules: touches files outside the allowed set")
    assert any("breaks the autolab rules" in w for w in accept_policy("ab_win")[1])        # the lab's own patch check (check_patch) is part of the policy
    c.accept_preflight = lambda name, against="HEAD": (dict(info), "")
    # a verdict that is not accept on the dev set
    al.set_meta("ab_win", verdict={"b": "clarify"})
    assert any("not accept" in w for w in accept_policy("ab_win")[1])
    al.set_meta("ab_win", verdict={"b": "accept"})

    # auto accept: a branch on a local bare "GitHub", never master
    bare, seed = tmp / "bare.git", tmp / "seed"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "master", str(seed)], check=True)
    (seed / "harness").mkdir()
    (seed / "harness" / "a.py").write_text("x = 1\n")
    for a_ in (("add", "-A"), ("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"), ("remote", "add", "origin", str(bare)), ("push", "-q", "origin", "master")):
        subprocess.run(["git", "-C", str(seed), *a_], check=True)
    (tmp / "key").write_text("dummy")
    os.environ.update(AUTOLAB_PUSH_REMOTE=str(bare), AUTOLAB_PUSH_CLONE_AUTO=str(tmp / "clone_auto"), AUTOLAB_DEPLOY_KEY=str(tmp / "key"))
    os.environ.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    host = tmp / "lab" / "autolab_repo"                  # the host copy: its own history, like on the host
    host.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(host)], check=True)
    (host / "harness").mkdir()
    (host / "harness" / "a.py").write_text("x = 1\n")
    for a_ in (("add", "-A"), ("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")):
        subprocess.run(["git", "-C", str(host), *a_], check=True)
    al.repo_root = lambda: host

    # plan parsing and validation
    assert parse_plan('text {"plan": "p", "actions": [{"action": "wait", "why": "x"}]} tail')["plan"] == "p" and parse_plan("no json") is None
    assert validate_action({"action": "wait"}) and validate_action({"action": "dance", "why": "x"}) and validate_action({"action": "rerun", "name": "ab_win", "repeats": 9, "why": "x"})
    assert validate_action({"action": "check", "name": "ab_win", "set": "hold_math", "why": "x"}) and validate_action({"action": "check", "name": "ab_win", "budget": 5, "why": "x"})
    assert validate_action({"action": "review", "name": "ab_nope", "why": "x"}) and validate_action({"action": "wait", "why": "ok"}) is None

    # ticks with a fake LLM
    fake = {"out": ""}
    al.llm = lambda system, user, max_tokens=4000, purpose="": fake["out"]
    al.chain = lambda: ["qwen"]
    al.limits_cfg_orig = al.limits_cfg
    cfg_over = {"chief_dry_run": True, "chief_digest_hour": 99}
    al.limits_cfg = lambda: dict(al.limits_cfg_orig(), **cfg_over)
    plan_act = lambda acts: json.dumps({"plan": "# goal\nstep list", "actions": acts})
    # the switch: off -> nothing, not even a model call
    calls = []
    al.llm = lambda system, user, max_tokens=4000, purpose="": calls.append(purpose) or fake["out"]
    assert "off" in cmd_chief(force=True) and not calls
    al.chief_flag().parent.mkdir(parents=True, exist_ok=True)
    al.chief_flag().write_text("on")
    # the autopilot yields to the chief
    (c.CFG_DIR).mkdir(exist_ok=True)
    al.autopilot_flag().write_text("on")
    assert "chief is on" in al.cmd_autopilot()
    # dry-run: a plan with check + rerun + accept executes nothing, tells the chat and writes the plan
    st0 = {"q": len(list((al.EXP / "queue" / "pending").glob("*.yaml"))), "m": sorted(_names())}
    for f in ("ab_win.meta.json",):
        pass
    al.set_meta("ab_win", auto_branch=None)
    fake["out"] = plan_act([{"action": "check", "name": "ab_win", "set": "dev_math", "why": "need a fresh set"}, {"action": "rerun", "name": "ab_win", "repeats": 2, "why": "noise"},
                            {"action": "accept", "name": "ab_win", "why": "all evidence is in"}, {"action": "reject", "name": "ab_win", "why": "x"}])
    out = cmd_chief(force=True)
    assert "dry-run" in out and calls == ["chief"] and not spawned and st0["q"] == len(list((al.EXP / "queue" / "pending").glob("*.yaml"))) and sorted(_names()) == st0["m"], out
    assert not al.decision_of("ab_win") and "auto/ab_win" not in subprocess.run(["git", "-C", str(bare), "branch", "--list", "auto/ab_win2"], capture_output=True, text=True).stdout
    msgs = [json.loads(f.read_text(encoding="utf-8"))["text"] for f in sorted((c.CFG_DIR / "tg_outbox").glob("*.json"))]
    assert msgs and "dry-run" in msgs[-1] and "разрешила бы" in msgs[-1] and "поставил бы проверку" in msgs[-1] and "Итог:" in msgs[-1], msgs[-1]
    assert "DRY-RUN" in plan_file().read_text(encoding="utf-8") and "step list" in plan_text()
    assert _state()["actions_today"] == 0 and "[dry]" in status_text()
    # too early: the gap between ticks
    assert "too early" in cmd_chief()
    # the actions per tick are capped
    fake["out"] = plan_act([{"action": "wait", "why": f"w{i}"} for i in range(8)])
    cmd_chief(force=True)
    assert len(_state()["log"]) == 2 * al.limits_cfg()["chief_max_actions"], len(_state()["log"])      # 4 planned actions are capped at 3 per tick
    # an invalid action is reported, not executed
    fake["out"] = plan_act([{"action": "accept", "name": "ab_nope", "why": "x"}])
    cmd_chief(force=True)
    assert "invalid" in _state()["log"][-1]["result"]
    # live: a refused accept (no review) and a done accept; the queue and active limits
    subprocess.run(["git", "-C", str(bare), "branch", "-D", "auto/ab_win"], capture_output=True)
    cfg_over["chief_dry_run"] = False
    al.set_meta("ab_win", review_recommendation=None)
    fake["out"] = plan_act([{"action": "accept", "name": "ab_win", "why": "recommend"}])
    cmd_chief(force=True)
    assert _state()["log"][-1]["result"].startswith("refused") and "no review yet" in _state()["log"][-1]["result"] and not al.meta("ab_win").get("auto_branch")
    al.set_meta("ab_win", review_recommendation="принять")
    fake["out"] = plan_act([{"action": "accept", "name": "ab_win", "why": "all evidence is in"}])
    cmd_chief(force=True)
    assert al.meta("ab_win").get("in_stack") and al.meta("ab_win").get("auto_branch") == "auto/stack" and _state()["actions_today"] == 1, _state()["log"][-1]
    assert al.decision_of("ab_win") is None and "already on the stack" in " ".join(accept_policy("ab_win")[1])      # the human still decides
    # live check/rerun go through the existing functions; the limit of rechecks per hypothesis; a full queue; a closed gateway gate
    import types
    fake_lab = types.ModuleType("lab")
    fake_lab.load_experiment = lambda p: {}
    sys.modules["lab"] = fake_lab
    # ================= the stack: a second hypothesis over the first, checks, culprit, the human accept, re-creation
    wt = al.stack_wt()
    assert al.stack_active() and al.stack_sig() == "ab_win" and al.stack_env() == {"HARNESS_SORTER": "fifo"} and (wt / "harness" / "a.py").read_text() == "x = 1\n"
    assert subprocess.run(["git", "-C", str(bare), "rev-parse", "--verify", "-q", "auto/stack"], capture_output=True).returncode == 0
    assert subprocess.run(["git", "-C", str(bare), "rev-parse", "master"], capture_output=True, text=True).stdout.strip() == subprocess.run(["git", "-C", str(seed), "rev-parse", "master"], capture_output=True, text=True).stdout.strip()
    seen, pre_map = [], {}

    def pre_mock(name, against="HEAD"):
        seen.append(against)
        return (dict(pre_map[name]), "") if name in pre_map else (dict(info), "")
    c.accept_preflight = pre_mock

    def git_(*a_, cwd=host):
        return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *a_], capture_output=True, text=True)

    def code_hyp(name, fname, content, envkey):
        """A hypothesis prepared and measured on top of the stack, with a check on other data and a second budget."""
        al.write_draft(dict(d, name=name, target_cells=["blk/a"], arms={"a": {}, "b": {"env": {envkey: "1"}}}))
        al.set_meta(name, sim_verdict="pass")
        al.cmd_submit(name)
        q = yaml.safe_load((al.EXP / "queue" / "pending" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert q["code_dir"] == str(wt) and q["arms"]["a"]["env"] == al.stack_env() and q["arms"]["b"]["env"] == {**al.stack_env(), envkey: "1"}, q      # reference = the stack, variant = the stack + the hypothesis
        assert al.meta(name)["stack_at_submit"] == al.stack_sig()
        (al.EXP / "queue" / "pending" / f"{name}.yaml").unlink()
        wtc = host.parent / "autolab_worktrees" / name
        assert git_("worktree", "add", "-b", f"autolab/{name}", str(wtc), "auto/stack").returncode == 0
        (wtc / "harness" / fname).write_text(content)
        git_("add", "-A", cwd=wtc)
        git_("commit", "-qm", name, cwd=wtc)
        sha = git_("rev-parse", "--short", "HEAD", cwd=wtc).stdout.strip()
        sig = al.stack_sig()
        al.set_meta(name, stage="verdict", branch=f"autolab/{name}", worktree=str(wtc), commit=sha, verdict={"b": "accept"}, result=res_ok(11.0), review_recommendation="принять",
                    target_cells=["blk/a"], queued_at=time.time(), stack_at_apply=sig, stack_at_submit=sig)
        for k, (ds_, bud, vv, mm) in enumerate((("dev_math", 30000, "clarify", 3.0), ("dev_math", 50000, "clarify", -1.0))):
            al.write_draft(dict(d, name=f"{name}_c{k}", set=ds_, budget=bud))
            al.set_meta(f"{name}_c{k}", stage="verdict", check_of=name, verdict={"b": vv}, result=res_ok(mm))
        pre_map[name] = dict(info, name=name, no_code=False, branch=f"autolab/{name}", commit=sha, files=[f"harness/{fname}"], stat=f"{fname} | 1 +")

    code_hyp("ab_code", "a.py", "x = 1\ny = 2\n", "HARNESS_X")
    ok, msg = stack_add("ab_code")
    assert ok and "item 2" in msg and "auto/stack" in seen, msg
    assert (wt / "harness" / "a.py").read_text() == "x = 1\ny = 2\n" and al.stack_sig() == "ab_win,ab_code" and "HARNESS_X" in al.stack_env()
    assert git_("show", "auto/stack:harness/a.py", cwd=bare).stdout == "x = 1\ny = 2\n"                       # GitHub has it too, on the branch only
    assert git_("show", "master:harness/a.py", cwd=bare).stdout == "x = 1\n" and al.decision_of("ab_code") is None
    # a hypothesis measured on another state of the stack, or before the stack, is refused
    al.set_meta("ab_code", stack_at_submit="ab_win", in_stack=False, auto_branch=None)
    assert any("another state of the stack" in w for w in accept_policy("ab_code")[1])
    al.set_meta("ab_code", stack_at_submit="ab_win,ab_code", in_stack=True)
    # the stack check: against main with the stack switches off, after every second accept
    cfg_over["chief_stack_check_every"] = 2
    assert "общий перепрогон stk_1" in " ".join(stack_housekeeping(False))
    q = yaml.safe_load((al.EXP / "queue" / "pending" / "stk_1.yaml").read_text(encoding="utf-8"))
    assert q["code_dir"] == str(wt) and q["arms"]["main"].get("env") in (None, {}) and q["arms"]["stack"]["env"] == al.stack_env() and not al.hyp_active("stk_1"), q
    why_ = accept_stack_preflight()[1]
    assert "общей проверки" in why_ and "accept refused" in c.handle(42, "принять стек")["text"], why_          # no passed check: no accept of the stack
    assert stack_housekeeping(False) == []                                                                       # still running: nothing to do
    (al.EXP / "queue" / "pending" / "stk_1.yaml").unlink()
    al.set_meta("stk_1", verdict={"stack": "reject"}, result={"main": {}, "stack": {"points_diff_vs_first": {"mean": -6.0, "ci95": [-9.0, -2.0], "threshold": 5.0}}})
    out = " | ".join(stack_housekeeping(False))
    assert "ПРОВАЛЕНА" in out and "поиск виновника stk_2_abl" in out, out
    q = yaml.safe_load((al.EXP / "queue" / "pending" / "stk_2_abl.yaml").read_text(encoding="utf-8"))
    assert list(q["arms"]) == ["full", "no_ab_win", "no_ab_code"] and q["arms"]["no_ab_code"]["env"] == {"HARNESS_SORTER": "fifo"}, q["arms"]
    (al.EXP / "queue" / "pending" / "stk_2_abl.yaml").unlink()
    al.set_meta("stk_2_abl", verdict={"no_ab_win": "clarify", "no_ab_code": "accept"}, result={"full": {}, "no_ab_win": {"points_diff_vs_first": {"mean": 1.0, "threshold": 5.0}}, "no_ab_code": {"points_diff_vs_first": {"mean": 9.0, "threshold": 5.0}}})
    out = " | ".join(stack_housekeeping(False))
    assert "виновник ab_code" in out, out
    assert (wt / "harness" / "a.py").read_text() == "x = 1\n" and al.stack_sig() == "ab_win" and al.stack_env() == {"HARNESS_SORTER": "fifo"}          # the culprit is out
    assert git_("show", "auto/stack:harness/a.py", cwd=bare).stdout == "x = 1\n" and not al.meta("ab_code").get("in_stack")
    # a third hypothesis over the stack, then the new check passes
    code_hyp("ab_code3", "b.py", "z = 3\n", "HARNESS_Z")
    ok, msg = stack_add("ab_code3")
    assert ok and al.stack_sig() == "ab_win,ab_code3", msg
    assert "общий перепрогон stk_3" in " ".join(stack_housekeeping(False))
    (al.EXP / "queue" / "pending" / "stk_3.yaml").unlink()
    al.set_meta("stk_3", verdict={"stack": "accept"}, result={"main": {}, "stack": {"points_diff_vs_first": {"mean": 5.0, "threshold": 3.0}}})
    assert "пройдена" in " ".join(stack_housekeeping(False))
    # the human accept of the stack: a button with a code, one user, the merge into main of the host copy, the push, the reset
    al.status_plan = lambda name, accept=True: list(info["plan"])
    assert "стек" in c.handle(42, "стек")["text"].lower() and "ab_code3" in stack_text()
    r = c.handle(42, "принять стек")
    assert "ПРИНЯТЬ СТЕК" in r["text"] and "ab_win, ab_code3" in r["text"] and "Код" in r["text"], r["text"]
    assert c.handle_callback(43, r["buttons"][0][0]["data"]) is None            # not on the allow list: nothing happens
    main_before = git_("rev-parse", "HEAD").stdout.strip()
    res = c.handle_callback(42, r["buttons"][0][0]["data"])["text"]
    assert "влито в main" in res and "запушено" in res and "Стек сброшен" in res, res
    assert (host / "harness" / "b.py").read_text() == "z = 3\n" and git_("rev-parse", "HEAD").stdout.strip() != main_before
    assert git_("show", "master:harness/b.py", cwd=bare).stdout == "z = 3\n" and git_("rev-parse", "--verify", "-q", "auto/stack", cwd=bare).returncode != 0   # master got it; the branch on GitHub is gone
    assert not al.stack_active() and not wt.exists() and al.stack_load()["items"] == [] and al.decision_of("ab_win") == "accepted" and al.decision_of("ab_code3") == "accepted"
    assert "ничего нет" in c.handle(42, "принять стек")["text"]
    assert stack_ensure() == "" and wt.exists() and (wt / "harness" / "b.py").exists() and al.stack_load()["items"] == []        # the next stack grows from the new main
    # no single culprit: the stack is blocked, a new accept is refused, the human drops it
    st_ = al.stack_load()
    st_["items"] = [{"name": "ab_x", "env": {"K1": "1"}, "active": True, "cells": []}, {"name": "ab_y", "env": {"K2": "1"}, "active": True, "cells": []}]
    st_["ops"], st_["check"] = [{"op": "add", "name": "ab_x"}, {"op": "add", "name": "ab_y"}], {"name": "stk_9", "sig": "ab_x,ab_y", "state": "failed", "verdict": "reject"}
    al.stack_save(st_)
    assert "поиск виновника" in " ".join(stack_housekeeping(False))
    abl_name = al.stack_load()["ablation"]["name"]
    (al.EXP / "queue" / "pending" / f"{abl_name}.yaml").unlink()
    al.set_meta(abl_name, verdict={"no_ab_x": "clarify", "no_ab_y": "reject"}, result={})
    assert "заблокирован" in " ".join(stack_housekeeping(False)) and al.stack_load().get("blocked")
    assert "blocked" in stack_add("ab_win")[1] or "policy" in stack_add("ab_win")[1]
    assert "заблокирован" in accept_stack_preflight()[1] and "Стек сброшен" in c.handle(42, "стек сбросить")["text"] and not al.stack_active() and "пуст" in c.handle(42, "стек")["text"]
    c.accept_preflight = lambda name, against="HEAD": (dict(info), "")
    al.set_meta("ab_code", decision="postponed")
    al.set_meta("ab_win", decision="postponed")          # frees the task type for the next hypotheses
    # ---- ticks: instant actions let it think again at once, a blocking one ends the tick, a change wakes it before the heartbeat
    llm_keep, seq, calls2 = al.llm, [], []

    def llm_seq(system, user, max_tokens=4000, purpose=""):
        calls2.append(purpose)
        return seq.pop(0) if seq else plan_act([{"action": "wait", "why": "idle"}])
    al.llm = llm_seq
    al.write_draft(dict(d, name="ab_rj", target_cells=["blk/a"]))
    al.set_meta("ab_rj", stage="verdict", sim_verdict="pass", verdict={"b": "clarify"}, result=res_ok(0.5), target_cells=["blk/a"], queued_at=time.time())
    seq[:] = [plan_act([{"action": "reject", "name": "ab_rj", "why": "no gain"}]), plan_act([{"action": "wait", "why": "nothing else"}])]
    out = cmd_chief(force=True)
    assert al.decision_of("ab_rj") == "rejected" and len(calls2) == 2 and "2 round" in out, (out, calls2)           # a reject is instant: it thinks again at once
    calls2.clear()
    al.write_draft(dict(d, name="ab_bk", target_cells=["blk/a"]))
    al.set_meta("ab_bk", stage="verdict", sim_verdict="pass", verdict={"b": "clarify"}, result=res_ok(1.0), target_cells=["blk/a"], queued_at=time.time())
    seq[:] = [plan_act([{"action": "check", "name": "ab_bk", "set": "dev_math", "why": "other data"}]), plan_act([{"action": "wait", "why": "x"}])]
    out = cmd_chief(force=True)
    assert len(calls2) == 1 and (al.EXP / "queue" / "pending" / "ab_bk_fresh.yaml").exists() and "1 round" in out, (out, calls2)   # an experiment runs: the tick ends
    calls2.clear()
    st_ = _state()
    st_["last_tick"], st_["sig"] = time.time() - 600, _world_sig()
    _save(st_)
    assert "nothing changed" in cmd_chief() and not calls2
    al.set_meta("ab_bk", review_recommendation="повторить")                           # a review arrived: wake up
    seq[:] = [plan_act([{"action": "wait", "why": "x"}])]
    cmd_chief()
    assert calls2 == ["chief"]
    al.set_meta("ab_bk", review_recommendation="принять")
    assert "too early" in cmd_chief()                                                  # but not twice within the event gap
    al.llm = llm_keep
    al.set_meta("ab_bk", decision="postponed")
    for f_ in ("ab_bk_fresh",):
        (al.EXP / "queue" / "pending" / f"{f_}.yaml").unlink()
        al.set_meta(f_, decision="postponed")
    al.write_draft(dict(d, name="ab_two", target_cells=["blk/a"]))
    al.set_meta("ab_two", stage="verdict", sim_verdict="pass", verdict={"b": "clarify"}, result=res_ok(2.0), target_cells=["blk/a"], queued_at=time.time())
    fake["out"] = plan_act([{"action": "check", "name": "ab_two", "set": "dev_math", "why": "other data"}])
    cmd_chief(force=True)
    assert "ab_two_fresh" in _state()["log"][-1]["result"] and (al.EXP / "queue" / "pending" / "ab_two_fresh.yaml").exists(), _state()["log"][-1]
    cfg_over["chief_max_rechecks"] = 1
    fake["out"] = plan_act([{"action": "rerun", "name": "ab_two", "repeats": 2, "why": "again"}])
    cmd_chief(force=True)
    assert "checks/repeats" in _state()["log"][-1]["result"]
    cfg_over["chief_max_rechecks"] = 3
    cfg_over["max_queue_depth"] = 0
    fake["out"] = plan_act([{"action": "check", "name": "ab_win", "set": "dev_math", "why": "x"}])
    cmd_chief(force=True)
    assert "queue is full" in _state()["log"][-1]["result"]
    cfg_over["max_queue_depth"] = 4
    gate_ok = (False, "weekly gateway use")
    globals()["_gate_ok"] = lambda: gate_ok
    fake["out"] = plan_act([{"action": "rerun", "name": "ab_win", "repeats": 2, "why": "x"}])
    cmd_chief(force=True)
    assert "gateway gate" in _state()["log"][-1]["result"]
    # the daily dollar limit stops the chief, tells the chat once, and no model call is made
    calls.clear()
    with open(al.USAGE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "model": "m", "backend": "claude", "purpose": "chief", "requests": 1, "ok": True, "in": 1, "out": 1, "cost": 1, "usd": 10.0}) + "\n")
    n_msg = len(list((c.CFG_DIR / "tg_outbox").glob("*.json")))
    out = cmd_chief(force=True)
    assert "stopped" in out and not calls and len(list((c.CFG_DIR / "tg_outbox").glob("*.json"))) == n_msg + 1, (out, calls, n_msg, len(list((c.CFG_DIR / "tg_outbox").glob("*.json"))))
    cmd_chief(force=True)
    assert len(list((c.CFG_DIR / "tg_outbox").glob("*.json"))) == n_msg + 1           # the notice is sent once a day
    # the caps of the dollars are enforced inside the model call too
    assert al.day_usd() >= 10.0 and al.limits_cfg_orig()["usd_per_week"] == 50.0 and al.limits_cfg_orig()["usd_per_day"] == 10.0
    # the switch off and the STOP file
    al.USAGE.unlink()
    # the deadline of the experiments: the phase in the context, no run that cannot finish, nothing at all after the end
    fmt = lambda dt: time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + dt))
    assert "no deadline" in time_text()
    cfg_over["chief_experiments_until"], cfg_over["chief_final_deadline"] = fmt(30 * 3600), fmt(34 * 3600)
    assert "PHASE: exploring" in time_text() and "29." in time_text() or "30." in time_text()
    cfg_over["chief_experiments_until"] = fmt(8 * 3600)
    assert "PHASE: converging" in time_text()
    cfg_over["chief_experiments_until"] = fmt(600)
    assert "PHASE: closing" in time_text() and "TIME:" in build_context()
    for kind_ in ("rerun", "check", "propose"):
        assert "not enough time" in execute({"action": kind_, "name": "ab_two", "repeats": 2, "set": "dev_math", "cell": "blk/a", "why": "x", "idea": "i" * 30}, False, _state(), (True, "ok")), kind_
    cfg_over["chief_experiments_until"] = fmt(-60)
    calls.clear()
    n_msg = len(list((c.CFG_DIR / "tg_outbox").glob("*.json")))
    out = cmd_chief(force=True)
    assert "over" in out and not calls and len(list((c.CFG_DIR / "tg_outbox").glob("*.json"))) == n_msg + 1, out         # no model, one notice
    cmd_chief(force=True)
    assert len(list((c.CFG_DIR / "tg_outbox").glob("*.json"))) == n_msg + 1
    cfg_over["chief_experiments_until"] = cfg_over["chief_final_deadline"] = ""
    # parallel runs: the chief may lower it (1..4), nothing else; the number reaches the queued experiment
    assert validate_action({"action": "parallel", "n": 9, "why": "x"}) and validate_action({"action": "parallel", "n": 0, "why": "x"}) and validate_action({"action": "parallel", "n": 2, "why": "x"}) is None
    assert al.parallel_runs_now() == 4 and "PARALLEL RUNS: 4" in build_context()
    assert "dry-run" in execute({"action": "parallel", "n": 2, "why": "x"}, True, _state(), (True, "ok")) and al.parallel_runs_now() == 4
    assert "4 -> 2" in execute({"action": "parallel", "n": 2, "why": "429 in the last experiment"}, False, _state(), (True, "ok")) and al.parallel_runs_now() == 2
    cfg_over["max_queue_depth"] = 12
    al.write_draft(dict(d, name="ab_par", target_cells=["blk/a"]))
    al.set_meta("ab_par", sim_verdict="pass")
    al.cmd_submit("ab_par")
    assert yaml.safe_load((al.EXP / "queue" / "pending" / "ab_par.yaml").read_text(encoding="utf-8"))["parallel_runs"] == 2
    (al.EXP / "queue" / "pending" / "ab_par.yaml").unlink()
    al.set_meta("ab_par", decision="postponed")
    cfg_over["max_queue_depth"] = 4
    (al.DRAFTS / "parallel.json").unlink()
    (al.EXP / "results" / "ab_old").mkdir(parents=True)
    (al.EXP / "results" / "ab_old" / "ABORTED.txt").write_text("gateway proxy is halted (429/limits guard)")
    assert "ab_old: gateway proxy is halted" in parallel_text()
    # a hypothesis on the stack holds no task types
    al.set_meta("ab_hold", stage="verdict", verdict={"b": "accept"}, target_cells=["blk/a"], queued_at=time.time())
    assert al.hyp_active("ab_hold")
    al.set_meta("ab_hold", in_stack=True)
    assert not al.hyp_active("ab_hold")
    al.write_draft(dict(d, name="ab_inv", target_cells=["blk/a"]))
    al.set_meta("ab_inv", stage="queued", target_cells=["blk/a"], queued_at=time.time())
    (al.EXP / "queue" / "done" / "ab_inv.yaml").write_text("name: ab_inv" + chr(10) * 2 + "# ---- written by lab.py: status and result of this experiment (the part above is the experiment) ----" + chr(10) + "status: invalid" + chr(10) + "reason: set no longer builds to its frozen hash" + chr(10), encoding="utf-8")
    assert not al.hyp_active("ab_inv") and "LAB invalid" in hyp_table()                 # the lab refused it: it holds no task types, the chief sees why
    al.chief_flag().unlink()
    assert "off" in cmd_chief(force=True)
    al.chief_flag().write_text("on")
    al.STOP_FILE.write_text("stop")
    assert "stop flag" in cmd_chief(force=True)
    al.STOP_FILE.unlink()
    assert "Шеф" in status_text() and "Дайджест" in digest_text()
    print("autolab_chief selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    elif "--status" in sys.argv:
        print(status_text())
    else:
        print(cmd_chief(force="--force" in sys.argv))
