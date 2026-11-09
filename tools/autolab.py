#!/usr/bin/env python3
"""Autonomous lab (S9): a hypothesis generator and an applying agent around the experiment queue. Lab tool, runs on the GPU host.

    python3 autolab.py cycle [--n 2] [--topic T] [--wait]   ONE command: propose -> validate/dedupe -> apply (code) -> sim filter -> submit -> (wait) -> verdict -> report
    python3 autolab.py cycle --names a,b [--wait]            the same for drafts that exist (skips propose); stages that are done are not repeated
    python3 autolab.py propose [--n 3] [--topic T] [--dry]   the generator asks the LLM for hypotheses -> drafts in experiments/autolab/
    python3 autolab.py list                                  drafts, their stage and verdict
    python3 autolab.py apply <name> [--patch-file F]         applying agent: a code change ONLY in a worktree on branch autolab/<name>
    python3 autolab.py sim <name>                            simulator as a FILTER before the expensive run (scheduler-level hypotheses)
    python3 autolab.py submit <name>                         draft -> queue/pending (the dispatcher runs the queue)
    python3 autolab.py verdict <name>                        result -> experiments/autolab/<name>.verdict.md (threshold = noise floor of the lab)
    python3 autolab.py export <name>                         the branch as a patch file for a human to review and merge
    python3 autolab.py usage                                 gateway REQUESTS spent by the autolab (the gateway counts requests, not tokens)
    python3 autolab.py selftest

Rules that are enforced in code (not by asking the model nicely):
  * the model writes hypotheses (YAML) and code edits (search/replace blocks) only; it never touches the frozen sets, the lab
    (tools/, experiments/, results), the acceptance criteria or keys: a change may only touch harness/, profiles/, run_agent.py,
    is checked against a deny-list, and a snapshot of the protected files is compared before and after;
  * code lives in a worktree on a branch autolab/<name> (on the host: a git repo seeded from the deployed archive); the run of such a
    hypothesis uses that tree through the experiment key `code_dir`; nothing is merged or deployed to the harness: MERGE ONLY BY A HUMAN;
  * limits per cycle and per week (LLM requests of the autolab itself, hypotheses per cycle, experiments in the queue, expected
    requests) are in LIMITS_DEFAULT, overridable by <lab>/autolab.yaml (a lab file, the model cannot write it);
  * the verdict uses the lab's rule: paired bootstrap interval AND the noise-floor threshold of `lab.py noisefloor`; this module only
    READS the threshold.

Model: AUTOLAB_MODEL (default qwen3.8-27b) through the shared proxy (127.0.0.1:18443); before each call the experiment gate of
dispatcher.yaml is checked, a 429 is waited out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stream_run as sr  # noqa: E402

LAB = sr.LAB
EXP = Path(os.environ.get("LAB_EXPERIMENTS") or LAB / "experiments")
DRAFTS = EXP / "autolab"
USAGE = DRAFTS / "usage.jsonl"
MODEL = os.environ.get("AUTOLAB_MODEL", "qwen3.8-27b")
BASE = os.environ.get("AUTOLAB_BASE_URL", f"http://127.0.0.1:{sr.PROXY_PORT}/v1")

LIMITS_DEFAULT = {
    "max_hyp_per_cycle": 3,            # hypotheses taken from one propose call
    "max_llm_calls_per_cycle": 14,     # gateway requests of the autolab itself (propose + apply + retries) in one process run
    "weekly_llm_calls": 1500,          # gateway requests of the autolab per ISO week (the gateway limit is 1 000 000 requests)
    "max_queue_depth": 4,              # experiments pending + running before submit refuses
    "max_expected_requests_cycle": 12000,   # estimated gateway requests of all experiments of one cycle
    "max_cost_tokens": 150000,         # expected ledger tokens of one experiment
    "max_code_lines": 120,             # added + removed lines of one model patch
    "wait_poll_s": 60, "max_wait_min": 300,
}
# The code a model-written patch may touch. Everything else (tools/, experiments/, frozen sets, dispatcher, keys, Docker) is refused.
ALLOWED_PREFIX = ("harness/", "profiles/", "run_agent.py")
FORBIDDEN = ("Dockerfile", "requirements.txt", "requirements-lab.txt", "tools/deploy.sh")
FORBIDDEN_ADDED = re.compile(r"holdout|noise_?floor|noisefloor|threshold_points|acceptance|\.ssh|api[_-]?key|secret|token_juiwen|"
                             r"experiments/|dispatcher|autolab|subprocess|os\.system|rmtree|urllib|socket|requests\.|\beval\(|\bexec\(|"
                             r"__import__|/home/|\.env\b", re.I)
PROTECTED = ("tools/lab.py", "tools/autolab.py", "tools/dispatcher.py", "tools/sim.py", "tools/stream_run.py", "experiments/sets", "dispatcher.yaml",
             "autolab.yaml")
DRAFT_KEYS = ("name", "hypothesis", "set", "model", "repeats", "budget", "arms", "expected_cost_tokens", "code_change")
ARM_KEYS = ("env", "batch_size", "model")
# Allowed values of the enumerated switches (anything else is silently treated as "off" by the harness, so it is refused here).
SWITCH_VALUES = {
    "HARNESS_VERIFY": ["off", "free", "judge", "other", "other_judge"], "HARNESS_VERIFY_ACTION": ["log", "retry", "recheck"],
    "HARNESS_RECHECK_MODE": ["strong", "other"], "HARNESS_RECHECK_QUEUE": ["inline", "later"],
    "HARNESS_DISAGREE": ["higher_p", "strong", "third"], "HARNESS_LOOP": ["off", "log", "retry"],
    "HARNESS_CENSOR": ["off", "log", "enforce"], "HARNESS_CENSOR_BASIS": ["fixed", "pred"],
    "HARNESS_BUDGET_HARD": ["off", "on"], "HARNESS_GUARD": ["off", "log", "strip"], "HARNESS_SANDBOX": ["off", "scrub", "limits"],
    "HARNESS_BATCH_PROMPT": ["plain", "compute"], "HARNESS_SORTER": ["greedy", "fifo", "reward"],
    "HARNESS_PREDICTOR": ["placeholder", "flat", "measured"], "HARNESS_PUNSOLV": ["off", "contradiction"],
    "HARNESS_SOLVERS": ["arith", "contradiction", "arith,contradiction"], "HARNESS_SUBMIT_UNSOLVABLE": ["on", "off"],
    "HARNESS_FALLBACK": ["on", "off", "late"], "HARNESS_GUESS_UNSOLVABLE": ["off", "free", "llm"],
    "HARNESS_VERIFY_JUDGE_TYPES": None, "HARNESS_VERIFY_OTHER_TYPES": None, "HARNESS_BATCH_TYPES": None, "HARNESS_RECHECK_TYPES": None,
}
# Switches the arms of a hypothesis may NOT set although SCRIPTS.md names them: they change the grading, the budget or the gateway.
DENY_SWITCHES = {"HARNESS_FORCE_MODEL", "HARNESS_LIMITS", "HARNESS_CWD", "HARNESS_LOG_DIR", "HARNESS_BUDGET_FILE", "BUDGET"}
MODELS = ["qwen3.6-35b-a3b-noreason", "qwen3.6-35b-a3b", "qwen3.8-27b-noreason", "qwen3.8-27b"]
SIM_KEYS = ("HARNESS_SORTER", "HARNESS_PREDICTOR", "HARNESS_PUNSOLV", "HARNESS_SOLVERS")
CALLS = {"n": 0}   # LLM requests made by this process


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def limits_cfg() -> dict:
    cfg = dict(LIMITS_DEFAULT)
    f = LAB / "autolab.yaml"
    if f.exists():
        import yaml
        cfg.update({k: v for k, v in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).items() if k in LIMITS_DEFAULT})
    return cfg


# ------------------------------------------------------------------ accounting of the autolab's own gateway requests
def usage_rows() -> list[dict]:
    return [json.loads(l) for l in open(USAGE, encoding="utf-8") if l.strip()] if USAGE.exists() else []


def week_start_ts() -> float:
    t = time.gmtime()
    return time.time() - (t.tm_wday * 86400 + t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec)   # Monday 00:00 UTC, the gateway's ISO week


def week_calls() -> int:
    ws = week_start_ts()
    return sum(int(r.get("requests", 1)) for r in usage_rows() if r["ts"] >= ws)


def gw_week_used() -> int | None:
    try:
        import dispatcher
        fr = dispatcher.fractions(sr.limits())
        return fr["week_used"] if fr else None
    except Exception:  # noqa: BLE001
        return None


def record_usage(purpose: str, u: dict, ok: bool) -> None:
    cost = u.get("prompt_tokens", 0) / 10.0 + u.get("completion_tokens", 0)
    DRAFTS.mkdir(parents=True, exist_ok=True)
    with open(USAGE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "model": MODEL, "purpose": purpose, "requests": 1, "ok": ok, "in": u.get("prompt_tokens", 0),
                            "out": u.get("completion_tokens", 0), "cost": round(cost, 1)}) + "\n")


# ------------------------------------------------------------------ the LLM through the shared gateway
def llm(system: str, user: str, max_tokens: int = 4000, purpose: str = "") -> str:
    import dispatcher
    cfg = dispatcher.load_cfg()
    lim = limits_cfg()
    for attempt in range(6):
        if CALLS["n"] >= lim["max_llm_calls_per_cycle"]:
            raise SystemExit(f"autolab limit: {CALLS['n']} LLM requests in this run (max_llm_calls_per_cycle={lim['max_llm_calls_per_cycle']})")
        if week_calls() >= lim["weekly_llm_calls"]:
            raise SystemExit(f"autolab limit: {week_calls()} LLM requests this week (weekly_llm_calls={lim['weekly_llm_calls']})")
        ok, why = dispatcher.exp_gate(cfg, dispatcher.fractions(sr.limits()))
        if not ok:
            raise SystemExit(f"gateway gate (dispatcher.yaml): {why}")
        body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0.7,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
        CALLS["n"] += 1
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
        except urllib.error.HTTPError as ex:
            record_usage(purpose, {}, False)
            if ex.code == 429:
                wait = 65 * (attempt + 1)
                log(f"429 from the gateway (concurrent-request limit); waiting {wait} s")
                time.sleep(wait)
                continue
            raise
        record_usage(purpose, r.get("usage") or {}, True)
        return r["choices"][0]["message"].get("content") or ""
    raise SystemExit("the gateway kept answering 429")


def cmd_usage() -> None:
    rows = usage_rows()
    ws = week_start_ts()
    wk = [r for r in rows if r["ts"] >= ws]
    cfg = limits_cfg()
    print(f"autolab gateway requests: total {sum(r.get('requests', 1) for r in rows)} ({sum(1 for r in rows if r.get('ok') is False)} failed/429), "
          f"this ISO week {sum(r.get('requests', 1) for r in wk)} of own cap {cfg['weekly_llm_calls']} "
          f"(gateway week limit 1 000 000 requests); tokens (input/10 + output) total {sum(r['cost'] for r in rows):.0f}, model {MODEL}")
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["purpose"], []).append(r)
    for p, v in sorted(by.items()):
        print(f"  {p:<10} {sum(x.get('requests', 1) for x in v):>4} requests {sum(x['cost'] for x in v):>8.0f} tokens")
    exp_req = 0
    for f in sorted(DRAFTS.glob("*.meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("experiment_requests"):
            exp_req += m["experiment_requests"]
            print(f"  experiment {f.name[:-10]:<34} {m['experiment_requests']:>6} requests (pipeline+agent calls in cells; verifier calls not included)")
    print(f"  experiments of autolab drafts: {exp_req} requests (lower bound)")


# ------------------------------------------------------------------ knowledge handed to the generator
def known_switches() -> set[str]:
    for f in (HERE.parent / "SCRIPTS.md", LAB / "harness" / "SCRIPTS.md"):
        if f.exists():
            return set(re.findall(r"HARNESS_[A-Z_]+", f.read_text(encoding="utf-8"))) - DENY_SWITCHES
    return set()


def switch_help() -> str:
    """The table rows of SCRIPTS.md that describe switches, shortened: what each switch does, for the generator."""
    for f in (HERE.parent / "SCRIPTS.md", LAB / "harness" / "SCRIPTS.md"):
        if f.exists():
            rows = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.startswith("| ") and "HARNESS_" in l]
            return "\n".join(r[:330] for r in rows[:30])
    return ""


def context_text() -> str:
    for f in (HERE.parent / "autolab_context.md", LAB / "harness" / "autolab_context.md"):
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
    return ""


def sets_available() -> dict[str, str]:
    import yaml
    out = {}
    for f in sorted((EXP / "sets").glob("*.yaml")):
        s = yaml.safe_load(f.read_text(encoding="utf-8"))
        if s.get("role") == "dev":   # holdout is never offered; fresh sets are not offered either (they change)
            out[f.stem] = f"tasks {s.get('tasks')}"
    return out


def _split_lab_block(text: str) -> tuple[dict, dict]:
    import yaml
    mark = "# ---- written by lab.py"
    try:
        e = yaml.safe_load(text.split(mark)[0]) or {}
        blk = yaml.safe_load(text.split(mark)[1].split("\n", 1)[1]) if mark in text else {}
    except Exception:  # noqa: BLE001
        return {}, {}
    return e, blk or {}


def all_experiments() -> list[dict]:
    """Every experiment the lab knows (done, running, pending, autolab drafts) as {name, e, block, where}."""
    out = []
    for kind in ("done", "running", "pending"):
        for f in sorted((EXP / "queue" / kind).glob("*.yaml")):
            e, blk = _split_lab_block(f.read_text(encoding="utf-8"))
            if e.get("name"):
                out.append({"name": e["name"], "e": e, "block": blk, "where": kind, "mtime": f.stat().st_mtime})
    for f in sorted(DRAFTS.glob("*.yaml")):
        e, _ = _split_lab_block(f.read_text(encoding="utf-8"))
        if e.get("name") and not any(x["name"] == e["name"] for x in out):
            out.append({"name": e["name"], "e": e, "block": {}, "where": "draft:" + meta(e["name"]).get("stage", "?"), "mtime": f.stat().st_mtime})
    return out


def diff_signature(e: dict) -> tuple:
    """What an experiment COMPARES: set, budget and, per non-reference arm, the settings that differ from the first arm.
    Two hypotheses with the same signature test the same thing, whatever their names and wording."""
    arms = e.get("arms") or {}
    if not arms:
        return ()
    first = next(iter(arms.values())) or {}

    def flat(a):
        a = a or {}
        d = {str(k): str(v) for k, v in (a.get("env") or {}).items()}
        if a.get("batch_size"):
            d["batch_size"] = str(a["batch_size"])
        if a.get("model"):
            d["model"] = str(a["model"])
        return d
    f0 = flat(first)
    sig = set()
    for a in list(arms.values())[1:]:
        fa = flat(a)
        sig.add(frozenset((k, fa.get(k), f0.get(k)) for k in set(fa) | set(f0) if fa.get(k) != f0.get(k)))
    return (e.get("set"), int(e.get("budget") or 0), frozenset(sig))


def find_duplicate(h: dict) -> str | None:
    s = diff_signature(h)
    for x in all_experiments():
        if x["name"] != h.get("name") and diff_signature(x["e"]) == s:
            return x["name"]
    return None


def digest() -> str:
    """Finished experiments, newest first: what was tested, per-arm points/tokens, interval, verdict, and why a failed one failed."""
    rows = []
    for x in sorted(all_experiments(), key=lambda x: -x["mtime"]):
        e, blk = x["e"], x["block"]
        res = blk.get("result") or {}
        if x["where"] == "done" and res:
            arms = {a: {"pts": v.get("points_per_run"), "tok": v.get("tokens_per_run"), "acc": v.get("accuracy"),
                        "diff": (v.get("points_diff_vs_first") or {}).get("mean"), "ci": (v.get("points_diff_vs_first") or {}).get("ci95")}
                    for a, v in (res.get("arms") or {}).items()}
            rows.append(f"- {e['name']} [set {e.get('set')}, budget {e.get('budget')}, repeats {e.get('repeats')}] {(e.get('hypothesis') or '').strip()[:220]} "
                        f"| arms {json.dumps(arms)} | verdict {res.get('verdict')}")
        elif x["where"] == "done":
            rows.append(f"- {e['name']}: status {blk.get('status')} ({str(blk.get('reason') or blk.get('result_error') or '')[:160]})")
        else:
            rows.append(f"- {e['name']}: {x['where']} (not finished; do not repeat it) {(e.get('hypothesis') or '').strip()[:120]}")
    return "\n".join(rows[:30])


SYSTEM = ("You are the hypothesis generator of an experiment lab for an LLM task-solving harness. You propose small, testable "
          "experiments that change harness behaviour ONLY through the existing environment switches (or, when a switch cannot express "
          "it, a small described code change). Reply with JSON only.")


def prompt_propose(n: int, topic: str = "", errors: str = "") -> str:
    sw = ", ".join(sorted(known_switches()))
    return f"""Competition conditions (organizers, 2026-11-09) and measured facts:
{context_text()}

Real costs for planning: a run on mixed100_dev at budget 30000 spends about 30-35k ledger tokens and about 250 gateway requests per arm and repeat; a run of the 40-task sets about 25k tokens; 16 math tasks 5-9k. The whole experiment (arms x repeats) should stay under 150000 tokens. The lab noise: one standard deviation of points between repeats of an unchanged harness is 7.69 points on mixed100 (about 3.4%); a difference below that is never accepted, so propose changes whose effect could plausibly exceed 8 points on a 100-task set (a 4-5% change), or cost savings of 15% and more at equal points.

Experiments already done, running, queued or drafted (newest first; NEVER propose the same comparison again; a result that is `clarify` means the effect was within noise; a `_fresh` experiment is the confirmation of the one it is named after):
{digest()}

Environment switches you may use in `env:` of an arm (nothing else is allowed): {sw}
What the switches do (from the documentation):
{switch_help()}
Allowed values of enumerated switches (any other value silently turns the block off, and will be refused): {json.dumps(SWITCH_VALUES)}
(None = comma-separated list of task types. Numeric switches: HARNESS_VERIFY_ACCEPT, HARNESS_VERIFY_MIN 0..1, HARNESS_RECHECK_MAX, HARNESS_BUDGET_STOP 0..1, HARNESS_CENSOR_TOKENS "math=1500,search=1200".)
A verifier recheck needs HARNESS_VERIFY=judge (or free) AND HARNESS_VERIFY_ACTION=recheck. `batch_size` of an arm (1,2,4,8) switches batching on; it needs HARNESS_BATCH_PROMPT=compute to keep accuracy.
Frozen sets you may use (`set:`): {json.dumps(sets_available())}
Models: qwen3.6-35b-a3b-noreason, qwen3.6-35b-a3b, qwen3.8-27b-noreason, qwen3.8-27b, auto.
`budget` is the token budget of the harness in the experiment (all arms): use 20000-40000 to test what the competition will look like; scheduler-level hypotheses (HARNESS_SORTER, HARNESS_PREDICTOR, HARNESS_PUNSOLV, HARNESS_SOLVERS) are pre-checked in an offline simulator at 20k/30k/40k.
{('Topic for this batch: ' + topic) if topic else 'Topics, in order of priority: task order / scheduler / predictor, cheap modes (noreason models, batching, fallback, censor), behaviour at budgets 20-40k. Verification and rechecks are secondary.'}
{('Your previous answer was refused: ' + errors) if errors else ''}
Propose {n} different hypotheses likely to be useful. Each: JSON object with keys
name (snake_case, unique, start with ab_), hypothesis (2-4 sentences: what changes, why, what you expect IN POINTS OR TOKENS, how it can fail),
set (one of the sets above), model (one model or auto), repeats (1 or 2), budget (integer 10000-100000),
arms (object: arm name -> {{env: {{SWITCH: "value"}}, batch_size: optional int}}); the FIRST arm is the reference (the behaviour you compare against; its env may be empty or hold the current best setting),
expected_cost_tokens (your estimate of the ledger tokens of the whole experiment), code_change (empty string unless an existing switch cannot express it; then one concrete sentence naming the file and the behaviour).
Return a JSON list of {n} objects and nothing else."""


def extract_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else None


def validate(h: dict) -> list[str]:
    errs = []
    for k in ("name", "hypothesis", "set", "arms"):
        if not h.get(k):
            errs.append(f"missing {k}")
    if errs:
        return errs
    extra = set(h) - set(DRAFT_KEYS)
    if extra:
        errs.append(f"keys not allowed: {sorted(extra)} (the lab's rules, thresholds and sets are not the generator's to set)")
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", str(h["name"])):
        errs.append("bad name")
    if h["set"] not in sets_available():
        errs.append(f"set {h['set']} is not an available dev set")
    arms = h["arms"]
    if not isinstance(arms, dict) or not 2 <= len(arms) <= 4:
        errs.append("2-4 arms")
        return errs
    ks = known_switches()
    for a, v in arms.items():
        if not isinstance(v, dict) and v is not None:
            errs.append(f"arm {a}: not an object")
            continue
        if set(v or {}) - set(ARM_KEYS):
            errs.append(f"arm {a}: keys not allowed {sorted(set(v) - set(ARM_KEYS))}")
        for k in (v or {}).get("env") or {}:
            if k in DENY_SWITCHES or (k not in ks and not (h.get("code_change") and re.fullmatch(r"HARNESS_[A-Z_]+", k))):
                errs.append(f"arm {a}: unknown or forbidden switch {k}")   # a code change may introduce a new switch or a new value
            elif h.get("code_change") and k in ks and SWITCH_VALUES.get(k) and str(((v or {}).get("env") or {})[k]) not in SWITCH_VALUES[k]:
                pass
            elif SWITCH_VALUES.get(k) and str(((v or {}).get("env") or {})[k]) not in SWITCH_VALUES[k]:
                errs.append(f"arm {a}: {k} must be one of {SWITCH_VALUES[k]}")
            elif k.endswith(("_MIN", "_ACCEPT", "_GIVEUP", "_THRESHOLD", "_TEMP", "_STOP", "_RECHECK", "_MAX", "_SIZE")):
                try:
                    float(((v or {}).get("env") or {})[k])
                except ValueError:
                    errs.append(f"arm {a}: {k} must be a number")
        if (v or {}).get("batch_size") not in (None, 1, 2, 4, 8):
            errs.append(f"arm {a}: batch_size must be 1, 2, 4 or 8")
        if (v or {}).get("model") and v["model"] not in MODELS + ["auto"]:
            errs.append(f"arm {a}: unknown model")
    try:
        if int(h.get("repeats", 1)) not in (1, 2):
            errs.append("repeats 1 or 2")
        if not 10000 <= int(h.get("budget", 100000)) <= 100000:
            errs.append("budget must be 10000..100000")
        if int(h.get("expected_cost_tokens", 0) or 0) > limits_cfg()["max_cost_tokens"]:
            errs.append(f"expected cost above {limits_cfg()['max_cost_tokens']} tokens")
    except (TypeError, ValueError):
        errs.append("repeats, budget and expected_cost_tokens must be integers")
    if h.get("model", "auto") not in MODELS + ["auto"]:
        errs.append("unknown model")
    if not errs:
        arms_l = list(arms.values())
        if all(diff_signature({"set": 1, "arms": {"a": arms_l[0], "b": b}}) == diff_signature({"set": 1, "arms": {"a": arms_l[0], "b": arms_l[0]}})
               for b in arms_l[1:]):
            errs.append("the arms do not differ from the first arm")
    return errs


def write_draft(h: dict) -> Path:
    import yaml
    DRAFTS.mkdir(parents=True, exist_ok=True)
    e = {"name": h["name"], "author": "orchestrator", "hypothesis": h["hypothesis"].strip(), "backend": "gateway",
         "model": h.get("model", "qwen3.6-35b-a3b-noreason"), "set": h["set"], "repeats": int(h.get("repeats", 1)),
         "budget": int(h.get("budget", 100000)), "fallback": "on",
         "arms": {a: {k: v for k, v in (spec or {}).items() if k in ARM_KEYS} for a, spec in h["arms"].items()}}
    if h.get("code_change"):
        e["change"] = "needs code: " + str(h["code_change"])[:400]
    path = DRAFTS / f"{h['name']}.yaml"
    path.write_text(yaml.safe_dump(e, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (DRAFTS / f"{h['name']}.meta.json").write_text(json.dumps(
        {"stage": "proposed", "expected_cost_tokens": h.get("expected_cost_tokens"), "code_change": h.get("code_change", ""),
         "generator_model": MODEL, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    return path


def meta(name: str) -> dict:
    f = DRAFTS / f"{name}.meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def set_meta(name: str, **kw) -> None:
    m = meta(name)
    m.update(kw)
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / f"{name}.meta.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_propose(n: int, dry: bool, topic: str = "") -> list[str]:
    lim = limits_cfg()
    n = min(n, lim["max_hyp_per_cycle"])
    prompt = prompt_propose(n, topic)
    if dry:
        print(prompt)
        return []
    names: list[str] = []
    errors = ""
    for attempt in range(2):   # one retry that tells the model what was refused
        reply = llm(SYSTEM, prompt, 6000, "propose")
        try:
            items = extract_json(reply) or []
        except ValueError:
            items = []
            log("the reply was not JSON")
        refused = []
        for h in items[:n + 1]:
            if len(names) >= n:
                break
            errs = validate(h) if isinstance(h, dict) else ["not an object"]
            if not errs and ((DRAFTS / f"{h['name']}.yaml").exists() or (EXP / "queue" / "done" / f"{h['name']}.yaml").exists()):
                errs = ["name already used"]
            if not errs:
                dup = find_duplicate(h)
                if dup:
                    errs = [f"duplicate of {dup} (same set, budget and settings compared)"]
            if errs:
                log(f"rejected {h.get('name') if isinstance(h, dict) else '?'}: {'; '.join(errs)}")
                refused.append(f"{h.get('name') if isinstance(h, dict) else '?'}: {'; '.join(errs)}")
                continue
            p = write_draft(h)
            log(f"draft {p.name} (expected cost {h.get('expected_cost_tokens')} tokens)")
            names.append(h["name"])
        if names or not refused:
            break
        errors = " | ".join(refused)[:900]
        prompt = prompt_propose(n, topic, errors)
        log("nothing usable; asking once more with the refusal reasons")
    return names


# ------------------------------------------------------------------ applying agent: a change only in a worktree on a branch
def repo_root() -> Path:
    """The git repo the branches are made in: the checkout on a dev machine; on the host a repo seeded from the deployed archive."""
    dev = HERE.parent
    if (dev / ".git").exists() and (dev / "harness" / "sorter.py").exists():
        return dev
    src = LAB / "harness"
    repo = LAB / "autolab_repo"
    ver = (src / "VERSION").read_text(encoding="utf-8").strip() if (src / "VERSION").exists() else "unknown"
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=autolab", "-c", "user.email=noreply@anthropic.com", *a],
                                    capture_output=True, text=True)
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        git("init", "-b", "main")
        (repo / ".gitignore").write_text("VERSION\nlogs/\n__pycache__/\n*.pyc\n.venv/\n", encoding="utf-8")
    base_file = repo / ".autolab_base"
    if not base_file.exists() or base_file.read_text(encoding="utf-8").strip() != ver:
        for p in repo.iterdir():   # replace the tracked tree with the deployed one
            if p.name not in (".git", ".gitignore", ".autolab_base"):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        for p in src.iterdir():
            if p.name in ("logs", "__pycache__", ".venv"):
                continue
            shutil.copytree(p, repo / p.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "logs")) if p.is_dir() else shutil.copy2(p, repo / p.name)
        (repo / ".autolab_base").write_text(ver, encoding="utf-8")
        git("add", "-A")
        git("commit", "-m", f"base {ver}", "--allow-empty")
    return repo


def snapshot_protected() -> dict:
    """Hashes of what the model must never change: lab code, frozen sets, thresholds source (noise-floor results), configs."""
    snap = {}
    roots = [HERE / n for n in ("lab.py", "autolab.py", "dispatcher.py", "sim.py", "stream_run.py")] + [LAB / n for n in ("lab.py", "dispatcher.py", "stream_run.py", "dispatcher.yaml", "autolab.yaml")]
    files = [p for p in roots if p.exists()] + sorted((EXP / "sets").glob("*.yaml")) + sorted((EXP / "results" / "baseline0").glob("cells.jsonl"))
    for p in files:
        snap[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return snap


def check_patch(patch_files: list[str], added: list[str], removed: int) -> str | None:
    """None if the change is within the rules, else the reason it is refused."""
    if not patch_files:
        return "the patch changes nothing"
    bad = [t for t in patch_files if t in FORBIDDEN or ".." in t or t.startswith("/") or not t.startswith(ALLOWED_PREFIX)]
    if bad:
        return f"touches files outside the allowed set {ALLOWED_PREFIX}: {bad}"
    hit = [l.strip()[:80] for l in added if FORBIDDEN_ADDED.search(l)]
    if hit:
        return f"added lines hit the deny-list (lab, thresholds, keys, network, shell): {hit[:3]}"
    if len(added) + removed > limits_cfg()["max_code_lines"]:
        return f"the patch is too large ({len(added) + removed} lines, max {limits_cfg()['max_code_lines']})"
    return None


def pick_files(root: Path, want: str) -> list[Path]:
    words = {w for w in re.findall(r"[a-z_]{4,}", want.lower())}
    scored = []
    for p in list((root / "harness").glob("*.py")) + list((root / "harness").glob("*/*.py")) + [root / "run_agent.py"]:
        if not p.exists() or p.stat().st_size > 14000 or p.name == "__init__.py":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        s = sum(2 for w in words if w in p.name.lower()) + sum(1 for w in words if w in text)
        scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    out, total = [], 0
    for s, p in scored:
        if s and len(out) < 3 and total + p.stat().st_size < 24000:
            out.append(p)
            total += p.stat().st_size
    return out or [root / "harness" / "sorter.py"]


EDIT_SYSTEM = ("You edit Python source with exact search/replace blocks. Reply with JSON only: "
               '{"edits": [{"file": "harness/x.py", "old": "exact existing text", "new": "replacement"}], "note": "one sentence"}. '
               "`old` must occur exactly once in the file, copied character for character including indentation. Keep the change small, "
               "make it controllable by an environment switch with the old behaviour as default, add no imports of network or shell modules.")


def parse_edits(reply: str) -> list[dict]:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    m = re.search(r"\{.*\}", reply, re.S)
    data = json.loads(m.group(0)) if m else {}
    edits = data.get("edits") or []
    if not edits or not all(isinstance(x, dict) and {"file", "old", "new"} <= set(x) for x in edits):
        raise ValueError("no edits in the reply")
    return edits


def apply_edits(wt: Path, edits: list[dict]) -> None:
    for ed in edits:
        f = (wt / ed["file"]).resolve()
        if wt.resolve() not in f.parents or not f.exists():
            raise ValueError(f"{ed['file']}: not a file of the tree")
        text = f.read_bytes().decode("utf-8").replace("\r\n", "\n")
        if text.count(ed["old"]) != 1:
            raise ValueError(f"{ed['file']}: `old` occurs {text.count(ed['old'])} times (must be exactly 1)")
        f.write_bytes(text.replace(ed["old"], ed["new"]).encode("utf-8"))


def worktree_of(name: str) -> Path:
    return repo_root().parent / "autolab_worktrees" / name


def cmd_apply(name: str, patch_file: str | None) -> None:
    m = meta(name)
    repo = repo_root()
    branch = f"autolab/{name}"
    wt = worktree_of(name)
    g = lambda *a, cwd=None: subprocess.run(["git", "-C", str(cwd or wt), "-c", "user.name=autolab", "-c", "user.email=noreply@anthropic.com", *a],
                                           capture_output=True, text=True)
    want = m.get("code_change") or ""
    if not want and not patch_file:
        raise SystemExit("this hypothesis needs no code change: nothing to apply")
    before = snapshot_protected()
    if not wt.exists():
        r = subprocess.run(["git", "-C", str(repo), "worktree", "add", "-b", branch, str(wt), "HEAD"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr)

    def abort(why: str):
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)], capture_output=True)
        subprocess.run(["git", "-C", str(repo), "branch", "-D", branch], capture_output=True)
        set_meta(name, stage="apply_refused", apply_error=why)
        raise SystemExit(f"apply refused: {why}")

    try:
        if patch_file:   # a human-written patch goes through the same rules
            pf = wt / ".autolab.patch"
            pf.write_bytes(Path(patch_file).read_bytes().decode("utf-8").replace("\r\n", "\n").encode("utf-8"))
            if g("apply", "--check", str(pf)).returncode or g("apply", str(pf)).returncode:
                abort("the patch does not apply")
            pf.unlink()
        else:
            files = pick_files(wt, want + " " + (DRAFTS / f"{name}.yaml").read_text(encoding="utf-8")[:600])
            src = "\n\n".join(f"### {p.relative_to(wt).as_posix()}\n{p.read_text(encoding='utf-8')}" for p in files)
            err = ""
            for attempt in range(2):
                reply = llm(EDIT_SYSTEM, f"Change wanted: {want}\nOnly these paths may change: {', '.join(ALLOWED_PREFIX)}.\n"
                            f"{('Your previous edits failed: ' + err) if err else ''}\nFiles:\n{src}", 4000, "apply")
                try:
                    g("checkout", "--", ".")
                    apply_edits(wt, parse_edits(reply))
                    err = ""
                    break
                except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as ex:
                    err = str(ex)[:300]
                    log(f"edit attempt {attempt + 1} failed: {err}")
            if err:
                abort("the model's edits did not apply: " + err)
        g("add", "-A")
        names = g("diff", "--cached", "--name-only").stdout.split()
        diff = g("diff", "--cached", "-U0").stdout.splitlines()
        added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        why = check_patch(names, added, removed)
        if why:
            abort(why)
        for n_ in names:
            if n_.endswith(".py"):
                c = subprocess.run([sys.executable, "-m", "py_compile", str(wt / n_)], capture_output=True, text=True)
                if c.returncode:
                    abort(f"{n_} does not compile: {c.stderr[-300:]}")
        py = LAB / ".venv" / "bin" / "python"
        t = subprocess.run([str(py) if py.exists() else sys.executable, "tools/selftest_blocks.py"], cwd=str(wt), capture_output=True, text=True)
        if t.returncode:
            abort("selftest_blocks failed on the branch:\n" + t.stdout[-600:] + t.stderr[-600:])
    except SystemExit:
        raise
    after = snapshot_protected()
    if after != before:
        abort(f"protected files changed during apply: {[k for k in after if after[k] != before.get(k)]}")
    g("commit", "-m", f"autolab: {name}")
    sha = g("rev-parse", "--short", "HEAD").stdout.strip()
    (wt / "VERSION").write_text(f"autolab/{name} {sha} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
    set_meta(name, stage="applied", branch=branch, worktree=str(wt), commit=sha, files=names, lines=len(added) + removed)
    log(f"change committed on branch {branch} ({', '.join(names)}, {len(added)}+/{removed}-) in {wt}. Master is untouched; merge is for a human.")


def cmd_export(name: str) -> None:
    m = meta(name)
    if not m.get("branch"):
        raise SystemExit("no branch for this draft")
    repo = repo_root()
    out = subprocess.run(["git", "-C", str(repo), "diff", f"HEAD...{m['branch']}"], capture_output=True, text=True).stdout
    p = DRAFTS / f"{name}.patch"
    p.write_text(out, encoding="utf-8")
    log(f"{p} ({len(out.splitlines())} lines): review and `git apply` it in the master checkout yourself")


# ------------------------------------------------------------------ simulation as a filter, submit, verdict
def sim_py() -> Path:
    for p in (HERE / "sim.py", LAB / "harness" / "tools" / "sim.py"):
        if p.exists():
            return p
    raise SystemExit("sim.py not found")


def parse_sim(out: str) -> dict[int, float]:
    res = {}
    for m in re.finditer(r"^\|\s*(\d+)k\s*\|\s*([\d.]+)\s*\|\s*[\d.nan]+\s*\|\s*([\d.]+)\s*\|", out, re.M):
        res[int(m.group(1)) * 1000] = float(m.group(2))   # "points (mean)", the sim's own headline column
    return res


def sim_arm(env_over: dict, budgets: list[int], harness_dir: Path | None) -> dict[int, float]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HARNESS_")}
    env.update({k: str(v) for k, v in env_over.items()})
    out = {}
    for b in budgets:
        cmd = [sys.executable, str(sim_py())] + (["--harness-dir", str(harness_dir)] if harness_dir else ["--harness-dir", str(sim_py().parent.parent)])
        cmd += ["curve", "--stream", "--draws", "10", "--budgets", f"{b}:{b}:1"]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        out.update(parse_sim(r.stdout))
    return out


def cmd_sim(name: str) -> str:
    """The cheap filter before the expensive run. Returns pass | skip_real | uninformative. The simulator replays the real Scheduler and
    Predictor on recorded outcomes, so it only speaks for hypotheses that change SORTER / PREDICTOR / PUNSOLV / SOLVERS (it overestimates
    points by ~10% at a binding budget of 30k, a bias shared by the arms: the arm DIFFERENCE is used, not the level). The run is
    cancelled when the simulator predicts every variant clearly worse than the reference, or no variant different enough to ever show
    above the noise threshold at the competition budgets."""
    import yaml
    e = yaml.safe_load((DRAFTS / f"{name}.yaml").read_text(encoding="utf-8"))
    arms = e["arms"]
    keys = {k for v in arms.values() for k in ((v or {}).get("env") or {})}
    sim_keys = keys & set(SIM_KEYS)
    m = meta(name)
    if not sim_keys or keys - set(SIM_KEYS) or any((v or {}).get("batch_size") or (v or {}).get("model") for v in arms.values()):
        set_meta(name, sim="uninformative: the hypothesis changes more than scheduler-level switches", sim_verdict="uninformative")
        log(f"{name}: not a pure scheduler-level hypothesis, the simulator cannot judge it; real run decides")
        return "uninformative"
    budgets = sorted({20000, 30000, 40000, int(e.get("budget") or 30000)})
    hd = Path(m["worktree"]) if m.get("worktree") else None
    pts = {a: sim_arm({k: str(x) for k, x in ((v or {}).get("env") or {}).items()}, budgets, hd) for a, v in arms.items()}
    first = next(iter(pts))
    lines, clear_worse, too_small = [], [], []
    for a, p in pts.items():
        if a == first:
            continue
        d = {b: p.get(b, 0) - pts[first].get(b, 0) for b in budgets}
        rel = {b: d[b] / pts[first][b] for b in budgets if pts[first].get(b)}
        lines.append(f"{a} vs {first}: " + ", ".join(f"{b // 1000}k {d[b]:+.1f} ({rel.get(b, 0):+.0%})" for b in budgets))
        if rel and sum(rel.values()) / len(rel) <= -0.05 and max(rel.values()) <= 0:
            clear_worse.append(a)
        elif rel and max(abs(x) for x in rel.values()) * max(pts[first].values()) < 4:
            too_small.append(a)
    summary = {a: {f"{b // 1000}k": p.get(b) for b in budgets} for a, p in pts.items()}
    non_ref = [a for a in arms if a != first]
    verdict = "skip_real" if len(clear_worse) + len(too_small) == len(non_ref) else "pass"
    reason = ("all variants predicted >=5% worse at every budget" if len(clear_worse) == len(non_ref) else
              "predicted effect below 4 points at every budget, invisible against the noise 7.69" if len(too_small) + len(clear_worse) == len(non_ref) else "")
    set_meta(name, sim="; ".join(lines), sim_points=summary, sim_verdict=verdict, sim_reason=reason)
    log(f"{name}: simulator {verdict} {reason}: " + "; ".join(lines))
    return verdict


def exp_requests_estimate(e: dict) -> int:
    n_tasks = 100 if "100" in str(e.get("set")) else 40
    per_task = 2.5 if (e.get("budget") or 0) >= 60000 else 2.0   # at binding budgets about 2/3 of the tasks run
    return int(n_tasks * per_task * len(e["arms"]) * int(e.get("repeats", 1)) * (0.7 if (e.get("budget") or 1e9) < 50000 else 1))


def queue_depth() -> int:
    return sum(1 for k in ("pending", "running") for _ in (EXP / "queue" / k).glob("*.yaml"))


def cmd_submit(name: str) -> None:
    import yaml
    src = DRAFTS / f"{name}.yaml"
    m = meta(name)
    e = yaml.safe_load(src.read_text(encoding="utf-8"))
    if m.get("code_change") and m.get("stage") not in ("applied", "queued"):
        raise SystemExit("needs a code change: run `apply` first")
    if m.get("sim_verdict") == "skip_real":
        raise SystemExit(f"the simulator filter stopped this hypothesis ({m.get('sim_reason')}); no real run")
    lim = limits_cfg()
    if queue_depth() >= lim["max_queue_depth"]:
        raise SystemExit(f"queue has {queue_depth()} experiments (max_queue_depth={lim['max_queue_depth']})")
    est = exp_requests_estimate(e)
    if est > lim["max_expected_requests_cycle"]:
        raise SystemExit(f"estimated {est} gateway requests (max_expected_requests_cycle={lim['max_expected_requests_cycle']})")
    if m.get("worktree"):
        e["code_dir"] = m["worktree"]
        e["change"] = f"branch {m['branch']} @ {m.get('commit')}: {m.get('code_change', '')[:300]}"
    dst = EXP / "queue" / "pending" / f"{name}.yaml"
    for kind in ("pending", "running", "done"):
        if (EXP / "queue" / kind / f"{name}.yaml").exists():
            raise SystemExit(f"{name} is already in the queue ({kind})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = DRAFTS / f"{name}.queued.yaml"
    tmp.write_text(yaml.safe_dump(e, sort_keys=False, allow_unicode=True), encoding="utf-8")
    try:   # validate with the lab's own loader (holdout guard, models, sets) before it reaches the queue
        import lab
        lab.load_experiment(tmp)
    except SystemExit as ex:
        tmp.unlink()
        raise SystemExit(f"the lab refused the experiment: {ex}")
    shutil.move(str(tmp), str(dst))
    set_meta(name, stage="queued", queued_at=time.time(), gw_week_used_at_submit=gw_week_used(), expected_requests=est)
    log(f"{name} is in queue/pending (expected ~{est} gateway requests); the dispatcher runs it")


def cells_requests(name: str) -> int | None:
    f = EXP / "results" / name / "cells.jsonl"
    if not f.exists():
        return None
    return sum(int(c.get("pipeline_calls", 0)) + int(c.get("agent_calls", 0)) for c in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip()))


def cmd_verdict(name: str) -> dict:
    import lab
    nf = lab.noise_floor()
    res = lab.summarize(name)
    thr = nf["threshold_points"] if nf else None
    lines = [f"# Verdict: {name}", "", f"Set {res['set']}, code {res['code_version']}, repeats {res['repeats']}.",
             f"Acceptance threshold (noise floor of {lab.NOISE_RUN}, read from the lab, never set here): {thr} points per run "
             f"({nf['repeats_per_arm'] if nf else '-'} repeats per arm, measured on a 100-task set). `accept` needs the paired 95% interval above 0 AND a "
             "difference not smaller than the threshold; this is the lab's rule (`lab.summarize`). **Merging stays with a human.**", "",
             "| arm | points per run | tokens per run | points per 1k tokens | accuracy | points diff vs first [95% CI] | CI half-width | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    notes = []
    for a, v in res["arms"].items():
        d = v.get("points_diff_vs_first")
        ppk = v["points_per_run"] / (v["tokens_per_run"] / 1000) if v["tokens_per_run"] else 0
        hw = (d["ci95"][1] - d["ci95"][0]) / 2 if d else None
        lines.append(f"| {a} | {v['points_per_run']} | {v['tokens_per_run']} | {ppk:.2f} | {v['accuracy']} | "
                     f"{('%+.1f %s' % (d['mean'], d['ci95'])) if d else '-'} | {('%.1f' % hw) if hw is not None else '-'} | {res['verdict'].get(a)} |")
        if d and thr and hw > thr:
            notes.append(f"{a}: the interval half-width {hw:.1f} is larger than the threshold {thr}: the experiment is underpowered for this question (more repeats or tasks).")
        if d and v["tokens_per_run"] and res["arms"][next(iter(res["arms"]))]["tokens_per_run"]:
            r0 = res["arms"][next(iter(res["arms"]))]
            notes.append(f"{a}: tokens {v['tokens_per_run'] / r0['tokens_per_run'] - 1:+.0%}, points per 1k tokens {ppk:.2f} vs {r0['points_per_run'] / (r0['tokens_per_run'] / 1000):.2f} (reference).")
    req = cells_requests(name)
    lines += [""] + [f"- {n}" for n in notes] + [f"- gateway requests of the experiment (pipeline+agent calls in cells): {req}"]
    (DRAFTS / f"{name}.verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    set_meta(name, stage="verdict", verdict=res["verdict"], threshold_used=thr, experiment_requests=req, result={a: {k: v.get(k) for k in ("points_per_run", "tokens_per_run", "accuracy", "points_diff_vs_first")} for a, v in res["arms"].items()})
    print("\n".join(lines))
    return res


def cmd_list() -> None:
    for f in sorted(DRAFTS.glob("*.yaml")):
        if f.name.endswith(".queued.yaml"):
            continue
        m = meta(f.stem)
        print(f"{f.stem:<36} {m.get('stage', '?'):<13} {m.get('verdict', '') or m.get('sim_verdict', '')}")


# ------------------------------------------------------------------ the one-command cycle
def dispatcher_alive() -> bool:
    r = subprocess.run(["pgrep", "-f", "dispatcher.py run"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def wait_done(names: list[str], poll: int, max_min: int) -> None:
    t0 = time.time()
    left = set(names)
    while left and time.time() - t0 < max_min * 60:
        for nm in list(left):
            if (EXP / "queue" / "done" / f"{nm}.yaml").exists():
                left.discard(nm)
                log(f"{nm}: finished")
        if left:
            time.sleep(poll)
    if left:
        log(f"gave up waiting for {sorted(left)} after {max_min} min (they stay in the queue; run `cycle --names ...` later)")


def cmd_cycle(n: int, topic: str, names: list[str] | None, wait: bool) -> None:
    t0 = time.time()
    used0, week0 = CALLS["n"], gw_week_used()
    if not names:
        names = cmd_propose(n, False, topic)
    report = [f"# Autolab cycle {time.strftime('%Y-%m-%d %H:%M')}", "", f"Topic: {topic or '(default)'}; generator {MODEL}; hypotheses: {names}", ""]
    ready = []
    for nm in names:
        m = meta(nm)
        try:
            if m.get("stage") in ("proposed", None, "apply_refused") and m.get("code_change"):
                cmd_apply(nm, None)
            if meta(nm).get("stage") in ("proposed", "applied") and not meta(nm).get("sim_verdict"):
                cmd_sim(nm)
            if meta(nm).get("stage") in ("proposed", "applied"):
                if meta(nm).get("sim_verdict") == "skip_real":
                    set_meta(nm, stage="filtered_by_sim")
                    log(f"{nm}: stopped by the simulator filter ({meta(nm).get('sim_reason')})")
                    continue
                cmd_submit(nm)
            if meta(nm).get("stage") in ("queued", "verdict"):
                ready.append(nm)
        except SystemExit as ex:
            log(f"{nm}: stopped: {ex}")
            set_meta(nm, stage_note=str(ex)[:300])
    if ready and not dispatcher_alive():
        log("WARNING: the dispatcher is not running; the queue will not move (start `dispatcher.py run` or `lab.py queue run`)")
    if wait and ready:
        wait_done(ready, limits_cfg()["wait_poll_s"], limits_cfg()["max_wait_min"])
    for nm in names:
        if (EXP / "queue" / "done" / f"{nm}.yaml").exists() and meta(nm).get("stage") != "verdict":
            try:
                cmd_verdict(nm)
            except Exception as ex:  # noqa: BLE001
                log(f"{nm}: verdict failed: {ex}")
    week1 = gw_week_used()
    for nm in names:
        m = meta(nm)
        report += [f"## {nm}", f"- stage: {m.get('stage')} {m.get('stage_note', '')}", f"- code change: {m.get('code_change') or 'none'} "
                   f"{('(branch ' + m['branch'] + ', ' + str(m.get('lines')) + ' lines)') if m.get('branch') else ''}",
                   f"- simulator: {m.get('sim_verdict', '-')} {m.get('sim_reason', '')} {m.get('sim', '')}",
                   f"- verdict: {m.get('verdict', '-')}; result {json.dumps(m.get('result', {}), ensure_ascii=False)}",
                   f"- gateway requests of the experiment: {m.get('experiment_requests', '-')} (expected {m.get('expected_requests', '-')})", ""]
    report += [f"Autolab LLM requests this run: {CALLS['n'] - used0}; this week {week_calls()} of {limits_cfg()['weekly_llm_calls']}.",
               f"Gateway week.used: {week0} -> {week1} (includes the filler stream and everyone else).", f"Wall time {(time.time() - t0) / 60:.0f} min."]
    p = DRAFTS / f"cycle_{time.strftime('%Y%m%d_%H%M%S')}.md"
    p.write_text("\n".join(report) + "\n", encoding="utf-8")
    log(f"cycle report: {p}")


# ------------------------------------------------------------------ selftest
def selftest() -> None:
    import tempfile
    global EXP, DRAFTS, USAGE
    EXP = Path(tempfile.mkdtemp())
    DRAFTS = EXP / "autolab"
    USAGE = DRAFTS / "usage.jsonl"
    (EXP / "sets").mkdir()
    (EXP / "sets" / "math16.yaml").write_text("role: dev\ntasks: {math: 16}\nseed: 1\n")
    good = {"name": "ab_test_one", "hypothesis": "x", "set": "math16", "repeats": 1, "budget": 30000,
            "arms": {"base": {}, "v": {"env": {"HARNESS_SORTER": "fifo"}}}}
    ks = known_switches()
    if "HARNESS_SORTER" in ks:
        assert validate(good) == [], validate(good)
    assert any("unknown or forbidden" in e for e in validate(dict(good, arms={"base": {}, "v": {"env": {"HARNESS_FORCE_MODEL": "x"}}})))
    assert any("must be one of" in e for e in validate(dict(good, arms={"base": {}, "v": {"env": {"HARNESS_VERIFY": "1"}}})))
    assert any("not an available" in e for e in validate(dict(good, set="mixed100_holdout")))
    assert any("keys not allowed" in e for e in validate(dict(good, threshold=1)))
    assert any("do not differ" in e for e in validate(dict(good, arms={"a": {}, "b": {}})))
    assert any("budget" in e for e in validate(dict(good, budget=500000)))
    assert extract_json('text [{"a": 1}] tail') == [{"a": 1}]
    # duplicates compare what is tested, not names
    d1 = {"set": "s", "budget": 30000, "arms": {"a": {}, "b": {"env": {"HARNESS_SORTER": "fifo"}}}}
    d2 = {"set": "s", "budget": 30000, "arms": {"x": {"env": {}}, "y": {"env": {"HARNESS_SORTER": "fifo"}}}}
    d3 = {"set": "s", "budget": 30000, "arms": {"a": {}, "b": {"env": {"HARNESS_SORTER": "reward"}}}}
    assert diff_signature(d1) == diff_signature(d2) != diff_signature(d3)
    # patch rules
    assert check_patch(["tools/lab.py"], [], 0)
    assert check_patch(["experiments/sets/x.yaml"], [], 0)
    assert check_patch(["harness/sorter.py"], ["x = open('experiments/baseline0')"], 0)
    assert check_patch(["harness/sorter.py"], ["NOISE_FLOOR = 0"], 0)
    assert check_patch(["harness/sorter.py"], ["import subprocess"], 0)
    assert check_patch(["harness/sorter.py"], ["x = 1"], 0) is None
    assert check_patch(["harness/sorter.py"], ["x = 1"] * 200, 0)
    assert check_patch(["../x.py"], [], 0)
    # edit blocks
    t = Path(tempfile.mkdtemp())
    (t / "harness").mkdir()
    (t / "harness" / "a.py").write_text("def f():\n    return 1\n")
    apply_edits(t, parse_edits('```json\n{"edits":[{"file":"harness/a.py","old":"return 1","new":"return 2"}]}\n```'))
    assert "return 2" in (t / "harness" / "a.py").read_text()
    for bad in ({"file": "harness/a.py", "old": "nope", "new": "x"}, {"file": "../x.py", "old": "a", "new": "b"}):
        try:
            apply_edits(t, [bad])
            raise AssertionError("accepted")
        except ValueError:
            pass
    # sim table parsing
    assert parse_sim("| 20k | 151.7 | 18.0 | 138.7 | 25084 | 45.7 | 54.3 | 0 |\n| 30k | 188.7 | 14.4 | 181.0 | 1 | 1 | 1 | 0 |") == {20000: 151.7, 30000: 188.7}
    # accounting limits
    record_usage("t", {"prompt_tokens": 100, "completion_tokens": 10}, True)
    assert week_calls() == 1
    CALLS["n"] = 999
    try:
        llm("s", "u")
        raise AssertionError("limit not enforced")
    except SystemExit as ex:
        assert "limit" in str(ex)
    CALLS["n"] = 0
    # this module must not write the noise threshold
    src = Path(__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(lab\.)?NOISE_RUN\s*=", src, re.M)
    print("autolab selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose"); p.add_argument("--n", type=int, default=3); p.add_argument("--dry", action="store_true"); p.add_argument("--topic", default="")
    for c in ("sim", "submit", "verdict", "export"):
        sub.add_parser(c).add_argument("name")
    p = sub.add_parser("apply"); p.add_argument("name"); p.add_argument("--patch-file")
    p = sub.add_parser("cycle"); p.add_argument("--n", type=int, default=2); p.add_argument("--topic", default="")
    p.add_argument("--names", help="comma-separated existing drafts (skips propose)"); p.add_argument("--wait", action="store_true")
    for c in ("list", "usage", "selftest"):
        sub.add_parser(c)
    a = ap.parse_args()
    {"propose": lambda: cmd_propose(a.n, a.dry, a.topic), "sim": lambda: cmd_sim(a.name), "submit": lambda: cmd_submit(a.name),
     "verdict": lambda: cmd_verdict(a.name), "apply": lambda: cmd_apply(a.name, a.patch_file), "list": cmd_list, "export": lambda: cmd_export(a.name),
     "usage": cmd_usage, "cycle": lambda: cmd_cycle(a.n, a.topic, a.names.split(",") if a.names else None, a.wait),
     "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    main()
