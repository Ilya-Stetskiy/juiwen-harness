#!/usr/bin/env python3
"""Autonomous lab (S9): a hypothesis generator and an applying agent around the experiment queue. Lab tool, runs on the GPU host.

    python3 autolab.py propose [--n 3] [--dry]   the generator asks the LLM for hypotheses -> drafts in experiments/autolab/
    python3 autolab.py list                      drafts, their stage and verdict
    python3 autolab.py apply <name> [--patch-file F]   applying agent: code change ONLY on branch autolab/<name> (dev checkout, git)
    python3 autolab.py sim <name>                simulator check for scheduler-level hypotheses (skipped for the others)
    python3 autolab.py submit <name>             draft -> queue/pending (the dispatcher or `lab.py queue run` runs it)
    python3 autolab.py verdict <name>            result -> experiments/autolab/<name>.verdict.md (acceptance threshold from the noise floor)
    python3 autolab.py cycle [--n 2] [--run-queue]   propose -> validate -> sim -> submit [-> run queue -> verdict]
    python3 autolab.py usage                     tokens the autolab spent at the gateway

The cycle: hypothesis (YAML experiment: hypothesis, arms, set, repeats) -> code/config change only in a separate git branch
(never master; Dockerfile and requirements.txt are forbidden) -> simulation -> real run through the queue -> report and verdict
(paired intervals, acceptance threshold = noise floor, `lab.py noisefloor`) -> MERGE ONLY BY A HUMAN. Nothing here merges,
deploys to the harness or changes the submission. A hypothesis that needs no code (switches in `env:` of the arms) goes the
whole way by itself; a hypothesis with `code_change` stops after `apply` until a human deploys the branch to the host.

Model: AUTOLAB_MODEL (default qwen3.8-27b, the best of the four on the stream data: see AUTOLAB_REPORT.md) through the same proxy
(127.0.0.1:18443) as everything else; before each call the experiment gate of dispatcher.yaml is checked, a 429 is waited out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
FORBIDDEN = ("Dockerfile", "requirements.txt", "requirements-lab.txt", "tools/deploy.sh")
ALLOWED_PREFIX = ("harness/", "profiles/", "run_agent.py", "tools/lab.py", "tools/selftest_blocks.py")
SIM_KEYS = ("HARNESS_SORTER", "HARNESS_PREDICTOR", "HARNESS_PUNSOLV", "HARNESS_SOLVERS")


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ------------------------------------------------------------------ the LLM through the shared gateway
def llm(system: str, user: str, max_tokens: int = 4000, purpose: str = "") -> str:
    import dispatcher
    cfg = dispatcher.load_cfg()
    for attempt in range(6):
        ok, why = dispatcher.exp_gate(cfg, dispatcher.fractions(sr.limits()))
        if not ok:
            raise SystemExit(f"gateway gate (dispatcher.yaml): {why}")
        body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0.7,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
        except urllib.error.HTTPError as ex:
            if ex.code == 429:
                wait = 65 * (attempt + 1)
                log(f"429 from the gateway (concurrent-request limit); waiting {wait} s")
                time.sleep(wait)
                continue
            raise
        u = r.get("usage") or {}
        cost = u.get("prompt_tokens", 0) / 10.0 + u.get("completion_tokens", 0)
        DRAFTS.mkdir(parents=True, exist_ok=True)
        with open(USAGE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "model": MODEL, "purpose": purpose, "in": u.get("prompt_tokens", 0),
                                "out": u.get("completion_tokens", 0), "cost": round(cost, 1)}) + "\n")
        return r["choices"][0]["message"].get("content") or ""
    raise SystemExit("the gateway kept answering 429")


def cmd_usage() -> None:
    rows = [json.loads(l) for l in open(USAGE, encoding="utf-8")] if USAGE.exists() else []
    print(f"{len(rows)} calls, cost {sum(r['cost'] for r in rows):.0f} tokens (input/10 + output), model {MODEL}")
    for r in rows[-10:]:
        print(f"  {time.strftime('%m-%d %H:%M', time.localtime(r['ts']))} {r['purpose']:<10} {r['cost']:.0f}")


# ------------------------------------------------------------------ knowledge handed to the generator
def known_switches() -> set[str]:
    f = HERE.parent / "SCRIPTS.md"
    if not f.exists():
        f = LAB / "harness" / "SCRIPTS.md"
    return set(re.findall(r"HARNESS_[A-Z_]+", f.read_text(encoding="utf-8"))) if f.exists() else set()


def sets_available() -> dict[str, str]:
    import yaml
    out = {}
    for f in sorted((EXP / "sets").glob("*.yaml")):
        s = yaml.safe_load(f.read_text(encoding="utf-8"))
        if s.get("role") == "dev":   # holdout is never offered; fresh sets are not offered either (they change)
            out[f.stem] = f"tasks {s.get('tasks')}"
    return out


def digest() -> str:
    """Short summary of finished experiments: name, what was tested, per-arm points and verdict."""
    import yaml
    rows = []
    for f in sorted((EXP / "queue" / "done").glob("*.yaml")):
        text = f.read_text(encoding="utf-8")
        try:
            e = yaml.safe_load(text.split("# ---- written by lab.py")[0])
            blk = yaml.safe_load(text.split("# ---- written by lab.py")[1].split("\n", 1)[1]) if "# ---- written by lab.py" in text else {}
        except Exception:  # noqa: BLE001
            continue
        res = (blk or {}).get("result") or {}
        arms = {a: (v.get("points_per_run"), v.get("tokens_per_run"), v.get("accuracy")) for a, v in (res.get("arms") or {}).items()}
        rows.append(f"- {e.get('name')}: {(e.get('hypothesis') or '').strip()[:240]} | arms (points, tokens/run, accuracy): {arms} | verdict {res.get('verdict')}")
    return "\n".join(rows[-14:])


SYSTEM = ("You are the hypothesis generator of an experiment lab for an LLM task-solving harness. You propose small, testable "
          "experiments that change harness behaviour ONLY through the existing environment switches. Reply with JSON only.")


def prompt_propose(n: int) -> str:
    sw = ", ".join(sorted(known_switches()))
    return f"""Known facts (measured):
- Budget 100000 tokens, cost = input/10 + output. baseline0 spends about 61k of 100k, accuracy about 0.68.
- Verification with an LLM judge cost 56% of all tokens on baseline0; the judge is useless on search (AUC 0.5-0.6), useful on math (0.9) and data_analysis (0.8).
- Rechecking a doubtful answer with the same pipeline and a stronger model replaced only 3 of 27 answers, points unchanged.
- Batching 4 math tasks into one request with prompt `compute` kept accuracy (0.91 vs 0.75, 16 tasks) and cut requests per task from 1.25 to 0.31.
- A checker model of another family (qwen3.8-27b) agreeing with the first answer is a strong signal (verifier level `other`).
Finished experiments:
{digest()}

Environment switches you may use in `env:` of an arm (nothing else is allowed): {sw}
Frozen sets you may use (`set:`): {json.dumps(sets_available())}
Models: qwen3.6-35b-a3b-noreason, qwen3.6-35b-a3b, qwen3.8-27b-noreason, qwen3.8-27b, auto.

Propose {n} different hypotheses likely to be useful. Each: JSON object with keys
name (snake_case, unique, start with ab_), hypothesis (2-4 sentences: what changes, why, what you expect, how it can fail),
set (one of the sets above), model (one model or auto), repeats (1 or 2), budget (integer, at most 100000),
arms (object: arm name -> {{env: {{SWITCH: "value"}}, batch_size: optional int}}); the FIRST arm is the reference with an empty env,
expected_cost_tokens (your estimate of the gateway tokens the whole experiment needs), code_change (empty string unless an existing switch cannot express it).
Return a JSON list of {n} objects and nothing else."""


def extract_json(text: str):
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else None


def validate(h: dict) -> list[str]:
    errs = []
    for k in ("name", "hypothesis", "set", "arms"):
        if not h.get(k):
            errs.append(f"missing {k}")
    if errs:
        return errs
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", str(h["name"])):
        errs.append("bad name")
    if h["set"] not in sets_available():
        errs.append(f"set {h['set']} is not an available dev set")
    arms = h["arms"]
    if not isinstance(arms, dict) or not 2 <= len(arms) <= 4:
        errs.append("2-4 arms")
        return errs
    first = next(iter(arms.values())) or {}
    if first.get("env"):
        errs.append("the first arm must be the reference (empty env)")
    ks = known_switches()
    for a, v in arms.items():
        for k in (v or {}).get("env") or {}:
            if k not in ks:
                errs.append(f"arm {a}: unknown switch {k}")
    if int(h.get("repeats", 1)) not in (1, 2):
        errs.append("repeats 1 or 2")
    if int(h.get("budget", 100000)) > 100000:
        errs.append("budget above 100000")
    if int(h.get("expected_cost_tokens", 0) or 0) > 150000:
        errs.append("expected cost above 150000 tokens")
    return errs


def write_draft(h: dict) -> Path:
    import yaml
    DRAFTS.mkdir(parents=True, exist_ok=True)
    e = {"name": h["name"], "author": "orchestrator", "hypothesis": h["hypothesis"].strip(), "backend": "gateway",
         "model": h.get("model", "qwen3.6-35b-a3b-noreason"), "set": h["set"], "repeats": int(h.get("repeats", 1)),
         "budget": int(h.get("budget", 100000)), "fallback": "on",
         "arms": {a: {k: v for k, v in (spec or {}).items()} for a, spec in h["arms"].items()}}
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
    (DRAFTS / f"{name}.meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


def cmd_propose(n: int, dry: bool) -> list[str]:
    prompt = prompt_propose(n)
    if dry:
        print(prompt)
        return []
    reply = llm(SYSTEM, prompt, 5000, "propose")
    try:
        items = extract_json(reply) or []
    except ValueError:
        items = []
        log("the reply was not JSON; nothing written")
    names = []
    for h in items:
        errs = validate(h) if isinstance(h, dict) else ["not an object"]
        if errs:
            log(f"rejected {h.get('name') if isinstance(h, dict) else '?'}: {'; '.join(errs)}")
            continue
        if (DRAFTS / f"{h['name']}.yaml").exists() or (EXP / "queue" / "done" / f"{h['name']}.yaml").exists():
            log(f"rejected {h['name']}: name already used")
            continue
        p = write_draft(h)
        log(f"draft {p.name} (expected cost {h.get('expected_cost_tokens')} tokens)")
        names.append(h["name"])
    return names


# ------------------------------------------------------------------ applying agent: a change only on a branch
def cmd_apply(name: str, patch_file: str | None) -> None:
    m = meta(name)
    repo = HERE.parent
    if subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-dir"], capture_output=True).returncode != 0:
        raise SystemExit("apply needs the git checkout (dev machine); the host has only the deployed archive")
    branch = f"autolab/{name}"
    wt = repo.parent / "autolab_worktrees" / name
    if patch_file:
        patch = Path(patch_file).read_bytes().decode("utf-8")
    else:
        want = m.get("code_change") or ""
        if not want:
            raise SystemExit("this hypothesis needs no code change: nothing to apply")
        files = [f for f in ("harness/executor.py", "harness/judge.py", "harness/batching.py") if (repo / f).exists()]
        src = "\n\n".join(f"### {f}\n{(repo / f).read_text(encoding='utf-8')[:6000]}" for f in files)
        patch = llm("You write unified diffs (git apply format) and nothing else.",
                    f"Change wanted: {want}\nOnly these paths may change: {', '.join(ALLOWED_PREFIX)}. Files:\n{src}", 4000, "apply")
        patch = re.sub(r"^```\w*\n|\n```\s*$", "", patch.strip()) + "\n"
    touched = re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)
    bad = [t for t in touched if t in FORBIDDEN or not t.startswith(ALLOWED_PREFIX)]
    if not touched or bad:
        raise SystemExit(f"patch refused: touches {bad or 'nothing'}")
    run = lambda *a: subprocess.run(["git", "-C", str(wt), *a], capture_output=True, text=True)
    if not wt.exists():
        r = subprocess.run(["git", "-C", str(repo), "worktree", "add", "-b", branch, str(wt), "HEAD"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr)
    pf = wt / ".autolab.patch"
    pf.write_bytes(patch.replace(chr(13) + chr(10), chr(10)).encode("utf-8"))   # LF only: git apply rejects CRLF patches
    if run("apply", "--check", str(pf)).returncode or run("apply", str(pf)).returncode:
        raise SystemExit("the patch does not apply")
    pf.unlink()
    t = subprocess.run([sys.executable, "tools/selftest_blocks.py"], cwd=str(wt), capture_output=True, text=True)
    if t.returncode:
        raise SystemExit("selftest_blocks failed on the branch:\n" + t.stdout[-800:] + t.stderr[-800:])
    run("add", "-A")
    run("-c", "user.name=autolab", "-c", "user.email=noreply@anthropic.com", "commit", "-m", f"autolab: {name}")
    set_meta(name, stage="applied", branch=branch, worktree=str(wt))
    log(f"change committed on branch {branch} in {wt}. Master is untouched; deploy and merge are for a human.")


# ------------------------------------------------------------------ simulation, submit, verdict
def cmd_sim(name: str) -> None:
    import yaml
    e = yaml.safe_load((DRAFTS / f"{name}.yaml").read_text(encoding="utf-8"))
    keys = {k for v in e["arms"].values() for k in ((v or {}).get("env") or {}) if k in SIM_KEYS}
    if not keys:
        set_meta(name, sim="skipped (no scheduler-level switch)")
        log(f"{name}: no scheduler-level switch, simulation skipped")
        return
    out = []
    for a, v in e["arms"].items():
        env = dict(os.environ, **{k: str(x) for k, x in ((v or {}).get("env") or {}).items()})
        r = subprocess.run([sys.executable, str(HERE / "sim.py"), "curve", "--stream", "--draws", "3", "--budgets", "30000:100000:35000"],
                           env=env, capture_output=True, text=True)
        out.append(f"arm {a}:\n{r.stdout[-700:]}")
    set_meta(name, sim="\n".join(out))
    log(f"{name}: simulation done, see meta")


def cmd_submit(name: str) -> None:
    src = DRAFTS / f"{name}.yaml"
    m = meta(name)
    if m.get("code_change") and m.get("stage") != "applied":
        raise SystemExit("needs a code change: run `apply` first and have a human deploy the branch")
    dst = EXP / "queue" / "pending" / f"{name}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    set_meta(name, stage="queued")
    log(f"{name} is in queue/pending")


def cmd_verdict(name: str) -> None:
    import lab
    nf = lab.noise_floor()
    res = lab.summarize(name)
    lines = [f"# Verdict: {name}", "", f"Set {res['set']}, code {res['code_version']}, repeats {res['repeats']}.",
             f"Acceptance threshold (noise floor of {lab.NOISE_RUN}): {nf['threshold_points'] if nf else 'not available'} points per run "
             f"({nf['repeats_per_arm'] if nf else '-'} repeats per arm). `accept` needs the paired 95% interval above 0 AND a difference "
             "not smaller than the threshold. **Merging stays with a human.**", "",
             "| arm | points per run | tokens per run | accuracy | points diff vs first [95% CI] | verdict |", "|---|---|---|---|---|---|"]
    for a, v in res["arms"].items():
        d = v.get("points_diff_vs_first")
        lines.append(f"| {a} | {v['points_per_run']} | {v['tokens_per_run']} | {v['accuracy']} | "
                     f"{('%+.1f %s' % (d['mean'], d['ci95'])) if d else '-'} | {res['verdict'].get(a)} |")
    (DRAFTS / f"{name}.verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    set_meta(name, stage="verdict", verdict=res["verdict"])
    print("\n".join(lines))


def cmd_list() -> None:
    for f in sorted(DRAFTS.glob("*.yaml")):
        m = meta(f.stem)
        print(f"{f.stem:<32} {m.get('stage', '?'):<9} {m.get('verdict', '')}")


def cmd_cycle(n: int, run_queue: bool) -> None:
    names = cmd_propose(n, False)
    ready = []
    for nm in names:
        if meta(nm).get("code_change"):
            log(f"{nm} needs a code change: stops before `apply` (human deploys the branch)")
            continue
        cmd_sim(nm)
        cmd_submit(nm)
        ready.append(nm)
    if run_queue and ready:
        subprocess.run([sys.executable, str(LAB / "lab.py"), "queue", "run"], cwd=str(LAB))
        for nm in ready:
            if (EXP / "queue" / "done" / f"{nm}.yaml").exists():
                cmd_verdict(nm)


def selftest() -> None:
    good = {"name": "ab_test_one", "hypothesis": "x", "set": "math16", "repeats": 1, "budget": 50000,
            "arms": {"base": {}, "v": {"env": {"HARNESS_VERIFY": "judge"}}}}
    import tempfile
    global EXP
    EXP = Path(tempfile.mkdtemp())
    (EXP / "sets").mkdir()
    (EXP / "sets" / "math16.yaml").write_text("role: dev\ntasks: {math: 16}\nseed: 1\n")
    assert validate(good) == [] or all("unknown switch" in e for e in validate(good)), validate(good)
    bad = dict(good, arms={"base": {"env": {"HARNESS_VERIFY": "judge"}}, "v": {}})
    assert any("reference" in e for e in validate(bad))
    assert any("not an available" in e for e in validate(dict(good, set="mixed100_holdout")))
    assert extract_json('text [{"a": 1}] tail') == [{"a": 1}]
    print("autolab selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose"); p.add_argument("--n", type=int, default=3); p.add_argument("--dry", action="store_true")
    for c in ("sim", "submit", "verdict"):
        sub.add_parser(c).add_argument("name")
    p = sub.add_parser("apply"); p.add_argument("name"); p.add_argument("--patch-file")
    p = sub.add_parser("cycle"); p.add_argument("--n", type=int, default=2); p.add_argument("--run-queue", action="store_true")
    for c in ("list", "usage", "selftest"):
        sub.add_parser(c)
    a = ap.parse_args()
    {"propose": lambda: cmd_propose(a.n, a.dry), "sim": lambda: cmd_sim(a.name), "submit": lambda: cmd_submit(a.name),
     "verdict": lambda: cmd_verdict(a.name), "apply": lambda: cmd_apply(a.name, a.patch_file), "list": cmd_list,
     "usage": cmd_usage, "cycle": lambda: cmd_cycle(a.n, a.run_queue), "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    main()
