#!/usr/bin/env python3
"""Experiment runner of the lab stand (runs on the GPU host next to stream_run.py).

An experiment is a YAML file: one fixed set of tasks, several ARMS (an arm = a set of environment overrides for
the harness, e.g. HARNESS_VARIANT_MATH=v1), REPEATS of every arm. All arms see exactly the same tasks (paired
design), repeats are interleaved (repeat-major) so that drift of the gateway hits every arm equally.

What it adds on top of the ad-hoc A/B scripts:
  * health check  - the proxy must be up and not halted, the weekly gateway usage must be under the guard, every
                    run is checked for infrastructure errors (they are retried once and excluded from scoring,
                    never counted as wrong answers), a run that produced no answers at all aborts the experiment,
                    and an arm may declare `expect: {math: v1}` which is checked against the `variant` events.
  * resumable     - finished (arm, repeat) pairs are skipped when the experiment is started again.
  * report        - accuracy and tokens per arm and type, noise floor (sd between repeats of one arm), and a paired
                    comparison of every arm against the first one with a bootstrap confidence interval over tasks.
  * metadata only - prompts and answers are never written to the results directory.

usage:
  lab.py run <experiment.yaml> [--dry]     run (or resume) and write the report
  lab.py report <name>                     rebuild the report from cells.jsonl
  lab.py sets list|freeze|verify [name..]  frozen named task sets (experiments/sets/*.yaml)
  lab.py queue submit <file..>             put experiment files into experiments/queue/pending
  lab.py queue run [--detach]              run pending experiments one after another (resumes a broken one)
  lab.py queue status | retry <name>       what is pending/running/done; put an aborted one back to pending
  lab.py selftest                          check the statistics code without any model call
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stream_run as sr  # noqa: E402  (sampler, grader, transient-error test, proxy helpers)

LAB = sr.LAB
# LAB_EXPERIMENTS / LAB_WORK let a test copy of the lab use its own results, sets, queue and scratch space
EXP_DIR = Path(os.environ.get("LAB_EXPERIMENTS") or LAB / "experiments")
RESULTS = EXP_DIR / "results"
SETS_DIR = EXP_DIR / "sets"
QUEUE = EXP_DIR / "queue"
WORK = Path(os.environ.get("LAB_WORK") or "/tmp/hl_lab")
SET_KEYS = ("tasks", "sources", "unanswerable", "rewards", "seed", "exclude")
sr.TAG = "_lab"          # state and log files of the shared sampler must not collide with the stream


class Abort(Exception):
    pass


def code_version() -> str:
    """Git hash of the deployed harness (written by tools/deploy.sh), so a result can be tied to its code."""
    try:
        return (LAB / "harness" / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown (not deployed with deploy.sh)"


def say(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ------------------------------------------------------------------ experiment definition
def load_set(name: str) -> dict:
    """A frozen named task set: experiments/sets/<name>.yaml (seed, tasks, sources, unanswerable, rewards, role)."""
    import yaml
    f = SETS_DIR / f"{name}.yaml"
    if not f.exists():
        raise SystemExit(f"unknown task set '{name}' (looked for {f})")
    s = yaml.safe_load(f.read_text(encoding="utf-8"))
    for k in ("tasks", "seed"):
        if k not in s:
            raise SystemExit(f"{f}: missing key '{k}'")
    s["name"] = name
    s.setdefault("role", "dev")
    if s["role"] not in ("dev", "holdout"):
        raise SystemExit(f"{f}: role must be dev or holdout")
    return s


def load_experiment(path: Path, allow_holdout: bool = False) -> dict:
    import yaml
    e = yaml.safe_load(path.read_text(encoding="utf-8"))
    if e.get("set"):   # `set: name` replaces tasks/sources/unanswerable/rewards/seed with the frozen set
        clash = [k for k in SET_KEYS if k in e]
        if clash:
            raise SystemExit(f"{path}: 'set' is given together with {clash}; a frozen set defines these keys")
        s = load_set(e["set"])
        if s["role"] == "holdout" and not allow_holdout:
            raise SystemExit(f"{path}: set '{s['name']}' is a HOLDOUT set (not to be used while tuning); "
                             "run with --allow-holdout only for the final check")
        for k in SET_KEYS:
            if k in s:
                e[k] = s[k]
        e["set_role"], e["set_frozen"] = s["role"], s.get("frozen")
    for k in ("name", "tasks", "arms"):
        if k not in e:
            raise SystemExit(f"{path}: missing key '{k}'")
    e.setdefault("hypothesis", "")
    e.setdefault("id", e.get("name"))      # fields of spec 4.6 that the runner only carries along
    e.setdefault("author", "human")        # human | orchestrator
    e.setdefault("change", "")             # what differs from the code/config of the baseline (free text or branch)
    e.setdefault("backend", "gateway")
    e.setdefault("model", "qwen3.6-35b-a3b-noreason")
    e.setdefault("repeats", 3)
    e.setdefault("fallback", "off")
    e.setdefault("workers", 5)
    e.setdefault("rpm", 55)
    e.setdefault("seed", 1000)
    e.setdefault("week_guard", 600_000)
    e.setdefault("max_infra_frac", 0.10)
    e.setdefault("budget", 1_000_000_000)
    bad = [t for t in e["tasks"] if t not in sr.SOURCES]
    if bad:
        raise SystemExit(f"unknown task types {bad}; known: {list(sr.SOURCES)}")
    for m in [e["model"]] + [a.get("model") for a in e["arms"].values() if a.get("model")]:
        if m != "auto" and m not in sr.ALL_MODELS:     # `model` may be set per arm (variant): one of the 4 or `auto`
            raise SystemExit(f"unknown model '{m}'; known: {sr.ALL_MODELS} or auto")
    if len(e["arms"]) < 1:
        raise SystemExit("an experiment needs at least one arm")
    return e


def build_task_set(e: dict, out: Path) -> dict:
    """Fixed task set for the whole experiment: `tasks: {math: 30, search: 10}` drawn without repetition."""
    shutil.rmtree(out, ignore_errors=True)
    (out / "tasks").mkdir(parents=True)
    sm = sr.Sampler(int(e["seed"]))
    grading: dict[str, dict] = {}
    skip_ids: set[str] = set()      # `exclude: [set names]`: a holdout set shares no task with the sets it excludes
    for other in e.get("exclude") or []:
        skip_ids |= set(build_named_set(other)[1])
    for t, n in e["tasks"].items():
        got, n_un, guard = 0, 0, 0
        pool = e.get("sources", {}).get(t) or sr.SOURCES[t]
        quota = (e.get("unanswerable") or {}).get(t)   # exact number of unanswerable tasks of this type, if given
        while got < n and guard < n * 300:
            guard += 1
            r = sm.next_row(sm.rng.choice(pool))
            if r is None or r["task_id"] in grading or r["task_id"] in skip_ids:
                continue
            is_un = r["check"] == "unanswerable"
            if quota is not None and ((is_un and n_un >= quota) or (not is_un and got - n_un >= n - quota)):
                continue
            n_un += is_un
            rw = e.get("rewards")
            if rw:  # random but reproducible rewards, as in the competition (points unrelated to difficulty)
                r["reward"] = random.Random(f"{rw.get('seed', 1)}:{r['task_id']}").randint(int(rw["low"]), int(rw["high"]))
            sr.write_task_dir(r, out / "tasks" / r["task_id"])
            grading[r["task_id"]] = sr.grading_entry(r, _source_of(r, pool), len(grading))
            got += 1
        if got < n:
            raise Abort(f"could only draw {got} of {n} {t} tasks")
    return grading


def set_hash(grading: dict) -> str:
    return hashlib.sha1("\n".join(sorted(grading)).encode()).hexdigest()[:12]


def set_summary(grading: dict) -> dict:
    by_type = collections.Counter(g["task_type"] for g in grading.values())
    return {"set_hash": set_hash(grading), "n_tasks": len(grading),
            "n_unanswerable": sum(bool(g["unanswerable"]) for g in grading.values()), "by_type": dict(sorted(by_type.items()))}


def _source_of(row: dict, pool: list[str]) -> str:
    tid = row["task_id"]
    best = max((s for s in pool if tid.startswith(s)), key=len, default=pool[0])
    return best


# ------------------------------------------------------------------ gateway / infrastructure checks
def check_infrastructure(e: dict) -> str:
    """Returns the base URL of the backend or raises Abort with the reason."""
    if e["backend"] == "local":
        if not sr.shim_ready():
            raise Abort("local backend selected but llama-server (:8011) or the shim (:18444) is not reachable")
        return f"http://127.0.0.1:{sr.SHIM_PORT}"
    lim = sr.limits()
    if lim is not None:
        week = (lim.get("chat", {}).get("week") or {}).get("used", 0)
        if week >= e["week_guard"]:
            raise Abort(f"weekly gateway usage {week} >= guard {e['week_guard']}")
    st = sr.proxy_state()
    if st is None:
        sr.start_proxy(e["rpm"])
        st = sr.proxy_state()
    if st is None:
        raise Abort("gateway proxy is not running and could not be started")
    if st.get("stopped"):
        raise Abort("gateway proxy is halted (429/limits guard): not restarting it from the lab")
    return f"http://127.0.0.1:{sr.PROXY_PORT}"


# ------------------------------------------------------------------ one harness run
def run_harness(e: dict, arm: dict, tasks_dir: Path, work: Path, base_url: str) -> None:
    env = dict(os.environ, LLM_BASE_URL=base_url, LLM_API_KEY="x",
               HARNESS_CWD=str(work / "cwd"), BUDGET=str(e["budget"]), HARNESS_LOG_DIR=str(work / "logs"),
               HARNESS_WORKERS=str(e["workers"]), HARNESS_LIMITS="off", HARNESS_FALLBACK=str(e["fallback"]))
    model = arm.get("model", e["model"])   # an arm may carry its own model (S4); `auto` leaves the choice to the harness
    if model != "auto":
        env["HARNESS_FORCE_MODEL"] = model
    else:
        env.pop("HARNESS_FORCE_MODEL", None)
    for k, v in {**(e.get("env") or {}), **(arm.get("env") or {})}.items():  # arm overrides experiment-wide env
        env[str(k)] = str(v)
    if arm.get("batch_size"):   # arm parameter of S11: same as env HARNESS_BATCH_SIZE
        env["HARNESS_BATCH_SIZE"] = str(arm["batch_size"])
    with open(work / "stderr.txt", "w") as err:
        subprocess.run([str(LAB / ".venv" / "bin" / "python"), str(LAB / "harness" / "run_agent.py"),
                        "--tasks-dir", str(tasks_dir)], cwd=str(LAB), env=env, stdout=subprocess.DEVNULL, stderr=err,
                       stdin=subprocess.DEVNULL, timeout=3 * 3600)


def variants_seen(log_dir: Path) -> dict[str, collections.Counter]:
    seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in sr.journal_rows(log_dir):
        if r.get("event") == "variant":
            seen[r.get("pipeline", "?")][r.get("name", "?")] += 1
    return seen


def mark_unfunded(rows: list[dict], journal: list[dict]) -> int:
    """A task that never started in a run that FINISHED (`run_end` in the journal) was left out by the budget (or the
    unsolvable skip), it is not an infrastructure failure: it must not be re-run with a fresh budget. Without
    `run_end` the run itself was cut and `not_run` stays transient."""
    if not any(r.get("event") == "run_end" for r in journal):
        return 0
    n = 0
    for r in rows:
        if r["status"] == "not_run":
            r["status"], n = "unfunded", n + 1
    return n


def execute(e: dict, arm_name: str, arm: dict, grading: dict, base: Path, base_url: str, label: str) -> list[dict]:
    """Run the harness on all tasks of `grading` (copied from the fixed set) and return graded rows."""
    work = WORK / e["name"] / label
    shutil.rmtree(work, ignore_errors=True)
    (work / "tasks").mkdir(parents=True)
    for tid in grading:
        shutil.copytree(base / "tasks" / tid, work / "tasks" / tid)
    run_harness(e, arm, work / "tasks", work, base_url)
    rows = sr.grade_chunk(arm.get("model", e["model"]), 0, work / "tasks", work / "logs", grading)
    jrows = sr.journal_rows(work / "logs")
    n_unfunded = mark_unfunded(rows, jrows)
    if n_unfunded:
        say(f"  {label}: {n_unfunded} tasks were never started (budget), counted as not solved, not retried")
    seen = variants_seen(work / "logs")
    for kind, want in (arm.get("expect") or {}).items():
        got = dict(seen.get(kind, {}))
        if got and set(got) != {want}:
            raise Abort(f"arm '{arm_name}' expected variant '{want}' for {kind} but the journal shows {got}: "
                        "the environment override did not take effect")
        if not got and any(r["type"] == kind for r in rows):
            raise Abort(f"arm '{arm_name}': no `variant` events for {kind}; cannot confirm variant '{want}'")
    keep = RESULTS / e["name"] / "journals"   # metadata journals only (events, tokens, statuses), never prompts
    keep.mkdir(parents=True, exist_ok=True)
    with open(keep / f"{label}.jsonl", "w", encoding="utf-8") as out:
        for r in jrows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    return rows


def run_with_health(e: dict, arm_name: str, arm: dict, grading: dict, base: Path, base_url: str, rep: int) -> list[dict]:
    rows = execute(e, arm_name, arm, grading, base, base_url, f"{arm_name}_{rep}")
    answered = sum(r["answered"] for r in rows)
    if len(rows) >= 5 and answered == 0:
        raise Abort(f"arm '{arm_name}' repeat {rep}: no task produced an answer; the setup is broken "
                    "(see /tmp/hl_lab and stderr), aborting before wasting more gateway tokens")
    by_id = {r["task_id"]: r for r in rows}
    broken = [r["task_id"] for r in rows if sr.is_transient(r)]
    if broken:
        say(f"  {arm_name} #{rep}: {len(broken)} infrastructure-broken cells, retrying once")
        time.sleep(30)
        sub = {t: grading[t] for t in broken}
        for r in execute(e, arm_name, arm, sub, base, check_infrastructure(e), f"{arm_name}_{rep}_retry"):
            r["attempt"] = 2
            by_id[r["task_id"]] = r
    out = list(by_id.values())
    left = sum(sr.is_transient(r) for r in out)
    if left > e["max_infra_frac"] * len(out):
        raise Abort(f"arm '{arm_name}' repeat {rep}: {left}/{len(out)} cells still infrastructure-broken "
                    f"(limit {e['max_infra_frac']:.0%})")
    for r in out:
        r["arm"], r["repeat"], r["infra"] = arm_name, rep, bool(sr.is_transient(r))
    return out


# ------------------------------------------------------------------ report
def bootstrap_ci(diffs: list[float], n: int = 4000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def group_labels(cells: list[dict]) -> list[str]:
    """all, every type, and answerable / unanswerable halves of a type that has both."""
    types = sorted({c["type"] for c in cells})
    labels = ["all"] + types
    for t in types:
        if len({bool(c["unanswerable"]) for c in cells if c["type"] == t}) == 2:
            labels += [f"{t} answerable", f"{t} unanswerable"]
    return labels


def in_group(c: dict, label: str) -> bool:
    if label == "all":
        return True
    if label.endswith(" unanswerable"):
        return c["type"] == label[:-13] and bool(c["unanswerable"])
    if label.endswith(" answerable"):
        return c["type"] == label[:-11] and not c["unanswerable"]
    return c["type"] == label


def skip_table(good: list[dict], arms: list[str], runs: list[dict]) -> list[str]:
    """How well each arm separates unanswerable tasks (no answer written) from answerable ones, and how long it takes."""
    out = ["", "## Unanswerable tasks and time", "",
           "Recall: share of unanswerable tasks left without an answer (what the competition wants). Precision: share "
           "of tasks left without an answer that were really unanswerable (the rest are solvable tasks we gave up). "
           "Seconds are the mean time per task and the wall time of a whole run.", "",
           "| arm | unanswerable cells | recall | precision | solvable tasks left unanswered | seconds per task | run seconds |",
           "|---|---|---|---|---|---|---|"]
    for a in arms:
        sel = [c for c in good if c["arm"] == a]
        un = [c for c in sel if c["unanswerable"]]
        silent = [c for c in sel if not c["answered"]]
        hit = [c for c in un if not c["answered"]]
        gave_up = [c for c in silent if not c["unanswerable"]]
        secs = [c["seconds"] for c in sel if c.get("seconds") is not None]
        wall = [r["seconds"] for r in runs if r["arm"] == a]
        out.append(f"| {a} | {len(un)} | {(len(hit) / len(un)) if un else float('nan'):.3f} | "
                   f"{(len(hit) / len(silent)) if silent else float('nan'):.3f} | "
                   f"{len(gave_up) / max(1, len({c['repeat'] for c in sel})):.1f} per run | "
                   f"{(sum(secs) / len(secs)) if secs else float('nan'):.1f} | "
                   f"{(sum(wall) / len(wall)) if wall else float('nan'):.0f} |")
    return out


def curve_table(d: Path, good: list[dict], arms: list[str], step: int = 10_000) -> list[str]:
    """Reward-weighted points reached by the time `x` tokens were spent, from the stored journals (task_end totals)."""
    jdir = d / "journals"
    if not jdir.is_dir():
        return []
    cell = {(c["arm"], c["repeat"], c["task_id"]): c for c in good}
    curves: dict[str, list[list[tuple[float, float]]]] = collections.defaultdict(list)
    for a in arms:
        for rep in sorted({c["repeat"] for c in good if c["arm"] == a}):
            f = jdir / f"{a}_{rep}.jsonl"
            if not f.exists():
                continue
            ends = sorted((e for e in map(json.loads, open(f, encoding="utf-8")) if e.get("event") == "task_end"),
                          key=lambda e: e["ts"])
            pts, acc = [], 0.0
            for e in ends:
                c = cell.get((a, rep, e["task"]))
                if c is None:
                    continue
                acc += c["reward"] * c["correct"]
                pts.append((e["total"], acc))
            curves[a].append(pts)
    if not curves:
        return []
    top = max(p[-1][0] for runs_ in curves.values() for p in runs_ if p)
    levels = list(range(step, int(top // step + 1) * step + 1, step))
    out = ["", "## Points by tokens spent", "",
           "Mean over repeats of the points collected by the time the run had spent `x` tokens (a task counts when it "
           "finished). The last column is the end of the run: final points and total tokens.", "",
           "| arm | " + " | ".join(f"{x // 1000}k" for x in levels) + " | end (points / tokens) |",
           "|---|" + "---|" * (len(levels) + 1)]
    for a, runs_ in curves.items():
        cols = []
        for x in levels:
            vals = [max((p for t, p in run if t <= x), default=0.0) for run in runs_]
            cols.append(f"{sum(vals) / len(vals):.1f}")
        end_p = sum(run[-1][1] for run in runs_ if run) / len(runs_)
        end_t = sum(run[-1][0] for run in runs_ if run) / len(runs_)
        out.append(f"| {a} | " + " | ".join(cols) + f" | {end_p:.1f} / {end_t:.0f} |")
    return out


def _arm_models(meta: dict) -> str:
    per = {a: v.get("model") for a, v in meta["arms"].items() if v.get("model")}
    return (" (per arm: " + ", ".join(f"{a}=`{m}`" for a, m in per.items()) + ")") if per else ""


def make_report(name: str) -> str:
    d = RESULTS / name
    meta = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cells = [json.loads(l) for l in open(d / "cells.jsonl", encoding="utf-8") if l.strip()]
    runs = [json.loads(l) for l in open(d / "runs.jsonl", encoding="utf-8") if l.strip()] if (d / "runs.jsonl").exists() else []
    arms = list(meta["arms"])
    lines = [f"# Experiment `{name}`", "", f"Hypothesis: {meta.get('hypothesis') or '-'}", "",
             f"Model `{meta['model']}`{_arm_models(meta)}, backend `{meta['backend']}`, fallback `{meta['fallback']}`, "
             f"{meta['n_tasks']} tasks {meta['tasks']}, repeats {meta['repeats']}, seed {meta['seed']}.",
             f"Code version: {meta.get('code_version', 'not recorded')}. Task set: {meta.get('set_name') or 'ad hoc'} "
             f"[{meta.get('set_role', 'adhoc')}] {meta.get('set_hash', 'not recorded')}, "
             f"unanswerable tasks in the set: {meta.get('n_unanswerable', 'not recorded')}.", ""]
    good = [c for c in cells if not c["infra"]]
    infra = len(cells) - len(good)
    lines.append(f"Cells: {len(cells)}; infrastructure-broken (excluded from scoring): {infra}.")
    lines += ["", "## Accuracy and cost per arm", "",
              "| arm | type | cells | accuracy | points per run | sd between repeats | tokens/task | tokens per correct |",
              "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        for t in group_labels(good):
            sel = [c for c in good if c["arm"] == a and in_group(c, t)]
            if not sel:
                continue
            per_rep = collections.defaultdict(list)
            pts_rep: dict[int, float] = collections.defaultdict(float)
            for c in sel:
                per_rep[c["repeat"]].append(c["correct"])
                pts_rep[c["repeat"]] += c["reward"] * c["correct"]
            points = sum(pts_rep.values()) / len(pts_rep)
            accs = [sum(v) / len(v) for v in per_rep.values()]
            acc = sum(c["correct"] for c in sel) / len(sel)
            sd = statistics.pstdev(accs) if len(accs) > 1 else float("nan")
            cost = sum(c["hybrid_cost"] for c in sel) / len(sel)
            ok = sum(c["correct"] for c in sel)
            lines.append(f"| {a} | {t} | {len(sel)} | {acc:.3f} | {points:.1f} | {sd:.3f} | {cost:.1f} | "
                         f"{(sum(c['hybrid_cost'] for c in sel) / ok) if ok else float('nan'):.1f} |")
    if any(v.get("model") for v in meta["arms"].values()):   # arms with different models: what each one cost per task
        lines += ["", "## Models per arm", "", "| arm | model | tokens per task | accuracy | seconds per task |", "|---|---|---|---|---|"]
        for a in arms:
            sel = [c for c in good if c["arm"] == a]
            if sel:
                secs = [c["seconds"] for c in sel if c.get("seconds") is not None]
                lines.append(f"| {a} | {meta['arms'][a].get('model', meta['model'])} | "
                             f"{sum(c['hybrid_cost'] for c in sel) / len(sel):.1f} | "
                             f"{sum(c['correct'] for c in sel) / len(sel):.3f} | "
                             f"{(sum(secs) / len(secs)) if secs else float('nan'):.1f} |")
    lines += skip_table(good, arms, runs)
    lines += curve_table(d, good, arms)
    if len(arms) > 1:
        base = arms[0]
        lines += ["", f"## Paired comparison against `{base}`", "",
                  "Per task the correctness is averaged over repeats; the difference is taken per task and the 95% "
                  "interval is a bootstrap over tasks. The interval excluding 0 is the criterion; 'within noise' means "
                  "this experiment cannot tell the arms apart.", "",
                  "Last column: difference in reward-weighted points per run (sum over the tasks of the group), same "
                  "bootstrap; it is the metric that sees the order in which tasks are solved.", "",
                  "| arm | type | tasks | mean diff | 95% CI | better / worse / same | verdict | points diff per run [95% CI] |",
                  "|---|---|---|---|---|---|---|---|"]

        def per_task(a, t):
            acc = collections.defaultdict(list)
            for c in good:
                if c["arm"] == a and in_group(c, t):
                    acc[c["task_id"]].append(c["correct"])
            return {k: sum(v) / len(v) for k, v in acc.items()}

        def per_task_points(a, t):
            acc = collections.defaultdict(list)
            for c in good:
                if c["arm"] == a and in_group(c, t):
                    acc[c["task_id"]].append(c["reward"] * c["correct"])
            return {k: sum(v) / len(v) for k, v in acc.items()}
        for a in arms[1:]:
            for t in group_labels(good):
                x, y = per_task(a, t), per_task(base, t)
                ids = sorted(set(x) & set(y))
                if len(ids) < 5:
                    continue
                diffs = [x[i] - y[i] for i in ids]
                lo, hi = bootstrap_ci(diffs)
                mean = sum(diffs) / len(diffs)
                better = sum(v > 1e-9 for v in diffs)
                worse = sum(v < -1e-9 for v in diffs)
                verdict = ("**better**" if lo > 0 else "**worse**" if hi < 0 else "within noise")
                px, py = per_task_points(a, t), per_task_points(base, t)
                pd = [px[i] - py[i] for i in ids]
                plo, phi = bootstrap_ci(pd)
                n = len(ids)
                lines.append(f"| {a} | {t} | {n} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                             f"{better} / {worse} / {n - better - worse} | {verdict} | "
                             f"{sum(pd):+.1f} [{plo * n:+.1f}, {phi * n:+.1f}] |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ commands
def cmd_run(path: Path, dry: bool, allow_holdout: bool = False) -> None:
    status, why = run_experiment(path, dry, allow_holdout)
    if status == "aborted":
        sys.exit(2)


def run_experiment(path: Path, dry: bool, allow_holdout: bool = False) -> tuple[str, str]:
    """Run (or resume) one experiment file. Returns (status, reason): status is ok / dry / aborted."""
    e = load_experiment(path, allow_holdout)
    name = e["name"]
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    base = WORK / name / "set"
    grading = build_task_set(e, base)
    frozen = e.get("set_frozen")
    if frozen and frozen.get("set_hash") != set_hash(grading):
        raise SystemExit(f"set '{e['set']}' no longer builds to its frozen hash {frozen.get('set_hash')} "
                         f"(got {set_hash(grading)}): the dataset or the sampler changed; do not use this set")
    manifest = dict(name=name, id=e["id"], author=e["author"], change=e["change"], mode="real", limits="off",
                    hypothesis=e["hypothesis"], model=e["model"], backend=e["backend"],
                    fallback=e["fallback"], tasks=e["tasks"], n_tasks=len(grading), repeats=e["repeats"], seed=e["seed"],
                    arms=e["arms"], task_ids=list(grading), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                    code_version=code_version(), budget=e["budget"], env=e.get("env"),
                    set_name=e.get("set"), set_role=e.get("set_role", "adhoc"), set_hash=set_hash(grading),
                    n_unanswerable=sum(bool(g["unanswerable"]) for g in grading.values()))
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    cells_path, runs_path = d / "cells.jsonl", d / "runs.jsonl"
    done = {(r["arm"], r["repeat"]) for r in map(json.loads, open(runs_path))} if runs_path.exists() else set()
    plan = [(rep, a) for rep in range(1, e["repeats"] + 1) for a in e["arms"] if (a, rep) not in done]
    say(f"experiment {name}: {len(grading)} tasks, {len(e['arms'])} arms x {e['repeats']} repeats, "
        f"{len(plan)} runs to do ({len(done)} already done)")
    if dry:
        for rep, a in plan:
            say(f"  would run arm {a} repeat {rep}: env {e['arms'][a].get('env')}")
        return "dry", ""
    try:
        for rep, a in plan:
            base_url = check_infrastructure(e)
            t0 = time.time()
            rows = run_with_health(e, a, e["arms"][a], grading, base, base_url, rep)
            with open(cells_path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(runs_path, "a") as f:
                f.write(json.dumps({"arm": a, "repeat": rep, "seconds": round(time.time() - t0)}) + "\n")
            ok = sum(r["correct"] for r in rows if not r["infra"])
            nv = sum(not r["infra"] for r in rows)
            say(f"  {a} #{rep}: {ok}/{nv} correct, {sum(r['hybrid_cost'] for r in rows):.0f} tokens, "
                f"{time.time() - t0:.0f}s")
    except Abort as ex:
        say(f"ABORTED: {ex}")
        (d / "ABORTED.txt").write_text(str(ex) + "\n", encoding="utf-8")
        if cells_path.exists():
            (d / "report.md").write_text(make_report(name) + f"\n**Experiment aborted:** {ex}\n", encoding="utf-8")
        return "aborted", str(ex)
    (d / "ABORTED.txt").unlink(missing_ok=True)
    rep_text = make_report(name)
    (d / "report.md").write_text(rep_text, encoding="utf-8")
    shutil.rmtree(WORK / name, ignore_errors=True)
    print(rep_text)
    return "ok", ""


# ------------------------------------------------------------------ machine-readable result and verdict
def summarize(name: str) -> dict:
    """Numbers that go into the experiment file: points per run by arm and the paired points difference with its CI.
    `verdict` is the lab's mechanical reading (accept / reject / clarify per arm against the first arm); the
    `decision` of spec 4.6 stays with a human."""
    d = RESULTS / name
    meta = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cells = [json.loads(l) for l in open(d / "cells.jsonl", encoding="utf-8") if l.strip()]
    good = [c for c in cells if not c["infra"]]
    arms = list(meta["arms"])
    out: dict = {"set": f"{meta.get('set_name') or 'ad hoc'} {meta.get('set_hash')}", "code_version": meta.get("code_version"),
                 "n_tasks": meta["n_tasks"], "repeats": meta["repeats"], "arms": {}, "log": str(d)}
    for a in arms:
        sel = [c for c in good if c["arm"] == a]
        reps = sorted({c["repeat"] for c in sel})
        if not sel:
            continue
        pts = [sum(c["reward"] * c["correct"] for c in sel if c["repeat"] == r) for r in reps]
        toks = [sum(c["hybrid_cost"] for c in sel if c["repeat"] == r) for r in reps]
        out["arms"][a] = {"model": meta["arms"][a].get("model", meta["model"]), "runs": len(reps),
                          "points_per_run": round(sum(pts) / len(pts), 2), "tokens_per_run": round(sum(toks) / len(toks)),
                          "accuracy": round(sum(c["correct"] for c in sel) / len(sel), 4),
                          "infra_broken_cells": sum(c["infra"] for c in cells if c["arm"] == a)}
    verdict: dict[str, str] = {}
    if len(arms) > 1 and arms[0] in out["arms"]:
        def per_task(a):
            acc = collections.defaultdict(list)
            for c in good:
                if c["arm"] == a:
                    acc[c["task_id"]].append(c["reward"] * c["correct"])
            return {k: sum(v) / len(v) for k, v in acc.items()}
        base = per_task(arms[0])
        for a in arms[1:]:
            x = per_task(a)
            ids = sorted(set(x) & set(base))
            if len(ids) < 5:
                verdict[a] = "clarify"
                continue
            diffs = [x[i] - base[i] for i in ids]
            lo, hi = bootstrap_ci(diffs)
            n = len(ids)
            out["arms"][a]["points_diff_vs_first"] = {"mean": round(sum(diffs), 2), "ci95": [round(lo * n, 2), round(hi * n, 2)]}
            verdict[a] = "accept" if lo > 0 else "reject" if hi < 0 else "clarify"
    else:
        verdict = {arms[0]: "baseline"}
    out["verdict"] = verdict
    return out


LAB_MARK = "# ---- written by lab.py: status and result of this experiment (the part above is the experiment) ----"


def write_lab_block(path: Path, block: dict) -> None:
    """Append (or replace) the result block at the end of the experiment file; the experiment text itself is never touched."""
    import yaml
    text = path.read_text(encoding="utf-8")
    if LAB_MARK in text:
        text = text[:text.index(LAB_MARK)]
    text = text.rstrip("\n") + "\n\n" + LAB_MARK + "\n" + yaml.safe_dump(block, sort_keys=False, allow_unicode=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ queue: pending / running / done
def qdir(kind: str) -> Path:
    d = QUEUE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def queue_submit(files: list[str]) -> None:
    import yaml
    for f in files:
        src = Path(f)
        e = yaml.safe_load(src.read_text(encoding="utf-8"))
        name = e.get("name")
        load_experiment(src, allow_holdout=True)   # validates; the holdout guard is applied when it runs
        for kind in ("pending", "running", "done"):
            if (qdir(kind) / f"{name}.yaml").exists():
                raise SystemExit(f"{name}: already in the queue ({kind}); rename the experiment")
        shutil.copyfile(src, qdir("pending") / f"{name}.yaml")
        print(f"queued {name}")


def queue_status() -> None:
    import yaml
    for kind in ("pending", "running", "done"):
        for f in sorted(qdir(kind).glob("*.yaml")):
            top = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            res = top.get("result") or {}
            print(f"{kind:8} {f.stem:36} {top.get('status', '-'):9} verdict {res.get('verdict', '-')}")
    pid_file = QUEUE / "runner.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text() or 0)
        print(f"runner: pid {pid} {'alive' if _alive(pid) else 'DEAD (stale lock; `queue run` will resume)'}")
    else:
        print("runner: not running")


def queue_run(allow_holdout: bool, keep_going: bool) -> None:
    lock = QUEUE / "runner.pid"
    QUEUE.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            pid = int(lock.read_text() or 0)
        except ValueError:
            pid = 0
        if pid and _alive(pid) and pid != os.getpid():
            raise SystemExit(f"queue runner already running (pid {pid})")
    lock.write_text(str(os.getpid()))
    try:
        while True:
            if (QUEUE / "STOP").exists():
                say("STOP file present: leaving the queue")
                break
            todo = sorted(qdir("running").glob("*.yaml")) or sorted(qdir("pending").glob("*.yaml"))
            if not todo:
                say("queue is empty")
                break
            f = todo[0]
            if f.parent.name == "pending":
                f = Path(shutil.move(str(f), str(qdir("running") / f.name)))
                say(f"queue: started {f.stem}")
            else:
                say(f"queue: resuming {f.stem} (interrupted earlier)")
            started = time.strftime("%Y-%m-%d %H:%M:%S")
            write_lab_block(f, {"status": "running", "started": started})
            try:
                status, why = run_experiment(f, False, allow_holdout)
            except SystemExit as ex:   # invalid experiment (unknown set, holdout without permission, ...)
                status, why = "invalid", str(ex)
            except Exception as ex:   # noqa: BLE001 - one broken experiment must not kill the queue
                status, why = "failed", f"{type(ex).__name__}: {ex}"
            block: dict = {"status": status, "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            if why:
                block["reason"] = why
            try:
                name = __import__("yaml").safe_load(f.read_text(encoding="utf-8"))["name"]
                if (RESULTS / name / "cells.jsonl").exists():
                    block["result"] = summarize(name)
            except Exception as ex:   # noqa: BLE001
                block["result_error"] = f"{type(ex).__name__}: {ex}"
            write_lab_block(f, block)
            os.replace(f, qdir("done") / f.name)
            say(f"queue: {f.stem} -> done ({status})")
            if status != "ok" and not keep_going:
                say("queue stops after a failed experiment (use --keep-going to go on); `queue retry <name>` re-queues it")
                break
    finally:
        lock.unlink(missing_ok=True)


def queue_retry(name: str) -> None:
    src = qdir("done") / f"{name}.yaml"
    if not src.exists():
        raise SystemExit(f"{name} is not in done/")
    text = src.read_text(encoding="utf-8")
    if LAB_MARK in text:
        text = text[:text.index(LAB_MARK)].rstrip("\n") + "\n"
    (qdir("pending") / src.name).write_text(text, encoding="utf-8")
    src.unlink()
    print(f"{name} is pending again (finished repeats of its results directory are kept and skipped)")


def cmd_queue(a) -> None:
    if a.action == "submit":
        queue_submit(a.names)
    elif a.action == "status":
        queue_status()
    elif a.action == "retry":
        queue_retry(a.names[0])
    elif a.detach:   # detached runner: own session, SIGHUP ignored, output in queue/runner.log (the ssh link may drop)
        QUEUE.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "queue", "run"] + (["--allow-holdout"] if a.allow_holdout else []) \
            + (["--keep-going"] if a.keep_going else [])
        with open(QUEUE / "runner.log", "a") as log:
            pr = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  start_new_session=True, env=os.environ)
        print(f"queue runner started, pid {pr.pid}, log {QUEUE / 'runner.log'}")
    else:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        queue_run(a.allow_holdout, a.keep_going)


# ------------------------------------------------------------------ frozen sets
def build_named_set(name: str) -> tuple[dict, dict]:
    s = load_set(name)
    e = {k: s[k] for k in SET_KEYS if k in s}
    grading = build_task_set(e, WORK / "sets_build" / name)
    shutil.rmtree(WORK / "sets_build" / name, ignore_errors=True)
    return s, grading


def cmd_sets(action: str, names: list[str]) -> None:
    """list / freeze / verify the named sets. `freeze` appends the `frozen:` block once; a frozen set never changes."""
    allnames = sorted(f.stem for f in SETS_DIR.glob("*.yaml"))
    names = names or allnames
    if action == "list":
        for n in names:
            s = load_set(n)
            fz = s.get("frozen") or {}
            print(f"{n:22} {s['role']:8} tasks {s['tasks']}  seed {s['seed']}  frozen {fz.get('set_hash', '-')}")
        return
    ids: dict[str, set] = {}
    bad = 0
    for n in names:
        s, grading = build_named_set(n)
        info = set_summary(grading)
        ids[n] = set(grading)
        fz = s.get("frozen")
        if action == "freeze" and not fz:
            f = SETS_DIR / f"{n}.yaml"
            text = f.read_text(encoding="utf-8")
            text += ("" if text.endswith("\n") else "\n") + "frozen:\n" + "".join(
                f"  {k}: {json.dumps(v)}\n" for k, v in info.items())
            f.write_text(text, encoding="utf-8")
            print(f"{n}: frozen {info['set_hash']} ({info['n_tasks']} tasks, {info['n_unanswerable']} unanswerable)")
        elif not fz:
            print(f"{n}: NOT FROZEN (builds to {info['set_hash']})")
            bad += 1
        elif fz["set_hash"] == info["set_hash"]:
            print(f"{n}: ok {info['set_hash']} ({info['n_tasks']} tasks, {info['n_unanswerable']} unanswerable)")
        else:
            print(f"{n}: MISMATCH frozen {fz['set_hash']} but builds to {info['set_hash']}")
            bad += 1
    holds = [n for n in ids if load_set(n)["role"] == "holdout"]
    for h in holds:   # the control set must not share a task with any development set
        for n in ids:
            if n != h and n not in holds and ids[n] & ids[h]:
                print(f"OVERLAP: holdout {h} shares {len(ids[n] & ids[h])} tasks with {n}")
                bad += 1
    if bad:
        sys.exit(1)


def selftest() -> None:
    rng = random.Random(1)
    same = [rng.choice([-1, 0, 0, 1]) * 0.33 for _ in range(40)]
    lo, hi = bootstrap_ci(same)
    assert lo < 0 < hi, ("noise must be 'within noise'", lo, hi)
    shifted = [0.33 if rng.random() < 0.5 else 0.0 for _ in range(40)]
    lo, hi = bootstrap_ci(shifted)
    assert lo > 0, ("a one-sided shift must be detected", lo, hi)
    rows = [{"status": "not_run"}, {"status": "solved"}]
    assert mark_unfunded(rows, [{"event": "run_end"}]) == 1 and rows[0]["status"] == "unfunded", "budget-skipped task is final"
    rows = [{"status": "not_run"}]
    assert mark_unfunded(rows, [{"event": "run_start"}]) == 0 and rows[0]["status"] == "not_run", "cut run: still transient"
    e = {"seed": 1000, "tasks": {"math": 6, "search": 4}}
    a = build_task_set(e, WORK / "selftest_a")
    b = build_task_set(e, WORK / "selftest_b")
    assert list(a) == list(b) and len(a) == 10, "task set must be reproducible"
    shutil.rmtree(WORK / "selftest_a")
    shutil.rmtree(WORK / "selftest_b")
    print("selftest ok:", "bootstrap detects shift and ignores noise;", "task set reproducible:", list(a)[:3], "...")


def _stub_infra(e: dict) -> str:
    return "stub"


def _stub_run(e, arm_name, arm, grading, base, base_url, rep):
    """LAB_STUB=1: synthetic outcomes, no harness and no model call (tests the queue, resume and report code)."""
    time.sleep(float(os.environ.get("LAB_STUB_SLEEP", "0")))
    boost = float((arm.get("env") or {}).get("STUB_BOOST", 0))
    rows = []
    for i, (tid, g) in enumerate(grading.items()):
        rg = random.Random(f"{arm_name}:{rep}:{tid}")
        p = min(0.95, 0.6 + boost)
        ok = rg.random() < p
        answered = not g["unanswerable"]
        rows.append({"chunk": 0, "attempt": 1, "model": arm.get("model", e["model"]), "task_id": tid, "type": g["task_type"],
                     "source": g["source"], "check": g["check"], "reward": g["reward"], "unanswerable": g["unanswerable"],
                     "order": i, "status": "solved", "errors": [], "answered": answered,
                     "correct": (True if g["unanswerable"] else ok), "fallback": False,
                     "hybrid_correct": ok, "hybrid_cost": round(100 + 50 * rg.random(), 1), "pure_correct": ok,
                     "pure_answered": answered, "pure_cost": 100.0, "pipeline_calls": 1, "agent_calls": 0,
                     "in_tokens": 100, "out_tokens": 50, "seconds": 1.0, "arm": arm_name, "repeat": rep, "infra": False})
    return rows


def main() -> None:
    if os.environ.get("LAB_STUB"):
        globals()["run_with_health"] = _stub_run
        globals()["check_infrastructure"] = _stub_infra
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("experiment")
    r.add_argument("--dry", action="store_true")
    r.add_argument("--allow-holdout", action="store_true", help="permit a holdout set (final check only)")
    st = sub.add_parser("sets")
    st.add_argument("action", choices=["list", "freeze", "verify"])
    st.add_argument("names", nargs="*")
    q = sub.add_parser("queue")
    q.add_argument("action", choices=["submit", "run", "status", "retry"])
    q.add_argument("names", nargs="*")
    q.add_argument("--detach", action="store_true")
    q.add_argument("--keep-going", action="store_true")
    q.add_argument("--allow-holdout", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("name")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(Path(a.experiment), a.dry, a.allow_holdout)
    elif a.cmd == "queue":
        cmd_queue(a)
    elif a.cmd == "sets":
        cmd_sets(a.action, a.names)
    elif a.cmd == "report":
        print(make_report(a.name))
    else:
        selftest()


if __name__ == "__main__":
    main()
