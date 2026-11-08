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
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stream_run as sr  # noqa: E402  (sampler, grader, transient-error test, proxy helpers)

LAB = sr.LAB
RESULTS = LAB / "experiments" / "results"
WORK = Path("/tmp/hl_lab")
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
def load_experiment(path: Path) -> dict:
    import yaml
    e = yaml.safe_load(path.read_text(encoding="utf-8"))
    for k in ("name", "tasks", "arms"):
        if k not in e:
            raise SystemExit(f"{path}: missing key '{k}'")
    e.setdefault("hypothesis", "")
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
    if len(e["arms"]) < 1:
        raise SystemExit("an experiment needs at least one arm")
    return e


def build_task_set(e: dict, out: Path) -> dict:
    """Fixed task set for the whole experiment: `tasks: {math: 30, search: 10}` drawn without repetition."""
    shutil.rmtree(out, ignore_errors=True)
    (out / "tasks").mkdir(parents=True)
    sm = sr.Sampler(int(e["seed"]))
    grading: dict[str, dict] = {}
    for t, n in e["tasks"].items():
        got, n_un, guard = 0, 0, 0
        pool = e.get("sources", {}).get(t) or sr.SOURCES[t]
        quota = (e.get("unanswerable") or {}).get(t)   # exact number of unanswerable tasks of this type, if given
        while got < n and guard < n * 300:
            guard += 1
            r = sm.next_row(sm.rng.choice(pool))
            if r is None or r["task_id"] in grading:
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
    if e["model"] != "auto":   # `model: auto` leaves the choice to the harness, as in the competition
        env["HARNESS_FORCE_MODEL"] = e["model"]
    else:
        env.pop("HARNESS_FORCE_MODEL", None)
    for k, v in {**(e.get("env") or {}), **(arm.get("env") or {})}.items():  # arm overrides experiment-wide env
        env[str(k)] = str(v)
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


def execute(e: dict, arm_name: str, arm: dict, grading: dict, base: Path, base_url: str, label: str) -> list[dict]:
    """Run the harness on all tasks of `grading` (copied from the fixed set) and return graded rows."""
    work = WORK / e["name"] / label
    shutil.rmtree(work, ignore_errors=True)
    (work / "tasks").mkdir(parents=True)
    for tid in grading:
        shutil.copytree(base / "tasks" / tid, work / "tasks" / tid)
    run_harness(e, arm, work / "tasks", work, base_url)
    rows = sr.grade_chunk(e["model"], 0, work / "tasks", work / "logs", grading)
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
        for r in sr.journal_rows(work / "logs"):
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


def make_report(name: str) -> str:
    d = RESULTS / name
    meta = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cells = [json.loads(l) for l in open(d / "cells.jsonl", encoding="utf-8") if l.strip()]
    runs = [json.loads(l) for l in open(d / "runs.jsonl", encoding="utf-8") if l.strip()] if (d / "runs.jsonl").exists() else []
    arms = list(meta["arms"])
    lines = [f"# Experiment `{name}`", "", f"Hypothesis: {meta.get('hypothesis') or '-'}", "",
             f"Model `{meta['model']}`, backend `{meta['backend']}`, fallback `{meta['fallback']}`, "
             f"{meta['n_tasks']} tasks {meta['tasks']}, repeats {meta['repeats']}, seed {meta['seed']}.",
             f"Code version: {meta.get('code_version', 'not recorded')}. Task set: {meta.get('set_hash', 'not recorded')}, "
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
def cmd_run(path: Path, dry: bool) -> None:
    e = load_experiment(path)
    name = e["name"]
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    base = WORK / name / "set"
    grading = build_task_set(e, base)
    manifest = dict(name=name, hypothesis=e["hypothesis"], model=e["model"], backend=e["backend"],
                    fallback=e["fallback"], tasks=e["tasks"], n_tasks=len(grading), repeats=e["repeats"], seed=e["seed"],
                    arms=e["arms"], task_ids=list(grading), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                    code_version=code_version(), budget=e["budget"], env=e.get("env"),
                    set_hash=hashlib.sha1("\n".join(sorted(grading)).encode()).hexdigest()[:12],
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
        return
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
        sys.exit(2)
    (d / "ABORTED.txt").unlink(missing_ok=True)
    rep_text = make_report(name)
    (d / "report.md").write_text(rep_text, encoding="utf-8")
    shutil.rmtree(WORK / name, ignore_errors=True)
    print(rep_text)


def selftest() -> None:
    rng = random.Random(1)
    same = [rng.choice([-1, 0, 0, 1]) * 0.33 for _ in range(40)]
    lo, hi = bootstrap_ci(same)
    assert lo < 0 < hi, ("noise must be 'within noise'", lo, hi)
    shifted = [0.33 if rng.random() < 0.5 else 0.0 for _ in range(40)]
    lo, hi = bootstrap_ci(shifted)
    assert lo > 0, ("a one-sided shift must be detected", lo, hi)
    e = {"seed": 1000, "tasks": {"math": 6, "search": 4}}
    a = build_task_set(e, WORK / "selftest_a")
    b = build_task_set(e, WORK / "selftest_b")
    assert list(a) == list(b) and len(a) == 10, "task set must be reproducible"
    shutil.rmtree(WORK / "selftest_a")
    shutil.rmtree(WORK / "selftest_b")
    print("selftest ok:", "bootstrap detects shift and ignores noise;", "task set reproducible:", list(a)[:3], "...")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("experiment")
    r.add_argument("--dry", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("name")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(Path(a.experiment), a.dry)
    elif a.cmd == "report":
        print(make_report(a.name))
    else:
        selftest()


if __name__ == "__main__":
    main()
