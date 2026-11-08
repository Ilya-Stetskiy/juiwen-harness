#!/usr/bin/env python3
"""Offline simulator of the harness scheduler (lab stand, spec 4.1): no model call, no gateway.

It runs the REAL `Scheduler` of harness/sorter.py (and the real `Predictor`, or one named with --predictor) on
recorded outcomes: for a (task, model) pair the recorded cost in tokens, correct / wrong, whether the pipeline gave
up (fallback) and the time. Several workers are simulated on a virtual clock from the recorded seconds; the
scheduler is called exactly like run_agent.py calls it: `next(spent)` when a worker is free, `report(task, model,
actual)` when a task is finished.

Data (everything is read-only):
  * stream     ~/datasets/token_juiwen/stream/results.jsonl (+ results_fix.jsonl, later files override): about 2 450
               tasks x 4 models, `hybrid_*` = pipeline + inline fallback agent, `pure_*` = pipeline alone.
  * lab        <results>/<experiment>/cells.jsonl + journals/<arm>_<rep>.jsonl: the model of a cell is read from the
               `task_start` event of the journal (cells of `model: auto` experiments only say "auto").
  * tasks      workspace sizes (the predictor uses them) come from the index built by `index-tasks`.

Modes (what happens when the pipeline of a task fails and the tool agent is needed):
  hybrid  agent runs inline, cost = hybrid_cost            (HARNESS_FALLBACK=on)
  pure    no agent, cost = pure_cost                        (HARNESS_FALLBACK=off)
  late    task is deferred; after the main queue the agent runs while the remaining budget covers its estimate,
          highest reward first; agent cost = hybrid_cost - pure_cost   (HARNESS_FALLBACK=late)

Scoring is the lab's: an unanswerable task is correct when it is left without an answer, so an unanswerable task
that is skipped or never started counts its reward; an answerable task that never ran is wrong.

usage:
  sim.py index-tasks                         build the workspace-size index (needs the dataset; run on the host)
  sim.py replay <experiment> [--arm A] [--rep N]
                                             re-run a recorded run through the scheduler and compare with the real result
  sim.py curve --replay <experiment>:<arm>:<rep>     points by budget on the tasks of a recorded run
  sim.py curve --stream [--draws 10]         points by budget on mixed sets drawn from the stream data (all 4 models known)
  sim.py selftest
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
STREAM_DIR = HOME / "datasets" / "token_juiwen" / "stream"
LAB = HOME / "harness-lab"
RESULTS = Path(os.environ.get("LAB_EXPERIMENTS") or LAB / "experiments") / "results"
INDEX = LAB / "sim" / "tasks_index.json"
MODELS_ALL = ("qwen3.6-35b-a3b-noreason", "qwen3.6-35b-a3b", "qwen3.8-27b-noreason", "qwen3.8-27b")
FALLBACK_ESTIMATE = {"coding": 6000.0, "math": 3000.0, "data_analysis": 4000.0, "search": 3000.0}  # harness/executor.py
TRANSIENT = re.compile(r"proxy halted|RateLimitError|InternalServerError|APITimeoutError|Request timed out|APIConnectionError|"
                       r"Connection error|Слишком много|Превышен лимит|Bad Gateway|Service Unavailable|Gateway Time-out", re.I)
DEFAULT_SECONDS = 10.0


# ------------------------------------------------------------------ harness classes (the real ones)
def import_harness(harness_dir: Path | None):
    """Returns (Scheduler, Predictor, TaskInfo, MODELS, fallback_estimate, version string)."""
    root = harness_dir or HERE.parent
    if not (root / "harness" / "sorter.py").exists():
        raise SystemExit(f"no harness package in {root}: give --harness-dir (the directory that contains `harness/`)")
    sys.path.insert(0, str(root))
    sorter = importlib.import_module("harness.sorter")
    predictor = importlib.import_module("harness.predictor")
    tasks = importlib.import_module("harness.tasks")
    try:
        MODELS = tuple(importlib.import_module("harness.config").MODELS)
    except Exception:   # noqa: BLE001
        MODELS = MODELS_ALL
    est = FALLBACK_ESTIMATE
    try:
        est = dict(importlib.import_module("harness.executor").FALLBACK_ESTIMATE)
    except Exception:   # noqa: BLE001 - the executor needs the agent framework; the table is copied above
        pass
    ver = (root / "VERSION").read_text().strip() if (root / "VERSION").exists() else "unknown"
    return sorter.Scheduler, predictor.Predictor, tasks.TaskInfo, MODELS, est, ver


def load_class(spec: str):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)


# ------------------------------------------------------------------ recorded outcomes
@dataclass
class Outcome:
    cost_h: float      # pipeline + inline agent
    ok_h: bool
    cost_p: float      # pipeline alone
    ok_p: bool
    fb: bool           # the pipeline gave up (an agent was needed)
    secs: float
    src: str = ""


def outcome_from_row(r: dict, src: str) -> Outcome:
    return Outcome(float(r["hybrid_cost"]), bool(r["hybrid_correct"]), float(r["pure_cost"]), bool(r["pure_correct"]),
                   bool(r.get("fallback")), float(r["seconds"]) if r.get("seconds") is not None else DEFAULT_SECONDS, src)


def transient(r: dict) -> bool:
    if r["status"] == "not_run":
        return True
    if r["status"] != "error":
        return False
    errs = r.get("errors")
    return (not errs) or any(TRANSIENT.search(e) for e in errs)


class Store:
    """(task_id, model) -> recorded outcomes; layers are searched in order (own run first, then the pool)."""

    kinds: dict = {}     # task id -> (type, unanswerable): set by the loaders, used to impute by kind

    def __init__(self, layers: list[dict] | None = None, kinds: dict | None = None) -> None:
        self.layers = layers or [{}]
        self.kinds = kinds if kinds is not None else {}

    def add(self, tid: str, model: str, o: Outcome, replace: bool = False) -> None:
        d = self.layers[0].setdefault((tid, model), [])
        if replace:
            d.clear()
        d.append(o)

    def get(self, tid: str, model: str) -> list[Outcome]:
        for layer in self.layers:
            if (tid, model) in layer:
                return layer[(tid, model)]
        return []

    def stats(self, ttype, model: str, types: dict) -> dict | None:
        """Mean outcome of one (type, unanswerable?, model) over all layers; used only to fill a pair never recorded."""
        k = (ttype, model)
        cache = self.__dict__.setdefault("_stats", {})
        if k not in cache:
            xs = [o for layer in self.layers for (tid, m), lst in layer.items() if m == model and types.get(tid) == ttype for o in lst]
            if not xs:   # this model was never seen on this kind of task: the mean over the other models of the kind
                xs = [o for layer in self.layers for (tid, m), lst in layer.items() if types.get(tid) == ttype for o in lst]
            cache[k] = None if not xs else {
                "cost_h": statistics.fmean(o.cost_h for o in xs), "ok_h": statistics.fmean(o.ok_h for o in xs),
                "cost_p": statistics.fmean(o.cost_p for o in xs), "ok_p": statistics.fmean(o.ok_p for o in xs),
                "fb": statistics.fmean(o.fb for o in xs), "secs": statistics.fmean(o.secs for o in xs)}
        return cache[k]


def load_stream(files: list[Path]) -> tuple[Store, dict[str, dict]]:
    """Valid (non-infrastructure) rows of the stream; a later row or file overrides an earlier one."""
    store, attrs = Store(), {}
    store.kinds = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if transient(r) or r["model"] not in MODELS_ALL:
                continue
            store.add(r["task_id"], r["model"], outcome_from_row(r, f.name), replace=True)
            store.kinds[r["task_id"]] = (r["type"], bool(r["unanswerable"]))
            attrs[r["task_id"]] = {"type": r["type"], "reward": float(r["reward"]), "unans": bool(r["unanswerable"]),
                                   "source": r.get("source", "")}
    return store, attrs


def load_index(path: Path) -> dict[str, int]:
    try:
        return json.load(open(path))
    except OSError:
        return {}


def read_run(exp: str, arm: str, rep: int) -> dict:
    """One recorded run of a lab experiment: tasks, per-pair outcomes (model from the journal), real totals."""
    d = RESULTS / exp
    meta = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cells = [c for c in map(json.loads, open(d / "cells.jsonl", encoding="utf-8")) if c["arm"] == arm and c["repeat"] == rep]
    if not cells:
        raise SystemExit(f"{exp}: no cells for arm {arm} repeat {rep}")
    jf = d / "journals" / f"{arm}_{rep}.jsonl"
    chosen: dict[str, str] = {}
    start_ts: dict[str, float] = {}
    ends: dict[str, dict] = {}
    run_start = run_end = {}
    if jf.exists():
        for e in map(json.loads, open(jf, encoding="utf-8")):
            if e["event"] == "task_start":
                chosen.setdefault(e["task"], e["model"])
                start_ts.setdefault(e["task"], e["ts"])
            elif e["event"] == "run_start":
                run_start = e
            elif e["event"] == "run_end":
                run_end = e
            elif e["event"] == "task_end":
                ends[e["task"]] = e
    arm_model = meta["arms"][arm].get("model", meta["model"])
    store = Store()
    tasks = []
    for c in cells:
        store.kinds[c["task_id"]] = (c["type"], bool(c["unanswerable"]))
        model = arm_model if arm_model != "auto" else chosen.get(c["task_id"])
        tasks.append({"task_id": c["task_id"], "type": c["type"], "reward": float(c["reward"]), "unans": bool(c["unanswerable"]),
                      "source": c["source"], "model": model, "infra": c["infra"], "status": c["status"]})
        if model and not c["infra"] and c["status"] not in ("not_run", "unfunded"):
            store.add(c["task_id"], model, outcome_from_row(c, f"{exp}/{arm}_{rep}"))
    env = {**(meta.get("env") or {}), **(meta["arms"][arm].get("env") or {})}
    fb = str(env.get("HARNESS_FALLBACK", meta.get("fallback", "off"))).lower()
    return {"meta": meta, "cells": cells, "tasks": tasks, "store": store, "mode": {"on": "hybrid", "off": "pure", "late": "late"}.get(fb, "hybrid"),
            "budget": float(meta["budget"]), "workers": int(run_start.get("workers", 5)), "arm_model": arm_model,
            "real_points": sum(c["reward"] * c["correct"] for c in cells if not c["infra"]),
            "real_spent": float(run_end.get("spent", sum(c["hybrid_cost"] for c in cells))),
            "real_done": sum(c["status"] not in ("not_run", "unfunded") for c in cells), "journal": jf.exists(), "chosen": chosen, "start_ts": start_ts}


# ------------------------------------------------------------------ the simulation
def simulate(tasks: list[dict], store: Store, *, budget: float, models: tuple, harness, predictor, mode: str = "hybrid",
             workers: int = 5, seed: int = 0, draw: str = "first", missing: str = "impute", late_order: str = "reward",
             ws: dict[str, int] | None = None, jitter: float = 0.0) -> dict:
    """tasks: dicts with task_id, type, reward, unans. Returns totals and the per-task log."""
    Scheduler, TaskInfo, est_table = harness["Scheduler"], harness["TaskInfo"], harness["est"]
    rng = random.Random(seed)
    ws = ws or {}
    types = {t["task_id"]: (t["type"], bool(t["unans"])) for t in tasks}
    tasks = sorted(tasks, key=lambda t: t["task_id"])   # harness.tasks.load_tasks reads the directory sorted by name; ties in the scheduler follow it
    infos = {t["task_id"]: TaskInfo(task_id=t["task_id"], task_type=t["type"], instruction="", reward=float(t["reward"]),
                                    check="", dir=Path("."), workspace=None, n_files=0,
                                    workspace_bytes=int(ws.get(t["task_id"], 0))) for t in tasks}
    unans = {t["task_id"]: bool(t["unans"]) for t in tasks}
    sched = Scheduler(list(infos.values()), list(models), predictor, float(budget))
    n_missing = n_imputed = 0

    def pick(tid: str, model: str) -> Outcome:
        nonlocal n_missing, n_imputed
        lst = store.get(tid, model)
        if lst:
            o = lst[0] if draw == "first" else rng.choice(lst)
            if jitter:   # the order in which tasks finish decides what the scheduler learns first: perturb the clock
                o = Outcome(o.cost_h, o.ok_h, o.cost_p, o.ok_p, o.fb, o.secs * (1 + jitter * (2 * rng.random() - 1)), o.src)
            return o
        n_missing += 1
        st = store.stats(types[tid], model, store.kinds)
        if missing == "error" or st is None:
            raise SystemExit(f"no recorded outcome for task {tid} with model {model}"
                             + ("" if missing == "error" else " (and none for its type to impute from)"))
        n_imputed += 1
        return Outcome(st["cost_h"], rng.random() < st["ok_h"], st["cost_p"], rng.random() < st["ok_p"],
                       rng.random() < st["fb"], st["secs"], "imputed")

    starts: list[tuple[str, str]] = []
    done: dict[str, dict] = {}       # task id -> {model, cost, ok, t, cum}
    deferred: list = []
    spent, now, seq = 0.0, 0.0, 0
    heap: list = []                  # (finish time, seq, worker, kind, payload)
    idle = list(range(workers))
    import heapq

    def finish_main(item):
        nonlocal spent
        tinfo, model, o, cost, status = item
        spent += cost
        sched.report(tinfo, model, cost)
        if status == "deferred":
            deferred.append((tinfo, model, o))
        else:
            done[tinfo.task_id] = {"model": model, "cost": cost, "ok": o.ok_h if mode == "hybrid" else o.ok_p, "t": now, "cum": spent}

    while True:
        for w in list(idle):
            nxt = sched.next(spent)
            if nxt is None:
                idle.remove(w)     # like run_agent.py: a worker that gets nothing leaves for good
                continue
            tinfo, model, _p = nxt
            starts.append((tinfo.task_id, model))
            o = pick(tinfo.task_id, model)
            if mode == "hybrid":
                cost, status = o.cost_h, "end"
            elif mode == "pure":
                cost, status = o.cost_p, "end"
            else:
                cost, status = o.cost_p, ("deferred" if o.fb else "end")
            idle.remove(w)
            seq += 1
            heapq.heappush(heap, (now + o.secs, seq, w, (tinfo, model, o, cost, status)))
        if not heap:
            break
        now, _, w, item = heapq.heappop(heap)
        finish_main(item)
        idle.append(w)

    # late phase: the agent for deferred tasks, only while the remaining budget covers its estimate
    n_unfunded = 0
    if mode == "late" and deferred:
        queue = list(deferred)
        if late_order != "fifo":
            queue.sort(key=lambda x: -x[0].reward)
        running: dict[str, float] = {}      # task id -> estimate (cost accrues at the end in the simulation)
        idle = list(range(workers))
        heap = []
        while queue or heap:
            progressed = False
            while queue and idle:
                tinfo, model, o = queue[0]
                est = est_table.get(tinfo.task_type, 4000.0)
                left = budget - spent
                promised = sum(running.values())
                if left < est or (left - promised < est and not running):
                    queue.pop(0)
                    n_unfunded += 1
                    progressed = True
                    continue
                if left - promised < est:
                    break                   # money is promised to agents still running: wait for them
                queue.pop(0)
                w = idle.pop()
                running[tinfo.task_id] = est
                seq += 1
                heapq.heappush(heap, (now + max(o.secs, 1.0), seq, w, (tinfo, model, o)))
                progressed = True
            if not heap:
                if queue and not progressed:
                    break
                continue
            now, _, w, (tinfo, model, o) = heapq.heappop(heap)
            running.pop(tinfo.task_id, None)
            cost = max(o.cost_h - o.cost_p, 0.0)
            spent += cost
            done[tinfo.task_id] = {"model": model, "cost": o.cost_p + cost, "ok": o.ok_h, "t": now, "cum": spent}
            idle.append(w)

    points = at_b = 0.0
    n_ok = n_run = 0
    for tid, ti in infos.items():
        d = done.get(tid)
        if d is None:                       # skipped, never started, deferred and unfunded
            ok = unans[tid]
            if ok:
                points += ti.reward
                at_b += ti.reward
            continue
        n_run += 1
        if d["ok"]:
            points += ti.reward
            n_ok += 1
            if d["cum"] <= budget:
                at_b += ti.reward
    return {"points": points, "points_at_budget": at_b, "spent": spent, "n_run": n_run, "n_correct_run": n_ok,
            "n_unrun": len(infos) - n_run, "n_skipped": len(sched.skipped), "n_unfunded": n_unfunded, "n_missing": n_missing,
            "n_imputed": n_imputed, "makespan": now, "done": done, "starts": starts}


# ------------------------------------------------------------------ commands
def make_context(a) -> dict:
    Scheduler, Predictor, TaskInfo, MODELS, est, ver = import_harness(Path(a.harness_dir) if a.harness_dir else None)
    pred_cls = load_class(a.predictor) if a.predictor else Predictor
    if a.scheduler:
        Scheduler = load_class(a.scheduler)
    return {"h": {"Scheduler": Scheduler, "TaskInfo": TaskInfo, "est": est}, "predictor": pred_cls(), "models": MODELS, "version": ver}


def stream_files(a) -> list[Path]:
    return [STREAM_DIR / f for f in a.stream_files.split(",")]


def cmd_index(a) -> None:
    """Workspace bytes per task (what the predictor sees as `workspace_bytes`) for stream tasks and all lab cells."""
    sys.path.insert(0, str(LAB))
    import stream_run as sr
    wanted: dict[str, str] = {}
    for f in stream_files(a):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            wanted[r["task_id"]] = r["source"]
    if RESULTS.is_dir():
        for cp in RESULTS.glob("*/cells.jsonl"):
            for line in open(cp, encoding="utf-8"):
                r = json.loads(line)
                wanted[r["task_id"]] = r["source"]
    index = load_index(Path(a.index))
    todo = collections.defaultdict(set)
    for tid, src in wanted.items():
        if tid not in index:
            todo[src].add(tid)
    for src, ids in todo.items():
        with open(sr.SRC_PATH[src], encoding="utf-8") as f:
            for line in f:
                m = re.search(r'"task_id":\s*"([^"]+)"', line[:400])   # cheap test before parsing the whole row
                if not m or m.group(1) not in ids:
                    continue
                r = json.loads(line)
                if r["task_id"] in ids:
                    index[r["task_id"]] = sum(len(c.encode("utf-8")) for c in (r.get("files") or {}).values())
        print(f"{src}: {len(ids)} tasks")
    Path(a.index).parent.mkdir(parents=True, exist_ok=True)
    json.dump(index, open(a.index, "w"))
    print(f"index: {len(index)} tasks -> {a.index}")


def run_replay(ctx, run: dict, a, budget: float | None = None, seed: int = 0, jitter: float = 0.0) -> dict:
    pool, _ = load_stream([f for f in stream_files(a) if f.exists()]) if a.pool else (Store(), {})
    store = Store([run["store"].layers[0]] + pool.layers, {**pool.kinds, **run["store"].kinds})
    models = (run["arm_model"],) if run["arm_model"] != "auto" else ctx["models"]
    return simulate(run["tasks"], store, budget=budget or run["budget"], models=models, harness=ctx["h"], predictor=ctx["predictor"],
                    mode=run["mode"], workers=run["workers"], seed=seed, missing=a.missing, ws=load_index(Path(a.index)),
                    jitter=jitter)


def cmd_replay(a) -> None:
    ctx = make_context(a)
    meta = json.loads((RESULTS / a.experiment / "manifest.json").read_text(encoding="utf-8"))
    arms = [a.arm] if a.arm else list(meta["arms"])
    reps = [a.rep] if a.rep else list(range(1, meta["repeats"] + 1))
    ws = load_index(Path(a.index))
    print(f"harness used by the simulator: {ctx['version']}; code of the real run: {meta.get('code_version')}; "
          f"workspace index: {len(ws)} tasks\n")
    print("| arm | rep | mode | real points | sim points | diff | real spent | sim spent | diff % | same model | unrun real/sim | imputed |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    worst = 0.0
    for arm in arms:
        for rep in reps:
            try:
                run = read_run(a.experiment, arm, rep)
            except (OSError, SystemExit):
                continue
            res = run_replay(ctx, run, a, budget=a.budget)
            if a.jitter_draws:
                js = [run_replay(ctx, run, a, budget=a.budget, seed=100 + k, jitter=0.25) for k in range(a.jitter_draws)]
                pts = [j["points"] for j in js]
                m, sd = statistics.fmean(pts), statistics.pstdev(pts)
                spt = statistics.fmean(j["spent"] for j in js)
                extra = f" (budget {a.budget:.0f})" if a.budget else ""
                print(f"  {arm} #{rep}{extra}: with the clock perturbed by +-25% ({a.jitter_draws} draws) sim points {m:.1f} sd {sd:.1f} "
                      f"[{min(pts):.0f}..{max(pts):.0f}], real {run['real_points']:.0f} ({(run['real_points'] - m) / max(sd, 1e-9):+.1f} sd); "
                      f"spent {spt:.0f} vs real {run['real_spent']:.0f}")
            same = sum(1 for tid, d in res["done"].items() if run["chosen"].get(tid) == d["model"])
            if a.diagnose:
                real_order = sorted(run["chosen"].items(), key=lambda kv: run["start_ts"].get(kv[0], 0))
                for i, ((rt, rm), (st, sm)) in enumerate(zip(real_order, res["starts"])):
                    if (rt, rm) != (st, sm):
                        print(f"  first divergence at start #{i}: real {rt} {rm}; sim {st} {sm}")
                        break
            dp = res["points"] - run["real_points"]
            ds = 100 * (res["spent"] - run["real_spent"]) / max(run["real_spent"], 1)
            worst = max(worst, abs(dp))
            print(f"| {arm} | {rep} | {run['mode']} | {run['real_points']:.0f} | {res['points']:.0f} | {dp:+.0f} | {run['real_spent']:.0f} | "
                  f"{res['spent']:.0f} | {ds:+.1f} | {same}/{len(res['done'])} | "
                  f"{sum(t['status'] in ('not_run', 'unfunded') for t in run['tasks'])}/{res['n_unrun']} | {res['n_imputed']} |")
    print(f"\nlargest points difference: {worst:.0f}")


def budgets_from(spec: str) -> list[int]:
    lo, hi, st = (int(x) for x in spec.split(":"))
    return list(range(lo, hi + 1, st))


def draw_stream_set(store: Store, attrs: dict, seed: int, per_type: int, unans_quota: dict[str, int]) -> list[dict]:
    """Mixed set from the stream pool: only tasks with all four models recorded; random rewards 1-5 as in the lab sets."""
    rng = random.Random(seed)
    by = collections.defaultdict(lambda: ([], []))
    for tid, at in sorted(attrs.items()):
        if all(store.get(tid, m) for m in MODELS_ALL):
            by[at["type"]][1 if at["unans"] else 0].append(tid)
    out = []
    for t in ("math", "search", "data_analysis", "coding"):
        ans, un = by[t]
        rng.shuffle(ans)
        rng.shuffle(un)
        q = unans_quota.get(t, 0)
        ids = un[:q] + ans[:per_type - len(un[:q])]
        for tid in ids:
            out.append({"task_id": tid, "type": t, "unans": attrs[tid]["unans"],
                        "reward": random.Random(f"{seed}:{tid}").randint(1, 5)})
    return out


def cmd_curve(a) -> None:
    ctx = make_context(a)
    budgets = budgets_from(a.budgets)
    ws = load_index(Path(a.index))
    sets: list[tuple[list[dict], Store, int, str, tuple]] = []
    if a.replay:
        exp, arm, rep = (a.replay.split(":") + [None, None])[:3]
        meta = json.loads((RESULTS / exp / "manifest.json").read_text(encoding="utf-8"))
        arm = arm or next(iter(meta["arms"]))
        run = read_run(exp, arm, int(rep or 1))
        pool, _ = load_stream([f for f in stream_files(a) if f.exists()]) if a.pool else (Store(), {})
        sets.append((run["tasks"], Store([run["store"].layers[0]] + pool.layers, {**pool.kinds, **run["store"].kinds}), run["workers"], run["mode"], ctx["models"]))
        title = f"tasks of {exp}/{arm}_{rep or 1} ({len(run['tasks'])}), outcomes of that run completed from the stream pool"
    else:
        store, attrs = load_stream([f for f in stream_files(a) if f.exists()])
        quota = {k: int(v) for k, v in (x.split(":") for x in a.unanswerable.split(",") if x)}
        for d in range(a.draws):
            sets.append((draw_stream_set(store, attrs, a.seed + d, a.per_type, quota), store, 5, a.mode, ctx["models"]))
        title = f"{a.draws} mixed sets drawn from the stream data ({a.per_type} per type, unanswerable {quota}), mode {a.mode}"
    print(f"# Points by budget: {title}\n")
    print(f"Simulator harness: {ctx['version']}; predictor {type(ctx['predictor']).__module__}.{type(ctx['predictor']).__name__}\n")
    print("| budget | points (mean) | sd | points within B | spent (mean) | tasks run | tasks never run | imputed |")
    print("|---|---|---|---|---|---|---|---|")
    rows = []
    for B in budgets:
        rs = [simulate(ts, st, budget=B, models=md, harness=ctx["h"], predictor=ctx["predictor"], mode=a.mode if not a.replay else m,
                       workers=w, seed=a.seed + i, missing=a.missing, ws=ws) for i, (ts, st, w, m, md) in enumerate(sets)]
        pts = [r["points"] for r in rs]
        row = (B, statistics.fmean(pts), statistics.pstdev(pts) if len(pts) > 1 else float("nan"),
               statistics.fmean(r["points_at_budget"] for r in rs), statistics.fmean(r["spent"] for r in rs),
               statistics.fmean(r["n_run"] for r in rs), statistics.fmean(r["n_unrun"] for r in rs), sum(r["n_imputed"] for r in rs))
        rows.append(row)
        print(f"| {B // 1000}k | {row[1]:.1f} | {row[2]:.1f} | {row[3]:.1f} | {row[4]:.0f} | {row[5]:.1f} | {row[6]:.1f} | {row[7]} |")
    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write("budget,points,sd,points_within_budget,spent,tasks_run,tasks_never_run,imputed\n")
            for r in rows:
                f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + "\n")


def selftest(a) -> None:
    """The real Scheduler on a tiny universe where the right answer is known without the simulator."""
    ctx = make_context(a)
    Pred = importlib.import_module("harness.predictor")

    class Fixed:
        def predict(self, task, model):
            return Pred.Prediction(100.0, 250.0, 1.0)

        def p_unsolvable(self, task):
            return 0.0

    tasks = [{"task_id": f"t{i}", "type": "math", "reward": r, "unans": False} for i, r in enumerate([1, 5, 3, 2])]
    store = Store()
    for t in tasks:
        store.add(t["task_id"], "m", Outcome(100.0, True, 100.0, True, False, 1.0))
    r = simulate(tasks, store, budget=250, models=("m",), harness=ctx["h"], predictor=Fixed(), workers=5)
    # 100 tokens each, budget 250: the third task does not fit (100 + 100 reserved + 100 > 250) -> the two best (5 and 3)
    assert r["n_run"] == 2 and r["points"] == 8 and r["spent"] == 200, r
    r = simulate(tasks, store, budget=10_000, models=("m",), harness=ctx["h"], predictor=Fixed(), workers=2)
    assert r["n_run"] == 4 and r["points"] == 11, r

    class FixedU(Fixed):
        def p_unsolvable(self, task):
            return 1.0 if task.task_id == "u" else 0.0

    un = tasks + [{"task_id": "u", "type": "math", "reward": 4, "unans": True}]
    store.add("u", "m", Outcome(100.0, False, 100.0, False, False, 1.0))
    r = simulate(un, store, budget=250, models=("m",), harness=ctx["h"], predictor=FixedU(), workers=5)
    assert r["points"] == 12 and r["n_skipped"] == 1, ("a skipped unanswerable task is correct", r)
    print("selftest ok: budget limit, order by reward/token, unanswerable credit")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness-dir", help="directory that contains the `harness/` package (default: the repo of this file)")
    ap.add_argument("--predictor", help="module:Class to use instead of harness.predictor.Predictor")
    ap.add_argument("--scheduler", help="module:Class to use instead of harness.sorter.Scheduler")
    ap.add_argument("--stream-files", default="results.jsonl,results_fix.jsonl")
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--missing", choices=["impute", "error"], default="impute",
                    help="pair without a recorded outcome: fill with the mean of its (type, model) or stop")
    ap.add_argument("--pool", action="store_true",
                    help="complete a replayed run with the stream pool (older code: lower accuracy); default is the run's own cells only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index-tasks")
    r = sub.add_parser("replay")
    r.add_argument("--jitter-draws", type=int, default=0, help="also run N simulations with a perturbed clock and print the spread")
    r.add_argument("--diagnose", action="store_true", help="print the first start where the simulated order differs from the journal")
    r.add_argument("experiment")
    r.add_argument("--arm")
    r.add_argument("--rep", type=int)
    r.add_argument("--budget", type=float, help="what-if: simulate the recorded run at another budget")
    c = sub.add_parser("curve")
    c.add_argument("--replay", help="experiment[:arm[:rep]]")
    c.add_argument("--stream", action="store_true")
    c.add_argument("--budgets", default="10000:150000:10000", help="from:to:step")
    c.add_argument("--draws", type=int, default=10)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--per-type", type=int, default=25)
    c.add_argument("--unanswerable", default="coding:8,search:10")
    c.add_argument("--mode", choices=["hybrid", "pure", "late"], default="hybrid")
    c.add_argument("--csv")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"index-tasks": cmd_index, "replay": cmd_replay, "curve": cmd_curve, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
