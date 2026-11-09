#!/usr/bin/env python3
"""Cheap labelled-error collector on the LOCAL model (llama-server via local_shim; no gateway tokens).

Every task is solved k times by the scripted pipeline: `--work` runs in the working regime (no temperature set, the
shim default) and the rest at raised temperatures. Each run is one output row with ONLY features: correctness label,
free signals of that run, agreement with the other runs of the same task (vote share, leave-one-out), temperature
class, the judge verdict (working-regime runs only; judge runs on the same local model). No prompts, programs or
answers are written (competition rule 2.4); answers exist in memory for the duration of one task.

Runs on the host next to verify_collect.py (imports it). usage:
  LLM_BASE_URL=http://127.0.0.1:18444 python verify_collect_multi.py --types search,data_analysis,math \
      --n-by-type search=80,data_analysis=80,math=60 --k 10 --work 4 --seed 9100 --workers 3 --out ~/harness-lab/verify/multi_a.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_collect as VC                      # noqa: E402  (patches pipelines.ask / run_python, defines helpers)
import harness.llm as L                          # noqa: E402

LOCAL = "local-35b-noreason"
VC.MAIN_MODEL = LOCAL                            # the judge uses this model
TEMP: contextvars.ContextVar = contextvars.ContextVar("temp", default=None)
L._sampling = lambda override=None: ({"temperature": override} if override is not None else
                                     ({"temperature": TEMP.get()} if TEMP.get() is not None else {}))
HOT = [0.7, 0.9, 1.0, 1.2]


def schedule(k: int, work: int) -> list:
    return [None] * work + [HOT[i % len(HOT)] for i in range(k - work)]


def clusters(check, answers, ws):
    """Cluster id per answer (None = no answer counts as its own cluster)."""
    reps, ids = [], []
    for a in answers:
        for j, r in enumerate(reps):
            if VC._eq(check, a, r, ws):
                ids.append(j)
                break
        else:
            reps.append(a)
            ids.append(len(reps) - 1)
    return ids


async def one_task(r, src, root: Path, cfg, out_path: Path, sem, k: int, work: int, judge_work: bool):
    async with sem:
        tid = r["task_id"]
        parent = root / "t" / tid
        shutil.rmtree(parent, ignore_errors=True)
        VC.sr.write_task_dir(r, parent / tid)
        task = VC.load_tasks(parent)[0]
        tmp = root / "tmp" / tid
        tmp.mkdir(parents=True, exist_ok=True)
        ws = task.workspace
        docs = VC.read_docs(ws)
        docs_norm = VC.norm_text(docs)
        exp, unans = r.get("expected"), r["check"] == "unanswerable"
        runs = []
        t0 = time.time()
        try:
            for i, temp in enumerate(schedule(k, work)):
                TEMP.set(temp)
                res = await VC.run_pass(task, LOCAL, cfg, tmp)
                jd = await VC.judge(task, res, cfg, tmp, docs) if (judge_work and temp is None) else {}
                runs.append((temp, res, jd))
        except Exception as exc:                  # noqa: BLE001
            print("ERR", tid, type(exc).__name__, str(exc)[:100], flush=True)
            shutil.rmtree(parent, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)
            return
        answers = [x[1]["answer"] for x in runs]
        cl = clusters(r["check"], answers, ws)
        n = len(runs)
        wk = [i for i, x in enumerate(runs) if x[0] is None]
        plural = max(set(cl), key=cl.count)
        rows = []
        for i, (temp, res, jd) in enumerate(runs):
            ans = res["answer"]
            label = (ans is None) if unans else (ans is not None and VC.sr.is_correct(r["check"], exp, ans, ws))
            others = [j for j in range(n) if j != i]
            others_w = [j for j in wk if j != i]
            row = {"task_id": tid, "type": task.task_type, "source": src, "check": r["check"], "unanswerable": unans,
                   "reward": task.reward, "instr_len": len(task.instruction), "ws_bytes": task.workspace_bytes,
                   "run": i, "temp": -1.0 if temp is None else temp, "regime": "work" if temp is None else "hot",
                   "label": bool(label), "answered": ans is not None, "status": res["status"],
                   "cost_main": round(res["cost"], 1), "calls_main": res["calls"], "seconds_task": round(time.time() - t0, 1),
                   "vote_frac": sum(cl[j] == cl[i] for j in others) / len(others),
                   "vote_frac_work": (sum(cl[j] == cl[i] for j in others_w) / len(others_w)) if others_w else None,
                   "is_plural": int(cl[i] == plural), "n_clusters": len(set(cl)),
                   "n_clusters_work": len({cl[j] for j in wk}), "agree_other": int(cl[i] == cl[(i + 1) % n])}
            row.update(VC.code_features(res["runs"]))
            row["fmt_ok"] = VC.fmt_ok(r["check"], ans, task.instruction)
            row.update(VC.extract_feats(task, ans, docs_norm))
            row.update(jd)
            rows.append(row)
        with open(out_path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        shutil.rmtree(parent, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        wl = [x["label"] for x in rows if x["regime"] == "work"]
        print(f"ok {tid} work_acc={sum(wl)}/{len(wl)} acc={sum(x['label'] for x in rows)}/{n} sec={time.time() - t0:.0f}", flush=True)


async def amain(a):
    cfg = VC.Config.from_env()
    out = Path(os.path.expanduser(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["task_id"] for x in open(out) if x.strip()}
    root = Path("/tmp/hl_verify_multi")
    (root / "t").mkdir(parents=True, exist_ok=True)
    nbt = dict(kv.split("=") for kv in a.n_by_type.split(",") if kv)
    rows = []
    for t in a.types.split(","):
        rows += VC.sample([t], int(nbt.get(t, a.n)), a.seed)
    exclude = set()
    if a.exclude:
        exclude = {x.strip() for x in open(os.path.expanduser(a.exclude)) if x.strip()}
    rows = [(r, s) for r, s in rows if r["task_id"] not in done and r["task_id"] not in exclude]
    import random
    random.Random(a.seed).shuffle(rows)             # interleave types so partial results cover all of them
    print(f"{len(rows)} tasks, k={a.k} (work {a.work})", flush=True)
    sem = asyncio.Semaphore(a.workers)
    await asyncio.gather(*(one_task(r, s, root, cfg, out, sem, a.k, a.work, not a.no_judge) for r, s in rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="search,data_analysis,math")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-by-type", default="")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--work", type=int, default=4)
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--exclude", default="")       # file with task_ids to skip (e.g. the gateway training/test sets)
    ap.add_argument("--out", default="~/harness-lab/verify/multi_a.jsonl")
    a = ap.parse_args()
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:18444")
    os.environ.setdefault("LLM_API_KEY", "x")
    os.environ.setdefault("HARNESS_LOG_DIR", "/tmp/hl_verify_multi/logs")
    os.environ["BUDGET"] = "1000000000"
    os.makedirs("/tmp/hl_verify_multi/cwd", exist_ok=True)
    os.chdir("/tmp/hl_verify_multi/cwd")
    asyncio.run(amain(a))
