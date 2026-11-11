#!/usr/bin/env python3
"""Rows for retraining the cost model on ANY task set (e.g. the organisers' tasks), from harness run journals.

Needs the harness venv (openjiuwen, tokenizers). Output: jsonl, one row per (task, model) with runtime features and the real cost.

usage: cost_features.py --tasks DIR [--tasks DIR2 ...] --journals LOGDIR [--journals LOGDIR2 ...] --out rows.jsonl
  DIR      a tasks directory in the competition format (task.json + workspace/), as passed to run_agent.py --tasks-dir
  LOGDIR   a harness log directory (HARNESS_LOG_DIR) or a lab journal folder: events task_start (model, task) and task_end (cost)
Rows use the cost of the task_end event (ledger formula input/10 + output, whole task incl. fallback and verification).
Then:  python predictor_export.py --rows rows.jsonl --out ../profiles/cost_model.json      (scratch venv with scikit-learn)
(tools/retrain_cost_model.sh runs both.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import predictor as PR                       # noqa: E402
from harness.tasks import load_tasks                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", action="append", required=True)
    ap.add_argument("--journals", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gp = PR.GbmPredictor.__new__(PR.GbmPredictor)          # only the feature extraction, no model needed
    gp._mat = {}
    tasks = {}
    for d in a.tasks:
        for t in load_tasks(d):
            tasks[t.task_id] = t
    n = 0
    with open(a.out, "w") as out:
        for jd in a.journals:
            files = glob.glob(os.path.join(jd, "**", "*.jsonl"), recursive=True)
            for f in files:
                model, cost = {}, {}
                for line in open(f, encoding="utf-8", errors="replace"):
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("event") == "task_start":
                        model[e["task"]] = e.get("model")
                    elif e.get("event") == "task_end" and e.get("status") in ("solved", "unsolvable", "failed") and e.get("cost", 0) > 0:
                        cost[e["task"]] = e["cost"]
                for tid, c in cost.items():
                    t, m = tasks.get(tid), model.get(tid)
                    if t is None or not m or t.task_type not in PR._TYPES:
                        continue
                    tq, tm, nf = gp._task_features(t)
                    out.write(json.dumps({"task_id": tid, "type": t.task_type, "model": m, "hybrid_cost": c, "ws_bytes": int(t.workspace_bytes),
                                          "n_files": nf, "tok_instr": tq, "tok_prompt": tm}) + "\n")
                    n += 1
    print("rows", n, "->", a.out)


if __name__ == "__main__":
    main()
