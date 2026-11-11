#!/usr/bin/env python3
"""Train the cost model on the stream results and export it as JSON trees (evaluated by harness/costmodel.py with numpy only).

Runtime features (what exists before a task is solved; no dataset name, no reward):
  type one-hot (math, search, data_analysis, coding), thinking, family 3.8, workspace bytes, number of files,
  tokens(instruction), tokens(prompt material) -- exact Qwen token counts.
Three gradient-boosted regressors on log(cost): median (p50) and quantiles 0.1 / 0.9 (range).
usage (scratch venv with sklearn): predictor_export.py --results results.jsonl --data pred_data.npz --out profiles/cost_model.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

TYPES = ["math", "search", "data_analysis", "coding"]
FEATURES = TYPES + ["thinking", "fam38", "ws_bytes", "n_files", "tok_instr", "tok_prompt"]


def export(m: HistGradientBoostingRegressor) -> dict:
    trees = []
    for stage in m._predictors:
        nodes = stage[0].nodes
        trees.append({k: nodes[k].tolist() for k in ("feature_idx", "num_threshold", "left", "right", "value", "is_leaf")})
    return {"baseline": float(np.ravel(m._baseline_prediction)[0]), "trees": trees}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="", help="stream results.jsonl (with --data)")
    ap.add_argument("--data", default="", help="pred_data.npz from predictor_dataset.py")
    ap.add_argument("--rows", default="", help="rows.jsonl from cost_features.py (any task set, e.g. the organisers') instead of --results/--data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="", help="file with task ids kept out of training (lab sets: no leakage into lab experiments)")
    a = ap.parse_args()
    if a.rows:
        rows = [json.loads(l) for l in open(a.rows)]
        X = np.column_stack([
            *[[r["type"] == t for r in rows] for t in TYPES],
            [not r["model"].endswith("-noreason") for r in rows], [r["model"].startswith("qwen3.8") for r in rows],
            [r["ws_bytes"] for r in rows], [r["n_files"] for r in rows], [r["tok_instr"] for r in rows], [r["tok_prompt"] for r in rows]]).astype(float)
        y = np.array([r["hybrid_cost"] for r in rows], float)
    else:
        D = np.load(a.data, allow_pickle=True)
        feat = {t: i for i, t in enumerate(D["task_id"])}
        rows = [json.loads(l) for l in open(a.results)]
        excl = {x.strip() for x in open(a.exclude)} if a.exclude else set()
        rows = [r for r in rows if r["task_id"] in feat and r["task_id"] not in excl and r.get("hybrid_cost", 0) and r["hybrid_cost"] > 0]
        idx = np.array([feat[r["task_id"]] for r in rows])
        y = np.array([r["hybrid_cost"] for r in rows], float)
        X = np.column_stack([
            *[[r["type"] == t for r in rows] for t in TYPES],
            [not r["model"].endswith("-noreason") for r in rows], [r["model"].startswith("qwen3.8") for r in rows],
            D["ws_bytes"][idx], D["n_files"][idx], D["tok_instr"][idx], D["tok_prompt"][idx]]).astype(float)
    out = {"features": FEATURES, "n_rows": len(rows), "n_tasks": len(set(r["task_id"] for r in rows)), "models": {}}
    for name, kw in (("p50", dict(loss="absolute_error")), ("p10", dict(loss="quantile", quantile=0.1)), ("p90", dict(loss="quantile", quantile=0.9))):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=20, random_state=0, **kw)
        m.fit(X, np.log(y))
        out["models"][name] = export(m)
        chk = m.predict(X[:50])
        out["models"][name]["check"] = {"x": X[:5].tolist(), "log_pred": chk[:5].tolist()}
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("saved", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
