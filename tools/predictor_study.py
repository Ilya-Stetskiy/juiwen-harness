#!/usr/bin/env python3
"""Cost predictor study on the accumulated stream results (needs numpy, sklearn: a scratch venv, not the container).

Joins stream results.jsonl (hybrid cost per task and model) with pred_data.npz from predictor_dataset.py and compares
predictors by cross-validated (grouped by task) APE = |pred - actual| / actual, the metric of the current predictor (0.61).

usage: predictor_study.py --results ~/datasets/token_juiwen/stream/results.jsonl --data ~/harness-lab/verify/pred_data.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

TYPES = ["math", "search", "data_analysis", "coding"]
SOURCES = ["gsm8k", "svamp", "asdiv", "math500", "math_algebra", "math_prealgebra", "math_number_theory", "math_counting_and_probability",
           "math_geometry", "math_intermediate_algebra", "math_precalculus", "hotpotqa", "2wikimultihopqa", "musique_full", "squad_v2",
           "tabfact", "wikitablequestions", "dabench", "humanevalfix", "mbpp", "codegen_fix"]


def placeholder(row) -> float:
    base = {"math": 60, "search": 250, "data_analysis": 100, "coding": 300}.get(row["type"], 1500)
    return base * (4.0 if row["thinking"] else 1.0) + row["ws_bytes"] / 40.0


def measured(row) -> float:
    base = {"math": 289.0, "data_analysis": 335.0, "search": 160.0, "coding": 810.0}.get(row["type"], 1500.0)
    return base * (3.0 if row["thinking"] else 1.0) + row["ws_bytes"] / 40.0


def ape(pred, y):
    return np.abs(pred - y) / y


def report(name, pred, y, types):
    a = ape(pred, y)
    per = {t: float(np.median(a[types == t])) for t in TYPES}
    print(f"{name:42s} median APE {np.median(a):.3f}  mean {a.mean():.3f}  | " + "  ".join(f"{t[:4]} {v:.2f}" for t, v in per.items()))
    return float(np.median(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    quick = a.quick
    D = np.load(a.data, allow_pickle=True)
    feat = {tid: i for i, tid in enumerate(D["task_id"])}
    rows = []
    for l in open(a.results):
        r = json.loads(l)
        if r["task_id"] in feat and r.get("hybrid_cost", 0) and r["hybrid_cost"] > 0 and not r.get("unanswerable_skip"):
            rows.append(r)
    n = len(rows)
    idx = np.array([feat[r["task_id"]] for r in rows])
    y = np.array([r["hybrid_cost"] for r in rows], float)
    types = np.array([r["type"] for r in rows])
    thinking = np.array([not r["model"].endswith("-noreason") for r in rows])
    fam38 = np.array([r["model"].startswith("qwen3.8") for r in rows])
    groups = np.array([r["task_id"] for r in rows])
    base = [dict(type=t, thinking=th, ws_bytes=int(D["ws_bytes"][i])) for t, th, i in zip(types, thinking, idx)]
    print(f"rows {n}, tasks {len(set(groups))}; median cost {np.median(y):.0f}, by type " +
          ", ".join(f"{t} {np.median(y[types == t]):.0f}" for t in TYPES))
    report("placeholder (current)", np.array([placeholder(b) for b in base]), y, types)
    report("measured per-type table", np.array([measured(b) for b in base]), y, types)

    def onehot(vals, cats):
        return np.array([[v == c for c in cats] for v in vals], float)
    Xc = np.column_stack([onehot(types, TYPES), onehot(D["source"][idx], SOURCES), thinking, fam38, D["reward"][idx], D["ws_bytes"][idx],
                          D["n_files"][idx], D["tok_instr"][idx], D["tok_prompt"][idx]])
    cols_cheap = Xc
    emb = D["emb"][idx]
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Xc, y, groups))

    # container-friendly alternative (numpy only): per type, log cost ~ log(prompt tokens) + log(ws bytes) + thinking + family
    lin = np.zeros(n)
    lo_q, hi_q = np.zeros(n), np.zeros(n)
    Xl = np.column_stack([np.ones(n), np.log1p(D["tok_prompt"][idx]), np.log1p(D["tok_instr"][idx]), np.log1p(D["ws_bytes"][idx]),
                          np.log1p(D["n_files"][idx]), thinking, fam38])
    for tr, te in folds:
        for t in TYPES:
            a_ = tr[types[tr] == t]
            b_ = te[types[te] == t]
            if len(a_) == 0 or len(b_) == 0:
                continue
            w, *_r = np.linalg.lstsq(Xl[a_], np.log(y[a_]), rcond=None)
            res = np.log(y[a_]) - Xl[a_] @ w
            lin[b_] = np.exp(Xl[b_] @ w + np.median(res))
            lo_q[b_] = np.exp(Xl[b_] @ w + np.quantile(res, 0.1))
            hi_q[b_] = np.exp(Xl[b_] @ w + np.quantile(res, 0.9))
    report("per-type linear in log tokens (numpy only)", lin, y, types)
    cov = (y >= lo_q) & (y <= hi_q)
    print(f"  linear model range p10..p90 (residual quantiles): coverage {cov.mean():.3f}, median width x{np.median(hi_q / lo_q):.2f}")

    def cv_gbm(X, name, loss="absolute_error", target="log"):
        pred = np.zeros(n)
        for tr, te in folds:
            m = HistGradientBoostingRegressor(loss=loss, max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
            m.fit(X[tr], np.log(y[tr]))
            pred[te] = np.exp(m.predict(X[te]))
        report(name, pred, y, types)
        return pred

    # type x thinking median (trained inside folds)
    pm = np.zeros(n)
    for tr, te in folds:
        for t in TYPES:
            for th in (False, True):
                mt = (types[tr] == t) & (thinking[tr] == th)
                med = np.median(y[tr][mt]) if mt.any() else np.median(y[tr])
                sel = te[(types[te] == t) & (thinking[te] == th)]
                pm[sel] = med
    report("median per type x thinking (CV)", pm, y, types)
    if quick:
        return
    p_cheap = cv_gbm(cols_cheap, "GBM cheap features (exact token counts)")
    # what exists at competition time: no `source` (task ids are not dataset names there) and no reward
    ns = len(SOURCES)
    keep = list(range(len(TYPES))) + list(range(len(TYPES) + ns, Xc.shape[1]))
    keep = [k for k in keep if k != len(TYPES) + ns + 2]        # drop reward (column after thinking, fam38)
    Xrt = Xc[:, keep]
    p_rt = cv_gbm(Xrt, "GBM runtime features (no source, no reward)")
    Xnotok = np.delete(Xc, [Xc.shape[1] - 2, Xc.shape[1] - 1], axis=1)
    cv_gbm(Xnotok, "GBM without exact token counts")
    pca = PCA(n_components=24, random_state=0).fit(emb)
    Xe = np.column_stack([Xc, pca.transform(emb)])
    p_emb = cv_gbm(Xe, "GBM cheap + potion embedding (PCA 24)")
    Xre = np.column_stack([Xrt, pca.transform(emb)])
    p_rte = cv_gbm(Xre, "GBM runtime + potion embedding")
    pca8 = PCA(n_components=8, random_state=0).fit(emb)
    cv_gbm(np.column_stack([Xrt, pca8.transform(emb)]), "GBM runtime + potion embedding (PCA 8)")
    # ridge on embedding + type, log target
    pr = np.zeros(n)
    Xr = np.column_stack([Xc[:, : len(TYPES)], thinking, fam38, np.log1p(D["tok_prompt"][idx]), np.log1p(D["ws_bytes"][idx]), emb])
    for tr, te in folds:
        m = Ridge(alpha=5.0).fit(Xr[tr], np.log(y[tr]))
        pr[te] = np.exp(m.predict(Xr[te]))
    report("ridge on embedding + type (log)", pr, y, types)

    # range: quantile GBM on the cheap features
    q = {}
    for al in (0.1, 0.5, 0.9):
        pred = np.zeros(n)
        for tr, te in folds:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=al, max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                              min_samples_leaf=20, random_state=0)
            m.fit(Xre[tr], np.log(y[tr]))
            pred[te] = np.exp(m.predict(Xre[te]))
        q[al] = pred
    cover = (y >= q[0.1]) & (y <= q[0.9])
    print(f"\nrange (quantile GBM, cheap + embedding): p10..p90 covers {cover.mean():.3f} of actual costs (target 0.80); median width x{np.median(q[0.9] / q[0.1]):.2f}")
    for t in TYPES:
        s = types == t
        print(f"  {t:14s} coverage {cover[s].mean():.2f}  median p50 {np.median(q[0.5][s]):.0f}  APE of p50 {np.median(ape(q[0.5][s], y[s])):.2f}")
    above90 = (y > q[0.9]).mean()
    print(f"  share of costs above p90: {above90:.3f}; below p10: {(y < q[0.1]).mean():.3f}")
    # what matters for the scheduler: ranking by reward per token, share of the oracle value in a budget
    rew = D["reward"][idx]
    for name, pr_ in (("placeholder", np.array([placeholder(b) for b in base])), ("GBM runtime", p_rt), ("GBM runtime + emb", p_rte), ("oracle", y)):
        order = np.argsort(-(rew / pr_))
        cum = np.cumsum(y[order])
        budget = 0.3 * y.sum()
        k = np.searchsorted(cum, budget)
        print(f"  reward per token ordering with {name:12s}: reward collected under a budget of 30% of total cost: {rew[order][:k].sum() / rew.sum():.3f}")


if __name__ == "__main__":
    main()
