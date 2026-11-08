#!/usr/bin/env python3
"""Does a verifier trained on many cheap local runs (mixed temperatures) hold up on REAL working-regime errors?

Data: multi_*.jsonl from verify_collect_multi.py (k runs per task on the local model: `work` rows in the working regime,
`hot` rows at raised temperature) and, optionally, the gateway feature files features_a/b.jsonl (working regime, real
gateway errors, none of its tasks appear in the local set).

Protocol (per task type): split TASKS (not rows) into train/test, repeated `--splits` times.
  train A: all rows of the train tasks (mix of temperatures)
  train B: working-regime rows of the train tasks only
  test 1 : working-regime rows of the test tasks (local model)
  test 2 : gateway rows (all, they are never used for training here)
Feature sets: free | vote (free + share of agreeing working-regime runs of the same task) |
              judge (free + judge, working rows only) | vote_judge.
Reports pooled AUC (mean over splits) with a task-bootstrap interval, and the error rate per regime (noise check).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from verify_analyze import FREE

warnings.filterwarnings("ignore")
VOTE = ["vote_frac_work", "n_clusters_work", "is_plural"]
JUDGE = ["judge_yes", "quote_in_docs", "quote_has_ans"]
SETS = {"free": FREE, "vote": FREE + VOTE, "judge": FREE + JUDGE, "vote_judge": FREE + VOTE + JUDGE}


def fit(tr, cols, y):
    cols = [c for c in cols if c in tr and tr[c].notna().any() and tr[c].nunique() > 1]
    X = tr[cols].astype(float)
    mu = X.mean()
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)).fit(X.fillna(mu).values, y)
    return m, cols, mu


def score(model, te):
    m, cols, mu = model
    X = te.reindex(columns=cols).astype(float).fillna(mu)
    return m.predict_proba(X.values)[:, 1]


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def boot_ci(te, p, B=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = te["task_id"].values
    uniq = np.unique(ids)
    idx = {u: np.where(ids == u)[0] for u in uniq}
    y = te["label"].astype(int).values
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([idx[u] for u in pick])
        out.append(auc(y[ii], p[ii]))
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("multi", nargs="+")
    ap.add_argument("--gateway", nargs="*", default=[])
    ap.add_argument("--splits", type=int, default=10)
    a = ap.parse_args()
    d = pd.concat([pd.read_json(f, lines=True) for f in a.multi], ignore_index=True)
    for c in d.columns:
        if d[c].dtype == bool:
            d[c] = d[c].astype(int)
    g_all = (pd.concat([pd.read_json(f, lines=True) for f in a.gateway], ignore_index=True) if a.gateway else None)
    if g_all is not None:
        for c in g_all.columns:
            if g_all[c].dtype == bool:
                g_all[c] = g_all[c].astype(int)
    rows = []
    for t, g in d.groupby("type"):
        reg = g.groupby("regime")["label"].agg(["size", "mean"]).round(3)
        print(f"\n## {t}: tasks={g.task_id.nunique()}, rows={len(g)}, wrong rows={int((1 - g.label).sum())} "
              f"(work wrong {int((1 - g[g.regime == 'work'].label).sum())})")
        print("accuracy by regime:", reg.to_dict("index"))
        gt = g_all[g_all["type"] == t] if g_all is not None else None
        tasks = g.task_id.unique()
        res = {}
        for sp in range(a.splits):
            rng = np.random.default_rng(sp)
            tr_t = set(rng.choice(tasks, len(tasks) // 2, replace=False))
            tr = g[g.task_id.isin(tr_t)]
            te = g[~g.task_id.isin(tr_t) & (g.regime == "work")].reset_index(drop=True)
            for name, cols in SETS.items():
                has_judge = name in ("judge", "vote_judge")
                for tag, trn in (("mix", tr[tr.regime == "work"] if has_judge else tr), ("work", tr[tr.regime == "work"])):
                    y = trn["label"].astype(int).values
                    if y.min() == y.max():
                        continue
                    mdl = fit(trn, cols, y)
                    p = score(mdl, te)
                    res.setdefault((name, tag, "local-work-test"), []).append(auc(te["label"].astype(int).values, p))
                    if gt is not None and len(gt) and name in ("free", "judge"):
                        res.setdefault((name, tag, "gateway"), []).append(
                            auc(gt["label"].astype(int).values, score(mdl, gt)))
        out = pd.DataFrame([{"features": k[0], "train": ("mix of temps" if k[1] == "mix" else "working only"), "test": k[2],
                             "auc": round(float(np.nanmean(v)), 3), "sd_splits": round(float(np.nanstd(v)), 3)}
                            for k, v in res.items()]).sort_values(["test", "features", "train"])
        print(out.to_markdown(index=False))
        rows.append((t, out))


if __name__ == "__main__":
    main()
