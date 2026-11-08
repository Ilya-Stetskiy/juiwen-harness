#!/usr/bin/env python3
"""Stage 1 of the verifier study: how well do metadata alone predict `correct`? (lower bound for every other signal)

Reads the stream results (one row per task x model, metadata only) and fits a gradient-boosting model with
grouped cross-validation (a task never sits in both train and test folds, even across models).
Reports AUC, Brier, ECE per task type, plus the AUC of simpler baselines (source prior only).

usage: verify_meta.py STREAM_RESULTS.jsonl [--model SUBSTR] [--no-answered]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def calib_table(y, p, bins=5):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"y": np.asarray(y, float), "p": p, "q": q})
    return [(round(g.p.mean(), 3), round(g.y.mean(), 3), len(g)) for _, g in df.groupby("q")]


def features(d: pd.DataFrame, use_answered: bool = True) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["source"] = d["source"].astype("category").cat.codes
    X["check"] = d["check"].astype("category").cat.codes
    X["reward"] = d["reward"]
    X["noreason"] = d["model"].str.contains("noreason").astype(int)
    X["m27"] = d["model"].str.contains("27b").astype(int)
    X["pipeline_calls"] = d["pipeline_calls"]
    X["agent_calls"] = d["agent_calls"]
    X["fallback"] = d["fallback"].astype(int)
    X["in_tokens"] = d["in_tokens"]
    X["out_tokens"] = d["out_tokens"]
    X["cost"] = d["hybrid_cost"]
    X["seconds"] = d["seconds"]
    X["status_unsolvable"] = (d["status"] == "unsolvable").astype(int)
    if use_answered:
        X["answered"] = d["answered"].astype(int)
    return X


def cv_predict(X, y, groups, folds=5, seed=0):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150, l2_regularization=1.0,
                                           random_state=seed)
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--model", default="")
    ap.add_argument("--no-answered", action="store_true")
    a = ap.parse_args()
    d = pd.read_json(a.path, lines=True)
    d = d[d["status"].isin(["solved", "unsolvable", "failed"])]          # drop infrastructure errors / not_run
    d = d[d["model"].str.contains(a.model)].reset_index(drop=True)
    out = {}
    for t, g in list(d.groupby("type")) + [("ALL", d)]:
        g = g.reset_index(drop=True)
        y = g["correct"].astype(int).values
        if y.min() == y.max():
            continue
        X = features(g, not a.no_answered)
        p = cv_predict(X, y, g["task_id"].values)
        # source prior only
        prior = g.groupby("source")["correct"].transform("mean").values
        row = {"n": len(g), "base_rate": round(y.mean(), 3), "auc_meta": round(roc_auc_score(y, p), 3),
               "auc_source_prior": round(roc_auc_score(y, prior), 3), "brier": round(brier_score_loss(y, p), 4),
               "brier_const": round(brier_score_loss(y, np.full(len(y), y.mean())), 4), "ece": round(ece(y, p), 3),
               "calib(p_mean,acc,n)": calib_table(y, p)}
        out[t] = row
        print(t, json.dumps(row, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
