#!/usr/bin/env python3
"""Fit the per-type, per-level logistic models used by harness/verifier.py from verify_collect.py feature files.

usage: verify_fit.py features_a.jsonl [...] [--out profiles/verifier_model.json]
Writes standardised ridge-logistic coefficients (features, mean, scale, coef, intercept) per type and level, plus a
per-type prior. Small samples: the models are meant as a calibrated ranking, not a final truth; refit on more data.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from verify_analyze import FREE, cv_scores, prep

warnings.filterwarnings("ignore")

LEVELS = {
    "free": FREE,
    "judge": FREE + ["judge_yes", "quote_in_docs", "quote_has_ans"],
    "other": FREE + ["agree_other", "other_answered"],
    "other_judge": FREE + ["agree_other", "other_answered", "judge_yes", "quote_in_docs", "quote_has_ans"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "profiles" / "verifier_model.json"))
    a = ap.parse_args()
    df = prep(pd.concat([pd.read_json(f, lines=True) for f in a.files], ignore_index=True))
    out = {"_prior": {}, "_meta": {"n": {}, "cv_auc": {}}}
    for t, g in df.groupby("type"):
        g = g.reset_index(drop=True)
        y = g["label"].astype(int).values
        out["_prior"][t] = float(y.mean())
        out["_meta"]["n"][t] = len(g)
        out[t] = {}
        for lvl, cols in LEVELS.items():
            cols = [c for c in cols if c in g and g[c].notna().any() and g[c].nunique() > 1]
            if not cols or y.min() == y.max():
                continue
            X = g[cols].astype(float)
            mean = X.mean()
            X = X.fillna(mean).values
            mu, sd = X.mean(0), X.std(0)
            sd[sd == 0] = 1.0
            lr = LogisticRegression(C=0.3, max_iter=1000).fit((X - mu) / sd, y)
            out[t][lvl] = {"features": cols, "mean": [float(m) for m in mu], "scale": [float(s) for s in sd],
                           "coef": [float(c) for c in lr.coef_[0]], "intercept": float(lr.intercept_[0])}
            p = cv_scores(X, y)
            if p is not None:
                out["_meta"]["cv_auc"][f"{t}/{lvl}"] = round(float(roc_auc_score(y, p)), 3)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", a.out, json.dumps(out["_meta"]))


if __name__ == "__main__":
    main()
