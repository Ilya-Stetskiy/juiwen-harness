#!/usr/bin/env python3
"""Stage 2 analysis: per-signal AUC / coverage / cost and combined scores, from verify_collect.py feature files.

usage: verify_analyze.py features_a.jsonl [more.jsonl ...] [--json out.json]
Prints markdown tables. Only the feature files (no text) are read.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

FREE = ["fmt_ok", "prog_ok", "prog_nan", "hardcode", "retry_used", "ans_len", "ans_in_docs", "ans_int", "ans_log10",
        "ans_numeric", "n_ops", "code_len", "answered"]
SIGNAL_COST = {"answerable_yes": "judge2_cost", "agree_rep": "cost_rep", "agree_other": "cost_other", "agree_indep": "cost_indep",
               "n_agree_main": "cost_all_agree", "n_distinct": "cost_all_agree", "judge_yes": "judge_cost",
               "quote_in_docs": "judge_cost", "quote_has_ans": "judge_cost"}
SETS = {
    "meta": ["cost_main", "calls_main", "instr_len", "ws_bytes", "retry_used", "answered"],
    "free": FREE,
    "free+rep": FREE + ["agree_rep", "rep_answered"],
    "free+judge": FREE + ["judge_yes", "quote_in_docs", "quote_has_ans"],
    "free+rep+judge": FREE + ["agree_rep", "rep_answered", "judge_yes", "quote_in_docs", "quote_has_ans"],
    "free+rep+other": FREE + ["agree_rep", "rep_answered", "agree_other", "other_answered"],
    "free+other": FREE + ["agree_other", "other_answered"],
    "free+judge2": FREE + ["judge_yes", "quote_in_docs", "quote_has_ans", "answerable_yes"],
    "free+other+judge": FREE + ["agree_other", "other_answered", "judge_yes", "quote_in_docs", "quote_has_ans",
                                "answerable_yes"],
    "all": FREE + ["agree_rep", "rep_answered", "agree_other", "other_answered", "agree_indep", "judge_yes",
                   "quote_in_docs", "quote_has_ans", "n_agree_main", "n_distinct"],
}
SET_COST = {  # tokens per task (mean) of the paid signals in each set; filled from the data
    "meta": [], "free": [], "free+rep": ["cost_rep"], "free+judge": ["judge_cost"],
    "free+rep+judge": ["cost_rep", "judge_cost"], "free+rep+other": ["cost_rep", "cost_other"],
    "free+other": ["cost_other"], "free+judge2": ["judge_cost", "judge2_cost"],
    "free+other+judge": ["cost_other", "judge_cost", "judge2_cost"],
    "all": ["cost_rep", "cost_other", "cost_indep", "judge_cost", "judge2_cost"],
}


def ece(y, p, bins=5):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"y": y, "p": p, "q": q})
    return float(sum(len(g) / len(df) * abs(g.p.mean() - g.y.mean()) for _, g in df.groupby("q")))


def cv_scores(X, y, repeats=20, seed=0):
    """Repeated stratified CV of a ridge logistic regression; returns out-of-fold mean predictions."""
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    k = min(5, n_pos, n_neg)
    if k < 2:
        return None
    pred = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
        m.fit(X[tr], y[tr])
        pred[te] += m.predict_proba(X[te])[:, 1]
        cnt[te] += 1
    return pred / cnt


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    df["cost_all_agree"] = df[["cost_rep", "cost_other", "cost_indep"]].fillna(0).sum(axis=1)
    return df


def single_signal_table(g: pd.DataFrame):
    rows = []
    y = g["label"].astype(int).values
    for c in sorted(set(FREE + list(SIGNAL_COST) + ["cost_main", "calls_main", "instr_len", "ws_bytes", "ans_neg", "answerable_yes"])):
        if c not in g or g[c].isna().all():
            continue
        v = g[c]
        cov = float(v.notna().mean())
        ok = v.notna()
        if v[ok].nunique() < 2 or y[ok.values].min() == y[ok.values].max():
            continue
        auc = roc_auc_score(y[ok.values], v[ok].astype(float))
        cost = SIGNAL_COST.get(c)
        rows.append({"signal": c, "auc": round(auc, 3), "auc_oriented": round(max(auc, 1 - auc), 3),
                     "sign": "+" if auc >= 0.5 else "-", "coverage": round(cov, 2),
                     "tokens_per_task": round(float(g[cost].fillna(0).mean()), 1) if cost else 0.0})
    return pd.DataFrame(rows).sort_values("auc_oriented", ascending=False)


def combined(g: pd.DataFrame):
    y = g["label"].astype(int).values
    out = []
    for name, cols in SETS.items():
        need = [c for c in cols if c not in FREE and c in SETS["all"] + ["answerable_yes"]]
        if any(c not in g or g[c].notna().sum() < 0.8 * len(g) for c in need if c in ("agree_rep", "agree_other", "judge_yes")):
            continue
        cols = [c for c in cols if c in g and g[c].notna().any() and g[c].nunique() > 1]
        if not cols:
            continue
        X = g[cols].astype(float).fillna(g[cols].astype(float).median()).values
        p = cv_scores(X, y)
        if p is None:
            continue
        cost = float(sum(g[c].fillna(0).mean() for c in SET_COST[name] if c in g))
        out.append({"set": name, "n_feat": len(cols), "auc": round(roc_auc_score(y, p), 3),
                    "brier": round(brier_score_loss(y, p), 4), "ece": round(ece(y, p), 3),
                    "tokens_per_task": round(cost, 1)})
    base = brier_score_loss(y, np.full(len(y), y.mean()))
    return pd.DataFrame(out), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    df = prep(pd.concat([pd.read_json(f, lines=True) for f in a.files], ignore_index=True))
    res = {}
    for t, g in df.groupby("type"):
        g = g.reset_index(drop=True)
        y = g["label"].astype(int)
        print(f"\n## {t}: n={len(g)}, correct={y.mean():.3f} (wrong: {int((1 - y).sum())}), "
              f"mean cost main/pass={g['cost_main'].mean():.0f}")
        s = single_signal_table(g)
        print(s.to_markdown(index=False))
        c, base = combined(g)
        print(f"\nCombined (CV, ridge-logistic); const-model Brier={base:.4f}")
        print(c.to_markdown(index=False))
        res[t] = {"n": len(g), "acc": float(y.mean()), "signals": s.to_dict("records"), "combined": c.to_dict("records"),
                  "brier_const": float(base)}
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
