#!/usr/bin/env python3
"""Stage 3: what is a verifier worth in points per token? Replays "second opinion" policies on verify_collect.py rows.

Every row has the outcome of the main pass (`label`) and of the same pipeline with another model (`label_other`),
plus the judge verdict and measured per-pass costs, so policies are evaluated without new gateway calls.

Policies (per task type), all relative to P0 = main answer only:
  Pall     run the other model for every task and take ITS answer (blind replacement; shows why a verdict is needed)
  Pjudge   judge every answer (cost judge_cost); where the judge says NO, run the other model and take its answer
  Pscore   the same gated by an out-of-fold combined score p < thr (free score + judge)
  Pdis     run the other model for every task; when the answers disagree AND the judge says NO keep the other answer
Reported per 100 tasks: extra points (rewards as in the sample) and extra tokens, points per 1000 extra tokens.
A "retry" is a second opinion of a different model, whose measured correctness is used (no assumption about rescue
rates). The fallback agent is not simulated here (not collected); its measured price is ~1 300 tokens per point
(SCRIPTS.md, coding_late_tier) and serves as the break-even reference.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from verify_analyze import FREE, cv_scores, prep

warnings.filterwarnings("ignore")


def oof_score(g, extra):
    cols = [c for c in FREE + extra if c in g and g[c].notna().any() and g[c].nunique() > 1]
    X = g[cols].astype(float)
    X = X.fillna(X.mean()).values
    return cv_scores(X, g["label"].astype(int).values)


def row(name, g, y_new, extra_cost, n):
    d = ((y_new - g["label"]) * g["reward"]).sum()
    s = 100.0 / n
    return {"policy": name, "extra_pts_per100": round(d * s, 2), "extra_tok_per100": round(extra_cost * s),
            "pts_per_1k_tok": round(1000 * d / extra_cost, 2) if extra_cost > 0 else np.nan}


def evaluate(g, thrs):
    g = g.reset_index(drop=True)
    n = len(g)
    lab, lo = g["label"].astype(int), g["label_other"].astype(int)
    jno = (g["judge_yes"] == 0)
    jc = g["judge_cost"].fillna(0)
    rows = [{"policy": "P0 main only", "extra_pts_per100": 0.0, "extra_tok_per100": 0, "pts_per_1k_tok": np.nan}]
    rows.append(row("Pall: replace by other model", g, lo, g["cost_other"].sum(), n))
    rows.append(row("Pjudge: judge all, NO -> other", g, np.where(jno, lo, lab), jc.sum() + (g["cost_other"] * jno).sum(), n))
    ps = oof_score(g, ["judge_yes", "quote_in_docs", "quote_has_ans"])
    g = g.assign(p=ps)
    for t in thrs:
        m = (g["p"] < t).values
        rows.append(row(f"Pscore p<{t}: -> other", g, np.where(m, lo, lab), jc.sum() + (g["cost_other"] * m).sum(), n))
    dis = (g["agree_other"] == 0)
    rows.append(row("Pdis: other for all; disagree & judge NO -> other", g, np.where(dis & jno, lo, lab),
                    jc.sum() + g["cost_other"].sum(), n))
    # accept/reject table of the judge
    tab = g.groupby("judge_yes")["label"].agg(["size", "mean"]).round(3)
    sel = []
    order = g.sort_values("p", ascending=False)
    for cov in (1.0, 0.8, 0.6, 0.4):
        k = int(round(cov * n))
        sel.append({"auto-accepted share (best p first)": cov, "accuracy of accepted": round(order.head(k)["label"].mean(), 3),
                    "wrong answers caught in rest": f"{int((1 - order.iloc[k:]['label']).sum())} of {int((1 - g['label']).sum())}"})
    return pd.DataFrame(rows), tab, pd.DataFrame(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--thr", default="0.6,0.8,0.9")
    a = ap.parse_args()
    df = prep(pd.concat([pd.read_json(f, lines=True) for f in a.files], ignore_index=True))
    thrs = [float(x) for x in a.thr.split(",")]
    for t, g in df.groupby("type"):
        g = g[g["agree_other"].notna() & g["judge_yes"].notna()]
        if g["label"].nunique() < 2:
            continue
        pol, tab, sel = evaluate(g, thrs)
        print(f"\n## {t}: n={len(g)}, acc={g['label'].mean():.3f}, acc of other model={g['label_other'].mean():.3f}, "
              f"mean reward={g['reward'].mean():.2f}")
        print(pol.to_markdown(index=False))
        print("\njudge verdict vs accuracy:\n" + tab.to_markdown())
        print("\nselective accuracy by combined score (out-of-fold):\n" + sel.to_markdown(index=False))


if __name__ == "__main__":
    main()
