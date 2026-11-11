#!/usr/bin/env python3
"""Offline evaluation of the free search signals from verify_search_collect.py (features only, no gateway tokens).

usage: verify_search_eval.py search_a.jsonl [--ctx-min 0.5]
Prints: error structure, AUC per feature, cross-validated AUC of a small logistic model, caught-wrong share at fixed
false-alarm rates, the second path (agreement, gain/loss of replacing the answer, token cost) restricted to doubt tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict

FEATS = ["sc_answered", "sc_ans_found", "sc_tokcov", "sc_ans_in_q", "sc_type_match", "sc_n_occ_files", "sc_n_occ",
         "sc_ctx_overlap", "sc_top_overlap", "sc_in_top_file", "sc_best_rank", "n_files"]


def auc(pos, neg):
    """P(score of a correct answer > score of a wrong one), ties 0.5. pos = scores of correct, neg = of wrong."""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def boot_auc(pos, neg, n=300, seed=1):
    rng = random.Random(seed)
    v = []
    for _ in range(n):
        v.append(auc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]))
    v.sort()
    return v[int(0.025 * n)], v[int(0.975 * n)]


def fit_lr(X, y, l2=1.0, iters=400, lr=0.1):
    d = len(X[0])
    w = [0.0] * (d + 1)
    for _ in range(iters):
        g = [0.0] * (d + 1)
        for xi, yi in zip(X, y):
            z = w[0] + sum(a * b for a, b in zip(w[1:], xi))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = p - yi
            g[0] += e
            for j in range(d):
                g[j + 1] += e * xi[j]
        n = len(X)
        w[0] -= lr * g[0] / n
        for j in range(d):
            w[j + 1] -= lr * (g[j + 1] / n + l2 * w[j + 1] / n)
    return w


def predict(w, x):
    z = w[0] + sum(a * b for a, b in zip(w[1:], x))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def design(rows):
    cols = [f for f in FEATS]
    X = []
    for r in rows:
        x = []
        for f in cols:
            v = r.get(f)
            x.append(0.0 if v is None else float(v))
        x.append(1.0 if r.get("source", "").startswith("squad_v2") else 0.0)
        X.append(x)
    # standardise
    m = [statistics.fmean(c) for c in zip(*X)]
    s = [statistics.pstdev(c) or 1.0 for c in zip(*X)]
    return [[(v - mu) / sd for v, mu, sd in zip(x, m, s)] for x in X]


def cv_scores(rows, k=5, seed=3):
    X, y = design(rows), [1.0 if r["label"] else 0.0 for r in rows]
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    out = [0.0] * len(rows)
    for f in range(k):
        te = set(idx[f::k])
        tr = [i for i in idx if i not in te]
        w = fit_lr([X[i] for i in tr], [y[i] for i in tr])
        for i in te:
            out[i] = predict(w, X[i])
    return out


def caught_at_fa(scores, labels, fa=0.10):
    """Share of wrong answers with score below the threshold that flags `fa` of the correct answers."""
    pos = sorted(s for s, l in zip(scores, labels) if l)
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None, None
    thr = pos[max(0, int(fa * len(pos)) - 1)] if int(fa * len(pos)) >= 1 else pos[0] - 1e-9
    caught = sum(s <= thr for s in neg) / len(neg)
    return caught, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ctx-min", type=float, default=0.5)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path) if l.strip()]
    n, wrong = len(rows), [r for r in rows if not r["label"]]
    print(f"tasks {n}, correct {n - len(wrong)} ({(n - len(wrong)) / n:.3f}), wrong {len(wrong)}")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["source"]][0] += 1
        by[r["source"]][1] += (not r["label"])
    print("\nby source (n, wrong):", {k: tuple(v) for k, v in sorted(by.items())})
    print("wrong answers: unanswerable task", sum(r["unanswerable"] for r in wrong), "| no answer given", sum(not r["answered"] for r in wrong),
          "| answer not in sources", sum(r["answered"] and not r.get("sc_ans_found") for r in wrong),
          "| answer in sources but wrong", sum(r["answered"] and r.get("sc_ans_found") == 1 for r in wrong))
    print("correct answers: not in sources", sum(r["label"] and r["answered"] and not r.get("sc_ans_found") for r in rows))
    print("\nAUC per feature (higher feature value = correct):")
    for f in FEATS:
        pos = [r.get(f) for r in rows if r["label"] and r.get(f) is not None]
        neg = [r.get(f) for r in rows if not r["label"] and r.get(f) is not None]
        v = auc(pos, neg)
        if v is not None:
            lo, hi = boot_auc(pos, neg, 150)
            print(f"  {f:18s} {v:.3f} [{lo:.2f}-{hi:.2f}]")
    cvs = cv_scores(rows)
    pos = [s for s, r in zip(cvs, rows) if r["label"]]
    neg = [s for s, r in zip(cvs, rows) if not r["label"]]
    lo, hi = boot_auc(pos, neg, 150)
    print(f"\nlogistic model on the free features, 5-fold CV AUC {auc(pos, neg):.3f} [{lo:.2f}-{hi:.2f}]")
    for fa in (0.05, 0.10, 0.20):
        c, thr = caught_at_fa(cvs, [r["label"] for r in rows], fa)
        print(f"  false alarms {fa:.0%}: wrong answers caught {c:.3f} (threshold p<={thr:.3f})")
    # rule-based doubt
    import sys
    sys.path.insert(0, ".")
    try:
        from harness import search_check as SC
    except Exception:  # noqa: BLE001
        SC = None
    if SC is not None:
        for cm in (0.3, 0.5, 0.7, a.ctx_min):
            fl = [SC.doubt(r, cm)[0] for r in rows]
            wr, co = [f for f, r in zip(fl, rows) if not r["label"]], [f for f, r in zip(fl, rows) if r["label"]]
            print(f"  doubt rule ctx_min={cm}: flags {sum(fl)}/{n}; wrong caught {sum(wr) / len(wr):.3f}; correct flagged {sum(co) / len(co):.3f}")
    # second path
    d = [r for r in rows if r["doubt"] and r["agree2"] is not None]
    print(f"\nsecond path on doubt tasks (rule ctx_min=0.5 at collection): {len(d)} of {n}")
    if d:
        agree = [r for r in d if r["agree2"]]
        dis = [r for r in d if not r["agree2"]]
        print(f"  agree {len(agree)} (first correct {sum(r['label'] for r in agree)}), disagree {len(dis)} (first correct {sum(r['label'] for r in dis)}, second correct {sum(bool(r['label2']) for r in dis)})")
        gain = sum((not r["label"]) and bool(r["label2"]) for r in dis)
        loss = sum(r["label"] and not r["label2"] for r in dis)
        print(f"  replace on disagree: fixes {gain}, breaks {loss}, net {gain - loss}")
        print(f"  second path mean tokens {statistics.fmean(r['cost2'] for r in d):.0f}; main pass mean tokens {statistics.fmean(r['cost_main'] for r in rows):.0f}")
    allp = [r for r in rows if r["agree2"] is not None]
    pos = [float(r["agree2"]) for r in allp if r["label"]]
    neg = [float(r["agree2"]) for r in allp if not r["label"]]
    print(f"  agreement of the two paths as a signal on ALL tasks: AUC {auc(pos, neg):.3f}; wrong caught (disagree) {sum(not r['agree2'] for r in allp if not r['label']) / max(1, len(neg)):.3f}, correct flagged {sum(not r['agree2'] for r in allp if r['label']) / max(1, len(pos)):.3f}")
    print("\nby source: accuracy of first path vs second path and agreement")
    for s in sorted(by):
        rs = [r for r in rows if r["source"] == s and r["agree2"] is not None]
        if rs:
            print(f"  {s:20s} n={len(rs):3d} first {statistics.fmean(r['label'] for r in rs):.2f} second {statistics.fmean(bool(r['label2']) for r in rs):.2f} agree {statistics.fmean(r['agree2'] for r in rs):.2f}")


if __name__ == "__main__":
    main()
