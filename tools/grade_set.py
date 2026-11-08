"""Grade a harness run on a materialized task set (lab tool).

usage: python grade_set.py <tasks_dir> <log_dir> [--list-wrong]
Rules: answerable task is correct iff answer.txt matches `expected` (or tests pass for code tasks);
a task marked `unanswerable` is handled correctly iff answer.txt does NOT exist.
"""
from __future__ import annotations

import collections
import glob
import json
import pathlib
import re
import subprocess
import sys


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip("`*\"'").rstrip(".").strip()).casefold()


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


def is_correct(check: str, expected, ans: str, ws: pathlib.Path) -> bool:
    if check == "exact_int":
        a, e = _num(ans), _num(str(expected))
        return a is not None and e is not None and a == e
    if check == "exact_num":
        a, e = _num(ans), _num(str(expected))
        return a is not None and e is not None and abs(a - e) <= 1e-6 * max(1.0, abs(e))
    if check == "exact_str":
        return _norm(ans) == _norm(str(expected))
    if check == "exact_set":
        sp = lambda t: sorted(_norm(x) for x in re.split(r"\|", t) if x.strip())
        return sp(ans) == sp(str(expected))
    if check in ("unittest", "pytest"):
        if ans.strip() != "DONE":
            return False
        r = subprocess.run([sys.executable, "test_mylib.py"], cwd=ws, capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    return False


def main(tasks_dir: str, log_dir: str, list_wrong: bool) -> None:
    root = pathlib.Path(tasks_dir)
    grading = json.load(open(str(root) + "_grading.json", encoding="utf-8"))
    rows = [json.loads(line) for f in sorted(glob.glob(f"{log_dir}/run-*.jsonl")) for line in open(f, encoding="utf-8")]
    cost = collections.defaultdict(float)
    calls = collections.Counter()
    for r in rows:
        if r["event"] == "llm_call":
            cost[r["task"]] += r["cost"]
            calls[r["task"]] += 1
    fallbacks = {r["task"] for r in rows if r["event"] == "pipeline_fallback"}
    stat = collections.defaultdict(lambda: collections.Counter())
    tok = collections.defaultdict(float)
    wrong = []
    score = max_score = 0.0
    for tid, g in grading.items():
        d = root / tid
        a = d / "answer.txt"
        ans = a.read_text(encoding="utf-8").strip() if a.exists() else None
        key = g["task_type"] + ("/unans" if g["unanswerable"] else "")
        s = stat[key]
        s["n"] += 1
        tok[key] += cost.get(tid, 0.0)
        if g["unanswerable"]:
            if ans is None:
                s["correct"] += 1
            else:
                s["false_attempt"] += 1
                wrong.append((tid, "answered unanswerable", ans[:60]))
            continue
        max_score += g["reward"]
        if ans is None:
            s["no_answer"] += 1
            wrong.append((tid, "no answer", ""))
        elif is_correct(g["check"], g["expected"], ans, d / "workspace"):
            s["correct"] += 1
            score += g["reward"]
        else:
            s["wrong"] += 1
            wrong.append((tid, "wrong", f"{ans[:50]!r} vs {str(g['expected'])[:50]!r}"))
        if tid in fallbacks:
            s["fallback"] += 1
    total_tok = sum(tok.values())
    if "--json" in sys.argv:
        print(json.dumps({
            "groups": {k: {"n": s["n"], "correct": s["correct"], "wrong": s["wrong"], "no_answer": s["no_answer"],
                           "false_attempt": s["false_attempt"], "tokens": round(tok[k], 1)} for k, s in stat.items()},
            "score": score, "max_score": max_score, "tokens": round(total_tok, 1), "calls": sum(calls.values()),
        }))
        return
    print(f"{'group':22s} {'n':>4s} {'correct':>8s} {'wrong':>6s} {'noans':>6s} {'falseatt':>9s} {'fallbk':>7s} {'tok/task':>9s}")
    for key in sorted(stat):
        s = stat[key]
        print(f"{key:22s} {s['n']:4d} {s['correct']:8d} {s['wrong']:6d} {s['no_answer']:6d} {s['false_attempt']:9d} "
              f"{s['fallback']:7d} {tok[key] / s['n']:9.1f}")
    print(f"score {score:.0f} of {max_score:.0f} (answerable only) | tokens {total_tok:.0f} | llm calls {sum(calls.values())}")
    for r in rows:
        if r["event"] == "run_end":
            print("run_end:", {k: v for k, v in r.items() if k in ("spent", "calls", "seconds", "solved_llm")})
    if list_wrong:
        for w in wrong:
            print("WRONG", *w)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--list-wrong" in sys.argv)
