"""Evaluate harness.contradiction on a unified coding dataset (lab tool, read-only).

usage: python eval_contradiction.py <dataset.jsonl> [--show N]
Prints, for rows with a test_mylib.py: how many unanswerable tasks the detector flags (recall) and how many answerable
tasks it flags by mistake (false positives), plus a few examples of misses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.contradiction import find_contradictions  # noqa: E402


def main(path: str, show: int) -> None:
    tp = fn = fp = tn = 0
    misses, false_pos = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        test = (r.get("files") or {}).get("test_mylib.py")
        if test is None:
            continue
        unans = r["check"] == "unanswerable"
        found = find_contradictions(test)
        if unans and found:
            tp += 1
        elif unans:
            fn += 1
            misses.append(r["task_id"])
        elif found:
            fp += 1
            false_pos.append((r["task_id"], found[0]))
        else:
            tn += 1
    n_u, n_a = tp + fn, fp + tn
    print(f"{path}: unanswerable {n_u} (flagged {tp}, recall {tp / n_u:.3f})" if n_u else f"{path}: no unanswerable rows")
    print(f"answerable {n_a} (flagged by mistake {fp}, false-positive rate {fp / n_a:.4f})" if n_a else "no answerable rows")
    for t in misses[:show]:
        print("MISS", t)
    for t, f in false_pos[:show]:
        print("FALSE POSITIVE", t, f)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 3)
