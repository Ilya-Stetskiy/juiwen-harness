"""Summarize /tmp/ab/results.jsonl (lab tool).

usage: python ab_summary.py <group> [results.jsonl]
"""
from __future__ import annotations

import json
import sys


def main(group: str, path: str) -> None:
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        g = r["result"]["groups"].get(group)
        if g is None:
            continue
        print(f"{r['label']:14s} #{r['repeat']} correct={g['correct']:3d}/{g['n']:<3d} wrong={g['wrong']:3d} "
              f"noans={g['no_answer']:2d} false_att={g['false_attempt']:2d} tokens={g['tokens']:8.1f} "
              f"score={r['result']['score']:.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/tmp/ab/results.jsonl")
