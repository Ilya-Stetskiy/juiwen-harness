"""Grade a harness run on a copy of the example tasks (uses `expected`; lab only).

usage: python grade_run.py <tasks_dir> <log_dir>
"""
from __future__ import annotations

import glob
import json
import pathlib
import subprocess
import sys


def main(tasks_dir: str, log_dir: str) -> None:
    root = pathlib.Path(tasks_dir)
    rows = [json.loads(line) for f in sorted(glob.glob(f"{log_dir}/run-*.jsonl")) for line in open(f, encoding="utf-8")]
    ends = {r["task"]: r for r in rows if r["event"] == "task_end"}
    calls = [r for r in rows if r["event"] == "llm_call"]
    for r in rows:
        if r["event"] == "run_end":
            print("run_end:", {k: v for k, v in r.items() if k != "ts"})
    score = total = 0
    for d in sorted(root.iterdir()):
        if not (d / "task.json").is_file():
            continue
        t = json.load(open(d / "task.json", encoding="utf-8"))
        a = d / "answer.txt"
        ans = a.read_text(encoding="utf-8").strip() if a.exists() else None
        ok = None
        if t.get("check") in ("exact_int", "exact_str"):
            ok = ans == str(t["expected"])
        elif t.get("check") == "unittest":
            r = subprocess.run(["python3", "test_mylib.py"], cwd=d / "workspace", capture_output=True, text=True)
            ok = r.returncode == 0 and ans == "DONE"
        e = ends.get(t["task_id"], {})
        n = sum(1 for c in calls if c["task"] == t["task_id"])
        total += t["reward"]
        score += t["reward"] if ok else 0
        print(f"{t['task_id']:20s} status={e.get('status')} cost={e.get('cost')} calls={n} "
              f"answer={ans!r} expected={t.get('expected')!r} correct={ok}")
    print("score:", score, "of", total)
    kinds = {}
    for r in rows:
        kinds[r["event"]] = kinds.get(r["event"], 0) + 1
    print("events:", kinds)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
