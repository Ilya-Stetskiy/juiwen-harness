"""Materialize a seeded sample of the unified datasets into task directories (lab tool).

The run directory gets task.json WITHOUT `expected` (like the final set); answers and metadata go to
<out>_grading.json so the harness provably never reads `expected`.

usage: python make_taskset.py <unified_dir> <out_dir> [seed]
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

# source -> number of tasks
PLAN = {
    "math/gsm8k": 15, "math/svamp": 5, "math/asdiv": 5, "math/math_algebra": 5,
    "math/math_number_theory": 5, "math/math500": 5,
    "coding/codegen_fix": 30, "coding/humanevalfix": 10, "coding/mbpp": 5,
    "data_analysis/wikitablequestions": 25, "data_analysis/tabfact": 10,
    "search/hotpotqa": 15, "search/2wikimultihopqa": 10, "search/musique_full": 20, "search/squad_v2": 10,
}


def sample_lines(path: Path, n: int, rng: random.Random) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rng.shuffle(rows)
    return rows[:n]


def main(unified: str, out: str, seed: int = 7) -> None:
    rng = random.Random(seed)
    out_dir = Path(out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    grading: dict[str, dict] = {}
    for src, n in PLAN.items():
        for r in sample_lines(Path(unified) / f"{src}.jsonl", n, rng):
            d = out_dir / r["task_id"]
            d.mkdir()
            task = {k: r[k] for k in ("task_id", "task_type", "instruction", "reward", "check") if k in r}
            (d / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
            files = r.get("files") or {}
            if files:
                for rel, content in files.items():
                    p = d / "workspace" / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content, encoding="utf-8")
            grading[r["task_id"]] = {
                "task_type": r["task_type"], "check": r["check"], "reward": r["reward"],
                "expected": r.get("expected"), "source": src,
                "unanswerable": r["check"] == "unanswerable",
            }
    Path(str(out_dir) + "_grading.json").write_text(json.dumps(grading, ensure_ascii=False), encoding="utf-8")
    print(f"{len(grading)} tasks -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 7)
