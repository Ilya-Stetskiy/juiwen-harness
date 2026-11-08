"""Copy the tasks of given types out of a materialized task set (lab tool).

usage: python subset_taskset.py <src_dir> <dst_dir> <type> [<type> ...]
Also writes <dst_dir>_grading.json with the matching entries.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main(src: str, dst: str, types: list[str]) -> None:
    grading = json.load(open(src.rstrip("/") + "_grading.json", encoding="utf-8"))
    out = Path(dst)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    keep = {}
    for tid, g in grading.items():
        if g["task_type"] in types and (Path(src) / tid).is_dir():
            shutil.copytree(Path(src) / tid, out / tid)
            keep[tid] = g
    Path(str(out).rstrip("/") + "_grading.json").write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    print(f"{len(keep)} tasks -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
