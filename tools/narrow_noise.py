#!/usr/bin/env python3
"""Noise floors of the narrow sets (nrw_*): sd of points between repeats of the unchanged baseline, from experiments/results.

Reads results/noise_local_<set>/cells.jsonl (local Qwen, PROVISIONAL) and results/noise_<set>/cells.jsonl (gateway, the real one),
writes experiments/sets/narrow/noise_floor.json:
  per set: local {sd_points, mean_points, repeats, flip_rate, ...}, gateway {...} when measured, and `threshold_points` with `status`:
    gateway   a gateway measurement with >= 3 repeats exists (the real threshold)
    provisional-local-scaled   local sd times the median gateway/local ratio of the validation sets (needs >= 3 sets with both)
    provisional-local          local sd as it is (no gateway validation yet)
The sd of 3 repeats is rough (about +-50%): a threshold is a first guess, never an acceptance rule on its own.

usage: python narrow_noise.py [--results DIR] [--out FILE]      (on the host: DIR defaults to ~/harness-lab/experiments/results)
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path


def measure(cells_file: Path) -> dict | None:
    if not cells_file.exists():
        return None
    cells = [json.loads(line) for line in open(cells_file, encoding="utf-8") if line.strip()]
    good = [c for c in cells if not c.get("infra")]
    by_rep: dict[int, float] = collections.defaultdict(float)
    ok: dict[int, int] = collections.defaultdict(int)
    verdict: dict[str, set] = collections.defaultdict(set)
    tok: dict[int, float] = collections.defaultdict(float)
    for c in good:
        by_rep[c["repeat"]] += c["reward"] * c["correct"]
        ok[c["repeat"]] += int(bool(c["correct"]))
        tok[c["repeat"]] += c.get("hybrid_cost") or 0
        verdict[c["task_id"]].add(bool(c["correct"]))
    if len(by_rep) < 2:
        return {"repeats": len(by_rep), "sd_points": None, "note": "fewer than 2 complete repeats"}
    pts = list(by_rep.values())
    total = sum(c["reward"] for c in good) / len(by_rep)
    flip = sum(len(v) > 1 for v in verdict.values()) / max(1, len(verdict))
    return {"repeats": len(by_rep), "n_tasks": len(verdict), "points_per_repeat": [round(p, 1) for p in pts], "mean_points": round(statistics.fmean(pts), 1),
            "sd_points": round(statistics.stdev(pts), 2), "max_points": round(total, 1), "sd_pct_of_max": round(100 * statistics.stdev(pts) / total, 2) if total else None,
            "correct_per_repeat": [ok[r] for r in sorted(ok)], "tokens_per_repeat": [round(tok[r]) for r in sorted(tok)], "flip_rate": round(flip, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path.home() / "harness-lab" / "experiments" / "results"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "experiments" / "sets" / "narrow" / "noise_floor.json"))
    a = ap.parse_args()
    res = Path(a.results)
    sets = sorted({p.name.split("noise_local_", 1)[-1] if p.name.startswith("noise_local_") else p.name.split("noise_", 1)[-1]
                   for p in res.glob("noise_*nrw_*") if p.is_dir()})
    out: dict = {}
    for s in sets:
        out[s] = {"local": measure(res / f"noise_local_{s}" / "cells.jsonl"), "gateway": measure(res / f"noise_{s}" / "cells.jsonl")}
    ratios = [v["gateway"]["sd_points"] / v["local"]["sd_points"] for v in out.values()
              if v["gateway"] and v["local"] and v["gateway"].get("sd_points") and v["local"].get("sd_points")]
    ratio = round(statistics.median(ratios), 2) if len(ratios) >= 3 else None
    for v in out.values():
        g, l = v["gateway"], v["local"]
        if g and g.get("sd_points") is not None and g["repeats"] >= 3:
            v["threshold_points"], v["status"] = g["sd_points"], "gateway"
        elif l and l.get("sd_points") is not None and ratio:
            v["threshold_points"], v["status"] = round(l["sd_points"] * ratio, 2), "provisional-local-scaled"
        elif l and l.get("sd_points") is not None:
            v["threshold_points"], v["status"] = l["sd_points"], "provisional-local"
        else:
            v["threshold_points"], v["status"] = None, "not measured"
    doc = {"updated": time.strftime("%Y-%m-%d %H:%M"), "gateway_over_local_sd_ratio": ratio, "ratio_sets": len(ratios),
           "note": "sd of points (sum of reward x correct) between 3 repeats of the unchanged baseline (as_is: model auto, fallback on). "
                   "local = Qwen3.6-35B-A3B Q4 behind the shim (all four model ids map to it): PROVISIONAL, an approximation of the gateway sd.",
           "sets": out}
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    for s, v in out.items():
        print(f"{s:34} {v['status']:26} threshold {v['threshold_points']}  local {v['local'] and v['local'].get('sd_points')}  gateway {v['gateway'] and v['gateway'].get('sd_points')}")


if __name__ == "__main__":
    main()
