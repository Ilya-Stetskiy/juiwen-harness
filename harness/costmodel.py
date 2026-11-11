"""Cost model: gradient-boosted trees exported to JSON (tools/predictor_export.py), evaluated with plain Python (no sklearn).

    model = load()                 # profiles/cost_model.json, None when missing
    model.predict(features) -> {"p10": .., "p50": .., "p90": ..}   tokens of the ledger formula (input/10 + output)

`features` follow model.features: type one-hot (math, search, data_analysis, coding), thinking, fam38, ws_bytes, n_files,
tok_instr, tok_prompt.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "profiles" / "cost_model.json"
_cache = None


class CostModel:
    def __init__(self, d: dict) -> None:
        self.features = d["features"]
        self.models = d["models"]
        self.meta = {k: d[k] for k in ("n_rows", "n_tasks") if k in d}

    @staticmethod
    def _tree(t: dict, x: list) -> float:
        i = 0
        fi, th, le, ri, va, lf = t["feature_idx"], t["num_threshold"], t["left"], t["right"], t["value"], t["is_leaf"]
        while not lf[i]:
            i = le[i] if x[fi[i]] <= th[i] else ri[i]
        return va[i]

    def _log(self, name: str, x: list) -> float:
        m = self.models[name]
        return m["baseline"] + sum(self._tree(t, x) for t in m["trees"])

    def predict(self, x: list) -> dict:
        lo, mid, hi = (self._log(n, x) for n in ("p10", "p50", "p90"))
        lo, hi = min(lo, mid), max(hi, mid)
        return {"p10": math.exp(lo), "p50": math.exp(mid), "p90": math.exp(hi)}


def load(path: str | None = None):
    global _cache
    if _cache is None or path:
        try:
            _cache = CostModel(json.loads(Path(path or _PATH).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            _cache = False
    return _cache or None
