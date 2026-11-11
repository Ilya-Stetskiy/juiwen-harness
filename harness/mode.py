"""Run modes: HARNESS_RUN_MODE=legacy|accuracy|economy|auto. Default legacy = nothing applied, behaviour as before.

(HARNESS_MODE is taken: it is the executor's pipeline|agent switch.)

    accuracy  profiles/mode_accuracy.yaml   most points within the budget; tokens may go to verification
    economy   profiles/mode_economy.yaml    the same points for fewer tokens (the organizers' tie-break: fewer tokens)
    adaptive  decides continuously during the run (harness/adaptive.py): the switches common to both profiles are applied at
              the start, the ones that differ are switched between the accuracy and the economy values while the run goes on
    auto      before the first task the gbm cost model sums the estimated cost of all tasks and compares with BUDGET
              (profiles/mode_auto.yaml): p50 sum <= 70% of the budget and p90 sum <= 130% -> accuracy, else economy;
              no estimate (no model file, no tasks) -> economy. The boundary goes to economy.

A profile is a set of env switches applied with setdefault: a variable already set in the environment wins over the profile.
"""
from __future__ import annotations

import os
from pathlib import Path

PROFILES = Path(__file__).resolve().parent.parent / "profiles"
MODES = ("legacy", "accuracy", "economy", "auto", "adaptive")


def requested() -> str:
    m = os.environ.get("HARNESS_RUN_MODE", "legacy").strip().lower()
    return m if m in MODES else "legacy"


def _yaml(name: str) -> dict:
    import yaml
    try:
        return yaml.safe_load((PROFILES / name).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}


def load_profile(mode: str) -> dict[str, str]:
    return {k: str(v) for k, v in (_yaml(f"mode_{mode}.yaml").get("env") or {}).items()}


def common_keys() -> dict[str, str]:
    """Switches with the same value in both profiles (applied once at the start of the adaptive mode)."""
    a, e = load_profile("accuracy"), load_profile("economy")
    return {k: v for k, v in e.items() if a.get(k) == v}


def switchable_keys() -> list[str]:
    """Switches that differ between the profiles (a key missing in one profile means: the default / unset there)."""
    a, e = load_profile("accuracy"), load_profile("economy")
    return sorted(k for k in set(a) | set(e) if a.get(k) != e.get(k))


def estimate(tasks, model: str | None = None) -> dict | None:
    """Summed gbm estimate over the tasks: {p10, p50, p90, n} or None when the cost model is unavailable."""
    from . import costmodel
    if costmodel.load() is None or not tasks:
        return None
    from .predictor import GbmPredictor
    cfg = _yaml("mode_auto.yaml")
    model = model or cfg.get("model", "qwen3.6-35b-a3b-noreason")
    g, tot = GbmPredictor(), {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    from .sorter import expected_basis   # R-S3: with HARNESS_COST_BASIS=mean the sum is the expected cost, not the (2x too low) sum of medians
    for t in tasks:
        p = expected_basis(g.predict(t, model))
        tot["p10"] += p.tokens_p10
        tot["p50"] += p.tokens_p50
        tot["p90"] += p.tokens_p90
    tot["n"] = len(tasks)
    return tot


def decide(tasks, budget: float, est: dict | None = None) -> tuple[str, dict]:
    """auto: (mode, info). Unknown estimate or the boundary -> economy."""
    cfg = _yaml("mode_auto.yaml")
    r50, r90 = float(cfg.get("accuracy_if_p50_ratio_at_most", 0.7)), float(cfg.get("accuracy_if_p90_ratio_at_most", 1.3))
    est = est if est is not None else estimate(tasks)
    if est is None or budget <= 0:
        return "economy", {"reason": "no estimate", "budget": budget}
    q50, q90 = est["p50"] / budget, est["p90"] / budget
    info = {"p10": round(est["p10"]), "p50": round(est["p50"]), "p90": round(est["p90"]), "budget": budget,
            "ratio_p50": round(q50, 3), "ratio_p90": round(q90, 3), "max_p50": r50, "max_p90": r90}
    mode = "accuracy" if (q50 < r50 and q90 < r90) else "economy"
    info["reason"] = "estimate below thresholds" if mode == "accuracy" else "estimate at or above a threshold"
    return mode, info


def apply(tasks, budget: float) -> dict:
    """Choose the mode and put its profile into os.environ (setdefault). Returns the record for the journal."""
    req = requested()
    rec: dict = {"requested": req, "mode": "legacy", "applied": {}, "overridden": []}
    if req == "legacy":
        return rec
    if req == "adaptive":
        rec["mode"] = "adaptive"
        for k, v in common_keys().items():
            if k in os.environ:
                rec["overridden"].append(k)
            else:
                os.environ[k] = v
                rec["applied"][k] = v
        os.environ["HARNESS_RUN_MODE_EFFECTIVE"] = "adaptive"
        return rec
    mode, info = (req, {}) if req != "auto" else decide(tasks, budget)
    rec["mode"], rec["auto"] = mode, info
    for k, v in load_profile(mode).items():
        if k in os.environ:
            rec["overridden"].append(k)
        else:
            os.environ[k] = v
            rec["applied"][k] = v
    os.environ["HARNESS_RUN_MODE_EFFECTIVE"] = mode
    return rec
