"""Censor, executor side: decides whether a running LLM executor may continue. Skeleton; OFF by default.

    HARNESS_CENSOR=off|log|enforce   off (default): nothing. log: write `censor` events with the decision that WOULD be
                                     taken, change nothing (the lab then counts saved tokens and wrongly cut tasks).
                                     enforce: the decision is carried out.
    HARNESS_CENSOR_BASIS=fixed|pred  fixed: token cap per task type (below, override with HARNESS_CENSOR_TOKENS=
                                     "math=1500,search=1500"); pred: HARNESS_CENSOR_FACTOR (default 3) x the predictor's p90.
    HARNESS_CENSOR_SECONDS=120       time cap per task (all types)

Interface: `decide(task, model, spent, elapsed, pred) -> Decision(action, reason, cap)`;
action is "continue" | "stop" | "retry" (stop and repeat differently; produced by the loop detector, harness/detectors.py,
which is part of this block: HARNESS_LOOP). Input is the executor state seen before the next call: tokens already spent on
the task (ledger), seconds since the task started. The reasoning stream is not visible (replies arrive whole), so a call in
flight cannot be cut; the cap works between calls and through max_tokens.
Metrics (lab report "Censor"): tasks the censor would have stopped, how many of them were solved anyway (stopped for nothing),
tokens spent above the cap (what stopping would have saved).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Token caps (ledger cost) per task type, about 3x the measured median cost of a pipeline run (predictor._MEASURED).
FIXED_CAP = {"math": 1500.0, "data_analysis": 2000.0, "search": 1200.0, "coding": 5000.0}


class CensorStop(Exception):
    """Raised inside a pipeline when the censor stops the executor (enforce mode)."""


@dataclass
class Decision:
    action: str          # continue | stop | retry
    reason: str = ""
    cap: float = 0.0


def mode() -> str:
    m = os.environ.get("HARNESS_CENSOR", "off").strip().lower()
    return m if m in ("off", "log", "enforce") else "off"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def token_cap(task, model: str, pred) -> float:
    if os.environ.get("HARNESS_CENSOR_BASIS", "fixed").strip().lower() == "pred" and pred is not None:
        return float(pred.tokens_p90) * _env_float("HARNESS_CENSOR_FACTOR", 3.0)
    caps = dict(FIXED_CAP)
    for part in os.environ.get("HARNESS_CENSOR_TOKENS", "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                caps[k.strip()] = float(v)
            except ValueError:
                pass
    cap = caps.get(task.task_type, 3000.0)
    return cap * (3.0 if not model.endswith("-noreason") else 1.0)   # a reasoning model thinks inside its tokens


def decide(task, model: str, spent: float, elapsed: float, pred=None) -> Decision:
    cap = token_cap(task, model, pred)
    if spent > cap:
        return Decision("stop", f"tokens {spent:.0f} > cap {cap:.0f}", cap)
    if elapsed > _env_float("HARNESS_CENSOR_SECONDS", 120.0):
        return Decision("stop", f"seconds {elapsed:.0f}", cap)
    return Decision("continue", "", cap)


def check(ctx) -> None:
    """Before each call of an executor: log / enforce the decision for the task of `ctx`."""
    import time

    m = mode()
    if m == "off":
        return
    t0 = ctx.extra.setdefault("t0", time.time())
    spent = ctx.ledger.task_cost(ctx.task.task_id)
    d = decide(ctx.task, ctx.model, spent, time.time() - t0, ctx.pred)
    if d.action == "continue":
        return
    if not ctx.extra.get("censor_logged"):   # one event per task, the first moment the cap is crossed
        ctx.extra["censor_logged"] = True
        ctx.journal.event("censor", task=ctx.task.task_id, model=ctx.model, action=d.action, reason=d.reason,
                          cap=round(d.cap, 1), spent=round(spent, 1), enforced=(m == "enforce"))
    if m == "enforce":
        raise CensorStop(d.reason)
