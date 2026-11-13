"""HARNESS_SKIP_PSUCC (default 0 = off): before a task is dispatched, if the gbm p_success of the task is below the threshold
and the summed gbm p50 cost of the remaining queue exceeds the remaining budget, the task is skipped (event skip_psucc) and
the zero-token answer of the HARNESS_GUESS_UNSOLVABLE=free path is written for it (no LLM call)."""
from __future__ import annotations

import os as _os


def threshold() -> float:
    try:
        return float(_os.environ.get("HARNESS_SKIP_PSUCC", "0") or 0)
    except ValueError:
        return 0.0


def should_skip(sched, pred, budget: float, spent: float) -> dict | None:
    thr = threshold()
    if thr <= 0 or pred is None:
        return None
    p = getattr(pred, "p_success", None)
    if p is None:
        return None
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    if p >= thr:
        return None
    try:
        q = sched.queue_estimate()
        queue = float(q.get("p50", 0.0)) + float(getattr(pred, "tokens_p50", 0.0) or 0.0)
    except Exception:  # noqa: BLE001 - no estimate: no skip
        return None
    remaining = float(budget) - float(spent)
    if queue <= remaining:
        return None
    return {"p_success": round(p, 4), "threshold": thr, "queue_p50": round(queue, 1), "remaining": round(remaining, 1)}


async def free_answer(task, model, pred, cfg, ledger, journal):
    """The answer of the GUESS_UNSOLVABLE=free path (zero tokens)."""
    from . import guess
    key = "HARNESS_GUESS_UNSOLVABLE"
    old = _os.environ.get(key)
    _os.environ[key] = "free"
    try:
        return await guess.guess(task, model, pred, cfg, ledger, journal)
    finally:
        if old is None:
            _os.environ.pop(key, None)
        else:
            _os.environ[key] = old
