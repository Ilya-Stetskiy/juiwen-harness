"""Hard shared budget (H1): reserve tokens before every call, stop new calls near the limit. Skeleton; OFF by default.

    HARNESS_BUDGET_HARD=off|on        on: every LLM call first reserves its worst-case cost; a call that would push
                                      spent + reserved above the stop line is not made (raises BudgetStop).
    HARNESS_BUDGET_STOP=0.97          stop line as a share of the budget (0.97-0.98)
    HARNESS_BUDGET_RECHECK=0.10       share of the budget kept for rechecks: first attempts stop at (stop - recheck)
    HARNESS_BUDGET_FILE=<path>        optional: shared counter file (flock) for several parallel sessions (open question H7:
                                      how the organisers share the budget between sessions)

Interface: `reserve(ctx, est) -> token`, `release(token, actual)`. The estimate of a call is the prompt (characters / 4 / 10)
plus max_tokens (worst case); the actual cost comes from the proxy usage and goes through the ledger as before. The ledger
(`harness/ledger.py`) stays the single place where spending is counted; this module only gates the start of a call.
"""
from __future__ import annotations

import os
import threading


class BudgetStop(Exception):
    """No new call may start: the budget (stop line) is used up."""


_lock = threading.Lock()
_reserved = 0.0


def enabled() -> bool:
    return os.environ.get("HARNESS_BUDGET_HARD", "off").strip().lower() == "on"


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def reserve(ctx, prompt_chars: int, max_tokens: int, recheck: bool = False, prompt_tokens: int | None = None) -> float:
    """Reserve the worst-case cost of one call; BudgetStop when it does not fit."""
    global _reserved
    if not enabled():
        return 0.0
    est = (prompt_tokens if prompt_tokens is not None else prompt_chars / 4.0) / 10.0 + max_tokens   # exact count with HARNESS_TOKENIZER=on
    stop = _f("HARNESS_BUDGET_STOP", 0.97)
    line = ctx.cfg.budget * (stop if recheck else stop - _f("HARNESS_BUDGET_RECHECK", 0.0))
    with _lock:
        spent = _shared_spent() if os.environ.get("HARNESS_BUDGET_FILE") else ctx.ledger.spent()
        if spent + _reserved + est > line:
            ctx.journal.event("budget_stop", task=ctx.task.task_id, spent=round(spent, 1), reserved=round(_reserved, 1),
                              est=round(est, 1), line=round(line, 1), recheck=recheck)
            raise BudgetStop(f"spent {spent:.0f} + reserved {_reserved:.0f} + {est:.0f} > {line:.0f}")
        _reserved += est
    return est


def release(est: float, actual: float) -> None:
    global _reserved
    if not est and not actual:
        return
    with _lock:
        _reserved = max(_reserved - est, 0.0)
        path = os.environ.get("HARNESS_BUDGET_FILE")
        if path:
            _shared_add(path, actual)


def _shared_spent() -> float:
    try:
        return float(open(os.environ["HARNESS_BUDGET_FILE"]).read().strip() or 0)
    except (OSError, ValueError):
        return 0.0


def _shared_add(path: str, amount: float) -> None:
    try:
        import fcntl
        with open(path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            cur = float(f.read().strip() or 0)
            f.seek(0)
            f.truncate()
            f.write(str(cur + amount))
            fcntl.flock(f, fcntl.LOCK_UN)
    except (ImportError, OSError, ValueError):
        pass
