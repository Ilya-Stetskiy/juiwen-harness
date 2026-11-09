"""Best guess for a task that was declared unsolvable (no answer exists). Skeleton; OFF by default.

    HARNESS_GUESS_UNSOLVABLE=off|free|llm
        off (default): no answer is written (the old behaviour).
        free: no tokens: coding -> "DONE" (the answer file of a coding task is only a marker, the tests judge the repository),
              other types get nothing.
        llm:  free for coding, one short LLM call "best guess" for the others; its tokens go through the ledger/journal.
The rules give no penalty for a wrong answer and no bonus for no answer, so a guess can only help (HARNESS_SUBMIT_UNSOLVABLE).
"""
from __future__ import annotations

import os

SYSTEM = ("The task below may be unanswerable from the data given. Even so, give your single best guess. "
          "End with one line: FINAL: <bare answer>")


def mode() -> str:
    m = os.environ.get("HARNESS_GUESS_UNSOLVABLE", "off").strip().lower()
    return m if m in ("off", "free", "llm") else "off"


async def guess(task, model: str, pred, cfg, ledger, journal) -> str | None:
    m = mode()
    if m == "off":
        return None
    if task.task_type == "coding":
        ans = "DONE"
    elif m == "llm":
        from .answers import extract_final, normalize
        from .llm import CallContext, ask
        from .pipelines import clean_instruction
        from .predictor import Prediction
        ctx = CallContext(task, model, pred or Prediction(500.0, 1500.0, 0.5), cfg, ledger, journal)
        try:
            reply = await ask(ctx, SYSTEM, clean_instruction(task.instruction), 300)
        except Exception:  # noqa: BLE001 - budget stop / censor stop / network: no guess
            return None
        final = extract_final(reply, fallback=True)
        ans = normalize(task.check, final) if final else None
    else:
        ans = None
    journal.event("guess", task=task.task_id, mode=m, answered=bool(ans))
    return ans
