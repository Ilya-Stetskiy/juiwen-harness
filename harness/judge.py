"""Censor, verification side (S10): estimate P(answer is correct) after a task is answered. Switchable, OFF by default.

    HARNESS_VERIFY=off|free|judge      off (default): nothing. free: no tokens, signals of the answer and of the program
                                       that produced it. judge: free + one short LLM call ("is this answer right?"),
                                       its tokens go through the ledger and the journal like any other call.
    HARNESS_VERIFY_ACTION=log|retry    log (default): only the `verify` event; retry: when p < HARNESS_VERIFY_MIN
                                       (default 0.5) the executor runs the pipeline once more and keeps the answer with
                                       the higher p (the repeat is paid from the budget).

Interface: `await verify_answer(ctx, task, answer) -> dict | None` with p, level, judge (True/False/None), tokens.
The statistical model comes from harness/verifier.py (session answer-verifier, uses profiles/verifier_model.json); if
that module or its model is missing the trivial placeholder below is used, so the block always exists.
`ctx.extra["runs"]` = [(code, returncode, stdout)] is filled by the scripted pipelines.
"""
from __future__ import annotations

import os
import re

JUDGE_SYSTEM = ("You check answers. Given a task and a proposed final answer, reply with exactly one word: "
                "YES if the answer is very likely correct, NO otherwise.")


def mode() -> str:
    m = os.environ.get("HARNESS_VERIFY", "off").strip().lower()
    return m if m in ("off", "free", "judge") else "off"


def action() -> str:
    return "retry" if os.environ.get("HARNESS_VERIFY_ACTION", "log").strip().lower() == "retry" else "log"


def min_p() -> float:
    try:
        return float(os.environ.get("HARNESS_VERIFY_MIN", "0.5"))
    except ValueError:
        return 0.5


def _placeholder(task, answer, context: dict):
    """Trivial verifier: an answer exists, has the right shape, the program ran cleanly."""
    runs = context.get("runs") or []
    ok = bool(answer) and bool(answer.strip())
    if runs:
        _, rc, out = runs[-1]
        ok = ok and rc == 0 and "nan" not in out.lower()
    if task.check == "exact_int":
        ok = ok and bool(re.fullmatch(r"-?\d+", answer.strip()))
    return (0.8 if ok else 0.2), "placeholder"


def free_p(task, answer, context: dict) -> tuple[float, str]:
    try:
        from .verifier import verify_detailed  # session answer-verifier (optional dependency)
        v = verify_detailed(task, answer, context)
        return float(v.p), v.level
    except Exception:  # noqa: BLE001 - module or model missing: the placeholder keeps the block alive
        return _placeholder(task, answer, context)


def _verifier():
    """harness/verifier.py of session answer-verifier, or None (module or model missing: the placeholder takes over)."""
    try:
        from . import verifier
        return verifier if verifier.load_model() else None
    except Exception:  # noqa: BLE001
        return None


async def verify_answer(ctx, task, answer: str | None) -> dict | None:
    m = mode()
    if m == "off" or not answer:
        return None
    context = dict(ctx.extra)
    judge = None
    spent0 = ctx.ledger.task_cost(task.task_id)
    p_free, level = free_p(task, answer, context)
    p = p_free
    ver = _verifier()
    spec = ver.judge_prompt(task, answer, context) if (m == "judge" and ver is not None) else None
    if m == "judge" and spec is not None:
        # the judge of the verifier session: its prompt and parser, its fitted model uses judge_yes (and the quote checks)
        from .llm import ask
        system, user, max_tokens = spec
        reply = await ask(ctx, system, user, max_tokens)
        context.update(ver.parse_judge(reply, task, answer, context))
        judge = bool(context.get("judge_yes"))
        v = ver.verify_detailed(task, answer, context)
        p, level = float(v.p), v.level
    elif m == "judge":
        from .llm import ask
        shown = answer if len(answer) <= 300 else answer[:300]
        reply = await ask(ctx, JUDGE_SYSTEM, f"Task:\n{task.instruction[:3000]}\n\nProposed final answer: {shown}\n\nCorrect?", 6)
        judge = reply.strip().upper().startswith("YES")
        context["judge_yes"] = judge
        p = (p_free + (0.9 if judge else 0.1)) / 2   # placeholder combination
        level += "+judge"
    tokens = ctx.ledger.task_cost(task.task_id) - spent0
    out = {"p": round(p, 4), "p_free": round(p_free, 4), "level": level, "judge": judge, "tokens": round(tokens, 1)}
    ctx.journal.event("verify", task=task.task_id, model=ctx.model, **out)
    return out
