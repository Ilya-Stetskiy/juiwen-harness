"""Censor, verification side (S10): estimate P(answer is correct) after a task is answered. Switchable, OFF by default.

    HARNESS_VERIFY=off|free|judge|other|other_judge   other: after the answer the same pipeline runs with ANOTHER model
                                       (HARNESS_VERIFY_OTHER_MODEL; default qwen3.8-27b-noreason if the main model is qwen3.6-*,
                                       else qwen3.6-35b-a3b-noreason) on types HARNESS_VERIFY_OTHER_TYPES (default
                                       search,data_analysis,math); agreement (`agree_other`) goes to the verifier level `other`.
                                       other_judge: the same plus the LLM judge (on HARNESS_VERIFY_JUDGE_TYPES).
                                       Other types fall back to free / judge. The second answer is kept in
                                       ctx.extra["other_answer"] for executor._verify_loop (HARNESS_DISAGREE).
    (off|free|judge text:)             off (default): nothing. free: no tokens, signals of the answer and of the program
                                       that produced it. judge: free + one short LLM call ("is this answer right?"),
                                       its tokens go through the ledger and the journal like any other call.
    HARNESS_VERIFY_ACTION=log|retry|recheck    recheck: see executor._verify_loop (stronger model, three verdicts). log (default): only the `verify` event; retry: when p < HARNESS_VERIFY_MIN
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
    return m if m in ("off", "free", "judge", "other", "other_judge") else "off"


def other_types() -> set[str]:
    raw = os.environ.get("HARNESS_VERIFY_OTHER_TYPES", "search,data_analysis,math")
    return {t.strip() for t in raw.split(",") if t.strip()}


def other_model_for(model: str) -> str:
    """Model of the second opinion: HARNESS_VERIFY_OTHER_MODEL or the other family of the main model."""
    m = os.environ.get("HARNESS_VERIFY_OTHER_MODEL", "").strip()
    if m:
        return m
    return "qwen3.8-27b-noreason" if model.startswith("qwen3.6") else "qwen3.6-35b-a3b-noreason"


def _norm(task, a: str | None) -> str:
    from .answers import normalize
    return normalize(task.check, a or "").strip().lower()


async def run_other(ctx, task, answer: str) -> dict:
    """Run the task's pipeline with another model. Returns signals for the verifier; the tokens go through the ledger."""
    from .llm import CallContext
    from .pipelines import PIPELINES

    m2 = other_model_for(ctx.model)
    pipeline = PIPELINES.get(task.task_type)
    if pipeline is None:
        return {}
    c2 = CallContext(task, m2, ctx.pred, ctx.cfg, ctx.ledger, ctx.journal)
    c2.extra["recheck"] = True
    try:
        res = await pipeline(c2)
    except Exception:  # noqa: BLE001 - CensorStop / BudgetStop / transport: no second opinion
        res = None
    a2 = res[1] if res is not None and res[0] == "solved" else None
    if a2 is None:
        return {"other_answered": False, "other_model": m2}
    return {"other_answered": True, "agree_other": _norm(task, a2) == _norm(task, answer), "other_answer": a2, "other_model": m2,
            "other_extra": dict(c2.extra)}


def action() -> str:
    a = os.environ.get("HARNESS_VERIFY_ACTION", "log").strip().lower()
    return a if a in ("retry", "recheck") else "log"


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


def judge_types() -> set[str]:
    """HARNESS_VERIFY_JUDGE_TYPES (default math,data_analysis,coding): task types that get the LLM judge in judge mode;
    the other types use the free level only (the verifier session found the judge useless on search)."""
    raw = os.environ.get("HARNESS_VERIFY_JUDGE_TYPES", "math,data_analysis,coding")
    return {t.strip() for t in raw.split(",") if t.strip()}


def verdict(p: float) -> str:
    """Three verdicts: accept | recheck | unsolvable. HARNESS_VERIFY_ACCEPT (default = HARNESS_VERIFY_MIN, 0.5) is the
    lowest p that is accepted; HARNESS_VERIFY_GIVEUP (default 0 = never) is the p below which no more tokens are spent."""
    try:
        giveup = float(os.environ.get("HARNESS_VERIFY_GIVEUP", "0") or 0)
    except ValueError:
        giveup = 0.0
    try:
        accept = float(os.environ.get("HARNESS_VERIFY_ACCEPT", "") or min_p())
    except ValueError:
        accept = min_p()
    return "accept" if p >= accept else ("unsolvable" if p < giveup else "recheck")


def free_verdict(task, answer: str, context: dict) -> dict | None:
    """p of an answer from the verifier model with the given context (agree_other, other_answered, runs, ...), no tokens."""
    ver = _verifier()
    if ver is None or not answer:
        return None
    v = ver.verify_detailed(task, answer, context)
    return {"p": float(v.p), "level": v.level}


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
    want_other = m in ("other", "other_judge") and task.task_type in other_types()
    if m in ("other", "other_judge"):
        m = "judge" if (m == "other_judge") else "free"
    if m == "judge" and task.task_type not in judge_types():   # e.g. search: the judge adds nothing there (AUC 0.5-0.6)
        m = "free"
    other = {}
    if want_other:
        other = await run_other(ctx, task, answer)
        for k in ("agree_other", "other_answered"):
            if k in other:
                context[k] = other[k]
        if "other_answer" in other:
            ctx.extra["other_answer"] = other["other_answer"]
            ctx.extra["other_model"] = other["other_model"]
        if other.get("other_answered") is not None:
            ctx.extra["agree_other"] = other.get("agree_other")
            ctx.extra["other_answered"] = other["other_answered"]
        if "agree_other" in context or "other_answered" in context:
            v0 = ver.verify_detailed(task, answer, context) if ver is not None else None
            if v0 is not None:
                p, level = float(v0.p), v0.level
    spec = ver.judge_prompt(task, answer, context) if (m == "judge" and ver is not None) else None
    other_cost = ctx.ledger.task_cost(task.task_id) - spent0
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
    out = {"p": round(p, 4), "p_free": round(p_free, 4), "level": level, "judge": judge, "tokens": round(tokens, 1),
           "verdict": verdict(p)}
    if want_other:
        out.update(agree_other=other.get("agree_other"), other_answered=other.get("other_answered"),
                   other_model=other.get("other_model"), other_tokens=round(other_cost, 1))
    ctx.journal.event("verify", task=task.task_id, model=ctx.model, **out)
    return out
