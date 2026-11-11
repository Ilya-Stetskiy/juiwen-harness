"""Batching (S11): several tasks of one type in ONE model request. Skeleton with one variant (math); OFF by default.

    HARNESS_BATCH_SIZE=1|2|4|8          1 (default): no batching. n > 1: the worker takes up to n tasks of the same type
                                        from the scheduler and sends them in a single request.
    HARNESS_BATCH_TYPES=math            task types that may be batched (only types without per-task documents make sense).
    HARNESS_BATCH_FALLBACK=single|skip  a task whose line is missing or unparsable: run it alone (default) or give up.
    HARNESS_BATCH_PROMPT=plain|compute  plain: "print the answers" (the model tends to print guesses, accuracy drops with
                                        the batch size, see the S11 report); compute: one numbered step-by-step computation
                                        per problem inside the program, nothing hard-coded.

The model writes ONE program that prints N lines, line i = the bare answer of problem i; the harness runs it and
splits the lines. One failing problem does not lose the others: only the missing lines are run alone. The tokens of a
batch are booked on the first task in the ledger and shared equally between the tasks when the scheduler is told.
Why: the gateway allows 60 requests per minute; fewer requests also pay the system prompt once.
"""
from __future__ import annotations

import os
import time

import asyncio

from . import guard
from .answers import normalize, write_answer
from .llm import CallContext, ask
from .pipelines import clean_instruction, extract_code, run_python

SYSTEM = ("Solve all the problems below. Write ONE Python program that prints exactly {n} lines: line i is the bare final "
          "answer of problem i (a number or a short value, no words). Return only the program in one ```python block.")


SYSTEM_COMPUTE = ("Solve all the problems below. Write ONE Python program with a separate section per problem: for each "
                  "problem compute the answer step by step from the given numbers (name the intermediate quantities in "
                  "variables; never hard-code a result) and print it. The program must print exactly {n} lines: line i is the "
                  "bare final answer of problem i (a number or a short value, no words). Return only the program in one "
                  "```python block.")


def per_problem_tokens() -> int:
    """Reply budget per problem: the step-by-step program of the compute prompt needs about 350, the plain one 300 (batch_math_compute: 8 problems at 300 each were cut)."""
    return 600 if os.environ.get("HARNESS_BATCH_PROMPT", "plain").strip() == "compute" else 300


def system_prompt(n: int) -> str:
    return (SYSTEM_COMPUTE if os.environ.get("HARNESS_BATCH_PROMPT", "plain").strip() == "compute" else SYSTEM).format(n=n)


def batch_size() -> int:
    try:
        return max(1, min(8, int(os.environ.get("HARNESS_BATCH_SIZE", "1"))))
    except ValueError:
        return 1


def batch_types() -> set[str]:
    return {t.strip() for t in os.environ.get("HARNESS_BATCH_TYPES", "math").split(",") if t.strip()}


async def solve_batch(ctx: CallContext, tasks: list) -> dict[str, str | None]:
    """task_id -> normalised answer or None (missing line). Two attempts, the second shows the program error."""
    n = len(tasks)
    user = "\n\n".join(f"Problem {i + 1}: {clean_instruction(t.instruction)}" for i, t in enumerate(tasks))
    prev = None
    for _ in range(2):
        prompt = user if prev is None else (f"{user}\n\nYour previous program printed {prev} instead of {n} lines. Fix it.")
        reply = await ask(ctx, system_prompt(n), prompt, per_problem_tokens() * n + 300)
        import tempfile
        with tempfile.TemporaryDirectory(prefix="b_") as tmp:
            rc, out = await run_python(extract_code(reply), tmp)
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()] if rc == 0 else []
        if len(lines) == n:
            return {t.task_id: normalize(t.check, ln) for t, ln in zip(tasks, lines)}
        prev = f"{len(lines)} lines" if rc == 0 else "an error"
    return {t.task_id: None for t in tasks}


def plan(task, pred, model: str, sched, ledger) -> list:
    """The batcher's decision: one task or a group. Input: the task just taken from the sorter, the queue behind it
    (`sched.take_more`), the state (tokens spent). Output: list of (task, prediction), length 1 = no batching.
    Trivial variant: HARNESS_BATCH_SIZE for the types in HARNESS_BATCH_TYPES, otherwise 1. Not used yet: request-per-minute
    state, heaviness from the predictor, remaining budget (the planner is the place for them)."""
    size = batch_size()
    if size <= 1 or task.task_type not in batch_types() or not hasattr(sched, "take_more"):
        return [(task, pred)]
    return [(task, pred)] + sched.take_more(task.task_type, model, size - 1, ledger.spent())


async def maybe_batch(task, model, pred, sched, cfg, ledger, journal, state, wid: int) -> bool:
    """Called by the worker right after `sched.next`. Returns True when the task was handled here as part of a batch."""
    group = plan(task, pred, model, sched, ledger)
    if len(group) <= 1:
        return False
    from .executor import run_task

    ids = [t.task_id for t, _ in group]
    before = ledger.task_cost(task.task_id)
    for t, p in group:
        journal.event("task_start", task=t.task_id, type=t.task_type, model=model, est_p50=round(p.tokens_p50, 1),
                      worker=wid, batch=len(group))
    journal.event("batch", tasks=ids, model=model, size=len(group))
    t0 = time.time()
    ctx = CallContext(task, model, pred, cfg, ledger, journal)
    try:
        coro_b = solve_batch(ctx, [t for t, _ in group])
        answers = await (asyncio.wait_for(coro_b, cfg.max_task_seconds) if cfg.limits_on else coro_b)   # R-X1: the batch has a time cap too
    except Exception as exc:  # noqa: BLE001 - a broken batch call must not lose the tasks: all go to the single path (also a timeout)
        journal.event("batch_error", tasks=ids, error=type(exc).__name__, detail=str(exc)[:200])
        answers = {i: None for i in ids}
    share = (ledger.task_cost(task.task_id) - before) / len(group)
    for t, p in group:
        ans = answers.get(t.task_id)
        status = "solved" if ans else "failed"
        cost = share
        if ans is None and os.environ.get("HARNESS_BATCH_FALLBACK", "single") == "single":
            b2 = ledger.task_cost(t.task_id)
            try:
                coro_s = run_task(t, model, p, cfg, ledger, journal)
                res = await (asyncio.wait_for(coro_s, cfg.max_task_seconds) if cfg.limits_on else coro_s)
                status, ans = res.status, res.answer
            except Exception as exc:  # noqa: BLE001 (a timeout too)
                journal.event("task_error", task=t.task_id, error=type(exc).__name__, detail=str(exc)[:200])
            cost += ledger.task_cost(t.task_id) - b2
        if status == "solved" and ans:
            write_answer(t.dir, guard.scrub_answer(ans, t.task_id, journal))   # R-X1: the same rail as the single path
            state["solved"] += 1
        sched.report(t, model, cost)
        if status == "deferred":
            state["deferred"].append((t, model, p))
            journal.event("task_deferred", task=t.task_id, cost=round(cost, 1), reward=t.reward)
            continue
        journal.event("task_end", task=t.task_id, status=status, cost=round(cost, 1),
                      total=round(ledger.spent(), 1), reward=t.reward, batch=len(group))
    return True
