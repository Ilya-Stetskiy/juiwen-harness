"""One plain LLM call (no tools) through an openjiuwen agent with the censor rail attached.

A tool-less DeepAgent adds almost no overhead (~50 input tokens), while the censor rail keeps
counting every call by the competition formula.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .config import Config
from .journal import Journal
from .ledger import TokenLedger
from .predictor import Prediction
from .tasks import TaskInfo


@dataclass
class CallContext:
    task: TaskInfo
    model: str
    pred: Prediction
    cfg: Config
    ledger: TokenLedger
    journal: Journal
    flagged: set = field(default_factory=set)
    n_calls: int = 0
    extra: dict = field(default_factory=dict)   # side information for the verifier, e.g. extra["runs"] (code, rc, stdout)


def _sampling(override: float | None = None) -> dict:
    """HARNESS_TEMPERATURE (optional): unset keeps the provider default; 0 makes runs repeatable."""
    if override is not None:
        return {"temperature": override}
    raw = os.environ.get("HARNESS_TEMPERATURE", "").strip()
    try:
        return {"temperature": float(raw)} if raw else {}
    except ValueError:
        return {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def effective_max_tokens(model: str, base: int) -> int:
    """A model with reasoning spends max_tokens on the thinking first; with the answer-sized limit of the
    scripted prompts (300-800) the reply is cut off before the answer line. Give such models extra room
    (HARNESS_THINK_EXTRA tokens, default 3000); -noreason models keep the tight limit."""
    if model.endswith("-noreason"):
        return base
    return base + _env_int("HARNESS_THINK_EXTRA", 3000)


async def ask(ctx: CallContext, system: str, user: str, max_tokens: int) -> str:
    """One call. With HARNESS_LOOP=log|retry the finished reply is checked for repetition / truncation (S13)."""
    from . import detectors

    mode = detectors.mode()
    if mode == "off":
        return await _ask_once(ctx, system, user, max_tokens)
    before = ctx.ledger.task_cost(ctx.task.task_id)
    reply = await _ask_once(ctx, system, user, max_tokens)
    cost = ctx.ledger.task_cost(ctx.task.task_id) - before
    score = detectors.loop_score(reply)
    flagged = detectors.is_loop(score)
    cut = detectors.truncated(reply, cost, effective_max_tokens(ctx.model, max_tokens))
    ctx.journal.event("loop_check", task=ctx.task.task_id, model=ctx.model, cost=round(cost, 1),
                      max_tokens=effective_max_tokens(ctx.model, max_tokens), loop=flagged, truncated=cut, **score)
    if mode == "retry" and (flagged or cut):
        ctx.journal.event("loop", task=ctx.task.task_id, model=ctx.model, wasted=round(cost, 1), action="retry")
        try:
            temp = float(os.environ.get("HARNESS_LOOP_TEMP", "0.6"))
        except ValueError:
            temp = 0.6
        reply = await _ask_once(ctx, system, user + "\n\nBe brief: answer directly and do not repeat yourself.",
                                max_tokens, temperature=temp)
    return reply


async def _ask_once(ctx: CallContext, system: str, user: str, max_tokens: int, temperature: float | None = None) -> str:
    """Gate (censor limits, hard budget reservation) around one raw call. Both gates are off by default."""
    from . import budget, limits, rpm

    limits.check(ctx)                                   # HARNESS_CENSOR=log|enforce
    if getattr(ctx.ledger, "capped", False):            # HARNESS_HARD_CAP=on: the API-reported cost has reached the budget
        raise budget.BudgetStop("hard cap reached")
    from . import tokcount
    ptoks = tokcount.chat_prompt_tokens(system, user, ctx.model) if tokcount.enabled() and budget.enabled() else None
    est = budget.reserve(ctx, len(system) + len(user), effective_max_tokens(ctx.model, max_tokens),
                         recheck=bool(ctx.extra.get("recheck")), prompt_tokens=ptoks)   # HARNESS_BUDGET_HARD=on
    before = ctx.ledger.task_cost(ctx.task.task_id)
    await rpm.acquire(ctx.journal, ctx.task.task_id)    # HARNESS_RPM_LIMIT>0 waits for a free request slot
    state = "cancelled"   # R-X6: "ok" after a reply, "failed" after an exception; anything else means the call was cancelled
    reply_len = 0
    try:
        out = await _call_with_retry(ctx, system, user, max_tokens, temperature, before)
        state, reply_len = "ok", len(out)
        return out
    except Exception as exc:  # noqa: BLE001
        state = "failed"
        if rpm.is_429(exc):
            ctx.journal.event("rate_429", task=ctx.task.task_id, model=ctx.model, in_last_minute=rpm.count_last_minute())
        raise
    finally:
        if state != "failed" and ctx.ledger.task_cost(ctx.task.task_id) == before and os.environ.get("HARNESS_LEDGER_ESTIMATE", "on").strip().lower() != "off":
            # R-X6: the proxy bills this call but the ledger saw no usage (usage missing, or the call was cancelled by the task
            # timeout): book an estimate so the budget is not silently under-counted. Reply length is unknown after a cancel:
            # assume half of max_tokens.
            in_tok = ptoks if ptoks is not None else (tokcount.chat_prompt_tokens(system, user, ctx.model) if tokcount.exact() else (len(system) + len(user)) / 4.0)
            out_tok = reply_len / 4.0 if state == "ok" else effective_max_tokens(ctx.model, max_tokens) / 2.0
            c = ctx.ledger.add(ctx.task.task_id, int(in_tok), int(out_tok))
            ctx.journal.event("ledger_estimate", task=ctx.task.task_id, model=ctx.model, reason=state, cost=round(c, 1))
        budget.release(est, ctx.ledger.task_cost(ctx.task.task_id) - before)


_SERIAL = None
_LIMITER = None
_COOLDOWN_UNTIL = 0.0


def _serial_gate():
    """One lock for the whole run (created inside the running loop)."""
    import asyncio
    global _SERIAL
    if _SERIAL is None:
        _SERIAL = asyncio.Lock()
    return _SERIAL


def _limiter():
    """HARNESS_MAX_CONCURRENT (default 3, 0 = unlimited): the gateway limits concurrent requests per key and answers 429 "retry in 60 s";
    the lab proxy kept this at 3. Applies to every LLM call, also the agent path."""
    import asyncio
    import contextlib
    global _LIMITER
    n = _env_int("HARNESS_MAX_CONCURRENT", 3)
    if n <= 0:
        return contextlib.nullcontext()
    if _LIMITER is None:
        _LIMITER = asyncio.Semaphore(n)
    return _LIMITER


_TRANSIENT = re.compile(r"\b(429|500|502|503|504)\b|ratelimit|rate limit|overloaded|connection (refused|reset|error)|connecterror", re.IGNORECASE)
_RETRY_AFTER = re.compile(r"(\d{1,3})\s*(?:с\b|сек|s\b|sec)", re.IGNORECASE)


def _transient(exc: BaseException) -> bool:
    """Not a timeout: a call that timed out on our side may still be generated and billed by the proxy, a repeat would pay twice."""
    return bool(_TRANSIENT.search(type(exc).__name__ + " " + str(exc)))


def _retry_after(exc: BaseException) -> float:
    """Seconds the gateway asks to wait ("Повторите через 60 с"); 60 when the text has no number, at most 120."""
    m = _RETRY_AFTER.search(str(exc))
    return float(min(int(m.group(1)), 120)) if m else 60.0


async def guarded(ledger, journal, task_id: str, model: str, factory):
    """Every LLM call (the scripted calls and the whole agent run) goes through here:
      * hard cap (HARNESS_HARD_CAP): no new call after the cap, and past HARNESS_HARD_CAP_SERIAL x cap one call at a time;
      * at most HARNESS_MAX_CONCURRENT calls at once (the gateway's per-key limit);
      * a transient failure (429, 5xx, connection) with nothing billed for the task is repeated: 5xx / connection HARNESS_RETRIES
        times (default 2, pause 2 s, 4 s), 429 HARNESS_RETRIES_429 times (default 4) after the pause the gateway asks for (60 s by default),
        and the pause is shared: while a 429 cooldown runs no call starts. Timeouts are not repeated (the proxy may have billed them).
    `factory` is a coroutine function that makes ONE attempt (a fresh model/agent/session)."""
    import asyncio
    import time

    global _COOLDOWN_UNTIL
    from . import budget, rpm

    before = ledger.task_cost(task_id)
    tries = max(_env_int("HARNESS_RETRIES", 2), 0)
    tries429 = max(_env_int("HARNESS_RETRIES_429", 4), 0)
    attempt = 0
    while True:
        delay = _COOLDOWN_UNTIL - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            async with _limiter():
                if getattr(ledger, "capped", False):
                    raise budget.BudgetStop("hard cap reached")
                gate = None
                if ledger.serial_needed():
                    gate = _serial_gate()
                    await gate.acquire()
                    if ledger.capped:
                        gate.release()
                        raise budget.BudgetStop("hard cap reached")
                    journal.event("cap_serial", task=task_id, model=model, spent=round(ledger.spent(), 1))
                ledger.active += 1
                try:
                    return await factory()
                finally:
                    ledger.active = max(ledger.active - 1, 0)
                    if gate is not None:
                        gate.release()
        except budget.BudgetStop:
            raise
        except Exception as exc:  # noqa: BLE001
            rate = rpm.is_429(exc)
            limit = tries429 if rate else tries
            if attempt >= limit or not _transient(exc) or ledger.task_cost(task_id) != before:
                raise
            wait = _retry_after(exc) if rate else 2.0 * (2 ** attempt)
            if rate:
                _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + wait)
            journal.event("call_retry", task=task_id, model=model, attempt=attempt + 1, wait=wait, rate_limit=rate,
                          error=type(exc).__name__, detail=str(exc)[:120])
            attempt += 1
            if not rate:
                await asyncio.sleep(wait)


async def _call_with_retry(ctx, system, user, max_tokens, temperature, before):
    return await guarded(ctx.ledger, ctx.journal, ctx.task.task_id, ctx.model,
                         lambda: _raw_call(ctx, system, user, max_tokens, temperature))


async def _raw_call(ctx: CallContext, system: str, user: str, max_tokens: int, temperature: float | None = None) -> str:
    from openjiuwen.core.foundation.llm.model import Model
    from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig
    from openjiuwen.core.session.agent import create_agent_session
    from openjiuwen.core.single_agent.schema.agent_card import AgentCard
    from openjiuwen.harness.factory import create_deep_agent

    from .censor import CensorRail

    ctx.n_calls += 1
    cid = f"llm_{ctx.task.task_id}_{ctx.n_calls}"
    mtok = effective_max_tokens(ctx.model, max_tokens)
    clamped = False
    if getattr(ctx.ledger, "cap", None) is not None:     # HARNESS_HARD_CAP=on: the reply may not run past the cap
        from . import tokcount
        mtok2 = ctx.ledger.clamp_output(mtok, (len(system) + len(user)) / 4.0)
        clamped, mtok = mtok2 < mtok, mtok2
        if clamped:
            ctx.journal.event("cap_clamp", task=ctx.task.task_id, model=ctx.model, max_tokens=mtok)
    model = Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI", api_key=ctx.cfg.api_key or "EMPTY", api_base=ctx.cfg.api_base,
            timeout=180, verify_ssl=False,
        ),
        model_config=ModelRequestConfig(model=ctx.model, max_tokens=mtok, **_sampling(temperature)),
    )
    card = AgentCard(id=cid, name=cid, description="single call")
    agent = create_deep_agent(
        model,
        card=card,
        system_prompt=system,
        rails=[CensorRail(ctx.task.task_id, ctx.model, ctx.ledger, ctx.journal, ctx.pred, ctx.flagged)],
        max_iterations=2,
        enable_read_image_multimodal=False,
        enable_security_rail=False,
        enable_model_anomaly_detection_rail=False,
    )
    query = {"query": user}
    session = create_agent_session(session_id=f"s_{cid}", card=card)
    await session.pre_run(inputs=query)
    out = await agent.invoke(query, session=session)
    return str(out.get("output", "")) if isinstance(out, dict) else str(out)
