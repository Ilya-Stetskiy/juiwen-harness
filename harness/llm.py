"""One plain LLM call (no tools) through an openjiuwen agent with the censor rail attached.

A tool-less DeepAgent adds almost no overhead (~50 input tokens), while the censor rail keeps
counting every call by the competition formula.
"""
from __future__ import annotations

import os
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


def _sampling() -> dict:
    """HARNESS_TEMPERATURE (optional): unset keeps the provider default; 0 makes runs repeatable."""
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
    from openjiuwen.core.foundation.llm.model import Model
    from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig
    from openjiuwen.core.session.agent import create_agent_session
    from openjiuwen.core.single_agent.schema.agent_card import AgentCard
    from openjiuwen.harness.factory import create_deep_agent

    from .censor import CensorRail

    ctx.n_calls += 1
    cid = f"llm_{ctx.task.task_id}_{ctx.n_calls}"
    model = Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI", api_key=ctx.cfg.api_key or "EMPTY", api_base=ctx.cfg.api_base,
            timeout=180, verify_ssl=False,
        ),
        model_config=ModelRequestConfig(model=ctx.model, max_tokens=effective_max_tokens(ctx.model, max_tokens),
                                        **_sampling()),
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
