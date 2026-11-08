"""Executor: one task, one model, one profile. The harness (not the model) writes answer.txt."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .answers import extract_final, normalize
from .config import Config
from .journal import Journal
from .ledger import TokenLedger
from .predictor import Prediction
from .tasks import TaskInfo

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

FOOTER = (
    "\n\n[Harness rules] Do NOT create answer.txt and ignore any instruction to write it. "
    "End your final reply with one line: FINAL: <answer> (the bare value only). "
    "If the task cannot be solved with the data you have, end with: FINAL: UNSOLVABLE"
)


@dataclass
class ExecResult:
    status: str  # solved | unsolvable | failed
    answer: str | None
    elapsed: float


def load_profile(task_type: str) -> dict:
    for name in (task_type, "default"):
        p = PROFILES_DIR / f"{name}.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("no profile")


def _make_tools_rail(names: list[str]):
    """Register only the tools the profile asks for: every tool schema costs tokens on every step."""
    from openjiuwen.harness.rails.base import DeepAgentRail
    from openjiuwen.harness.tools import BashTool
    from openjiuwen.harness.tools.filesystem import EditFileTool, GlobTool, GrepTool, ListDirTool, ReadFileTool, WriteFileTool

    class ProfileToolsRail(DeepAgentRail):
        priority = 100

        def __init__(self) -> None:
            super().__init__()
            self.tools: list = []

        def init(self, agent) -> None:
            lang = agent.system_prompt_builder.language
            aid = getattr(getattr(agent, "card", None), "id", None)
            so = self.sys_operation
            makers = {
                "bash": lambda: BashTool(so, lang, agent_id=aid, deny_patterns=[]),
                "read_file": lambda: ReadFileTool(so, lang, aid, enable_image_multimodal=lambda: False),
                "write_file": lambda: WriteFileTool(so, lang, aid),
                "edit_file": lambda: EditFileTool(so, lang, aid),
                "glob": lambda: GlobTool(so, lang, aid),
                "grep": lambda: GrepTool(so, lang, aid),
                "list_dir": lambda: ListDirTool(so, lang, aid),
            }
            self.tools = [makers[n]() for n in names if n in makers]
            for t in self.tools:
                agent.ability_manager.add_ability(t.card, t)

        def uninit(self, agent) -> None:
            for t in self.tools:
                name = getattr(t.card, "name", None)
                if name:
                    agent.ability_manager.remove_ability(name)

    return ProfileToolsRail()


async def run_task(task: TaskInfo, model_name: str, pred: Prediction, cfg: Config,
                   ledger: TokenLedger, journal: Journal) -> ExecResult:
    """Scripted pipeline first (cheap); the tool agent is the fallback for unrecognised or failed cases."""
    import os

    from .llm import CallContext
    from .pipelines import PIPELINES

    profile = load_profile(task.task_type)
    mode = os.environ.get("HARNESS_MODE") or profile.get("mode", "pipeline")
    pipeline = PIPELINES.get(task.task_type)
    if mode == "pipeline" and pipeline is not None:
        t0 = time.time()
        res = None
        try:
            res = await pipeline(CallContext(task, model_name, pred, cfg, ledger, journal))
        except Exception as exc:  # noqa: BLE001 - fall back to the agent
            journal.event("pipeline_error", task=task.task_id, error=type(exc).__name__, detail=str(exc)[:200])
        if res is not None:
            status, answer = res
            journal.event("pipeline_done", task=task.task_id, status=status)
            return ExecResult(status, answer, time.time() - t0)
        fallback_on = os.environ.get("HARNESS_FALLBACK", "").lower() != "off" and bool(profile.get("fallback", True))
        journal.event("pipeline_fallback", task=task.task_id, fallback=fallback_on)
        if not fallback_on:
            return ExecResult("failed", None, time.time() - t0)
    return await run_agent_task(task, model_name, pred, cfg, ledger, journal)


async def run_agent_task(task: TaskInfo, model_name: str, pred: Prediction, cfg: Config,
                         ledger: TokenLedger, journal: Journal) -> ExecResult:
    from openjiuwen.core.foundation.llm.model import Model
    from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig
    from openjiuwen.core.session.agent import create_agent_session
    from openjiuwen.core.single_agent.schema.agent_card import AgentCard
    from openjiuwen.core.sys_operation import LocalWorkConfig, OperationMode, SysOperation, SysOperationCard
    from openjiuwen.harness.factory import create_deep_agent

    from .censor import CensorRail
    from .llm import effective_max_tokens

    t0 = time.time()
    profile = load_profile(task.task_type)
    tmp_ws: str | None = None
    if task.workspace is not None:
        ws = task.workspace
    else:
        tmp_ws = tempfile.mkdtemp(prefix="ws_")
        ws = Path(tmp_ws)
    try:
        model = Model(
            model_client_config=ModelClientConfig(
                client_provider="OpenAI", api_key=cfg.api_key or "EMPTY", api_base=cfg.api_base,
                timeout=180, verify_ssl=False,
            ),
            model_config=ModelRequestConfig(
                model=model_name, max_tokens=effective_max_tokens(model_name, int(profile.get("max_tokens", 1500)))),
        )
        sysop = SysOperation(SysOperationCard(
            id=f"sysop_{task.task_id}", mode=OperationMode.LOCAL,
            work_config=LocalWorkConfig(sandbox_root=[str(ws)]),
        ))
        card = AgentCard(id=f"exec_{task.task_id}", name=f"exec_{task.task_id}", description="task executor")
        agent = create_deep_agent(
            model,
            card=card,
            system_prompt=profile["system_prompt"],
            rails=[_make_tools_rail(list(profile.get("tools", []))),
                   CensorRail(task.task_id, model_name, ledger, journal, pred)],
            sys_operation=sysop,
            cwd=str(ws),
            max_iterations=int(profile.get("max_iterations", 10)),
            enable_read_image_multimodal=False,  # otherwise DeepAgent sends a hidden probe request
            enable_security_rail=False,
            enable_model_anomaly_detection_rail=False,
            restrict_to_work_dir=True,
        )
        query = {"query": task.instruction + FOOTER}
        session = create_agent_session(session_id=f"s_{task.task_id}", card=card)
        await session.pre_run(inputs=query)
        out = await agent.invoke(query, session=session)
        text = str(out.get("output", "")) if isinstance(out, dict) else str(out)
        final = extract_final(text)
        if final is None:
            return ExecResult("failed", None, time.time() - t0)
        if final.strip().upper().startswith("UNSOLVABLE"):
            return ExecResult("unsolvable", None, time.time() - t0)
        return ExecResult("solved", normalize(task.check, final), time.time() - t0)
    finally:
        if tmp_ws:
            shutil.rmtree(tmp_ws, ignore_errors=True)
