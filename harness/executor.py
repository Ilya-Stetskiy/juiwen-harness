"""Executor: one task, one model, one profile. The harness (not the model) writes answer.txt."""
from __future__ import annotations

import json
import re
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


# Tokens one fallback agent run is expected to cost, by task type. Measured on codegen_fix (experiment
# coding_pipeline_vs_hybrid): about 9 000 per failed answerable task, 11 000 per unanswerable one; the stream
# data give about 4 000 as the mean over all fallback runs. The other types are rough placeholders.
FALLBACK_ESTIMATE = {"coding": 6000.0, "math": 3000.0, "data_analysis": 4000.0, "search": 3000.0}


def fallback_estimate(task_type: str) -> float:
    return FALLBACK_ESTIMATE.get(task_type, 4000.0)


def route_model(task, model_name: str, journal=None) -> str:
    """HARNESS_MODEL_SEARCH (empty = auto): every executor call of a search task uses that model instead of the auto
    choice. The censor (reasoning x3) and the calls follow the returned model."""
    import os
    m = os.environ.get("HARNESS_MODEL_SEARCH", "").strip()
    if m and getattr(task, "task_type", None) == "search" and m != model_name:
        if journal is not None:
            try:
                journal.event("model_override", task=task.task_id, auto=model_name, model=m, switch="HARNESS_MODEL_SEARCH")
            except Exception:  # noqa: BLE001
                pass
        return m
    return model_name


@dataclass
class ExecResult:
    status: str  # solved | unsolvable | failed | deferred (late fallback, not finished yet)
    answer: str | None
    elapsed: float
    recheck: dict | None = None   # set when the verifier wants a later recheck (HARNESS_RECHECK_QUEUE=later)


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

    from . import budget, guard, judge, limits

    model_name = route_model(task, model_name, journal)
    profile = load_profile(task.task_type)
    mode = os.environ.get("HARNESS_MODE") or profile.get("mode", "pipeline")
    pipeline = PIPELINES.get(task.task_type)
    guard.inspect_task(task, journal)   # HARNESS_GUARD=log|strip (S12); off by default
    if mode == "pipeline" and pipeline is not None:
        t0 = time.time()
        res = None
        ctx = CallContext(task, model_name, pred, cfg, ledger, journal)
        try:
            res = await pipeline(ctx)
        except (limits.CensorStop, budget.BudgetStop) as exc:   # the censor / the hard budget said stop: no fallback
            journal.event("task_stopped", task=task.task_id, by=type(exc).__name__, detail=str(exc)[:120])
            return ExecResult("failed", None, time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - fall back to the agent
            journal.event("pipeline_error", task=task.task_id, error=type(exc).__name__, detail=str(exc)[:200])
        if res is not None:
            status, answer = res
            journal.event("pipeline_done", task=task.task_id, status=status)
            if status == "solved":
                _early(task, answer)   # in every mode: a cut of the run (hard cap) after this point must not lose the ready answer
            if status == "solved" and judge.mode() != "off":   # verifier: accept / recheck / unsolvable (S10)
                try:
                    status, answer = await _verify_loop(ctx, pipeline, task, model_name, pred, cfg, ledger, journal, status, answer)
                except (limits.CensorStop, budget.BudgetStop) as exc:   # the first answer stands
                    journal.event("task_stopped", task=task.task_id, by=type(exc).__name__, detail=str(exc)[:120], stage="verify")
            return ExecResult(status, answer, time.time() - t0, ctx.extra.get("defer"))
        fb_mode = os.environ.get("HARNESS_FALLBACK", "on").lower()  # on: agent right away | late | off
        fallback_on = fb_mode != "off" and bool(profile.get("fallback", True))
        journal.event("pipeline_fallback", task=task.task_id, fallback=fallback_on, mode=fb_mode)
        if not fallback_on:
            return ExecResult("failed", None, time.time() - t0)
        if fb_mode == "late":  # the agent is a second tier: the harness runs it after the whole queue is done
            return ExecResult("deferred", None, time.time() - t0)
    return await run_agent_task(task, model_name, pred, cfg, ledger, journal)


def recheck_model(default: str) -> str:
    """HARNESS_RECHECK_MODEL: the stronger executor that repeats a doubtful task (default: the reasoning qwen3.6)."""
    import os
    return os.environ.get("HARNESS_RECHECK_MODEL", "qwen3.6-35b-a3b").strip() or default


def other_model(model_name: str) -> str:
    """HARNESS_OTHER_MODEL: an executor of ANOTHER family for the independent answer (default qwen3.8-27b, reasoning).
    If the first answer already came from it, the qwen3.6 reasoning model is used."""
    import os
    m = os.environ.get("HARNESS_OTHER_MODEL", "qwen3.8-27b").strip() or "qwen3.8-27b"
    return "qwen3.6-35b-a3b" if model_name.startswith("qwen3.8") else m


def recheck_queue() -> str:
    """HARNESS_RECHECK_QUEUE=inline (default: the recheck runs at once in the same worker) | later (the task goes back to
    the sorter: after the main queue is done, doubtful tasks are ranked by reward x (1 - p) / expected cost and rechecked
    while the budget allows)."""
    import os
    return "later" if os.environ.get("HARNESS_RECHECK_QUEUE", "inline").strip().lower() == "later" else "inline"


async def recheck_one(task, model_name, pred, cfg, ledger, journal, answer, p, extra):
    """One recheck of a doubtful answer. HARNESS_RECHECK_MODE:
         strong  the same pipeline with the stronger model; the answer with the higher verifier p stays (HARNESS_DISAGREE=strong:
                 the stronger model wins);
         other   an independent answer from a model of another family (no judge call on it); agreement is a signal for the
                 verifier (agree_other, level `other`), a disagreement goes to HARNESS_DISAGREE: higher_p (default: the second
                 answer is scored by the free verifier) | strong (trust the second) | third (a third opinion from the strong
                 model, majority of three, no majority = first answer).
       Returns (answer, p, changed)."""
    import os

    from . import judge
    from .llm import CallContext
    from .pipelines import PIPELINES

    pipeline = PIPELINES.get(task.task_type)
    if pipeline is None:
        return answer, p, False
    mode = os.environ.get("HARNESS_RECHECK_MODE", "strong").strip().lower()
    rule = os.environ.get("HARNESS_DISAGREE", "higher_p")

    async def run(model):
        c = CallContext(task, model, pred, cfg, ledger, journal)
        c.extra["recheck"] = True
        try:
            res = await pipeline(c)
        except Exception:  # noqa: BLE001 - also CensorStop / BudgetStop: the first answer stands
            return None, c
        return (res[1] if res is not None and res[0] == "solved" else None), c

    from .answers import normalize
    norm = lambda a: normalize(task.check, a or "").strip().lower()   # R-X9: the same comparison as resolve_disagree
    if mode == "other":
        m2 = other_model(model_name)
        a2, c2 = await run(m2)
        if a2 is None:
            journal.event("verify_retry", task=task.task_id, action="other", model=m2, p_before=p, p_after=None, same_answer=None, taken=False)
            return answer, p, False
        same = norm(a2) == norm(answer)
        v = judge.free_verdict(task, answer, dict(extra, agree_other=same, other_answered=True))
        p_new = v["p"] if v else p
        final, taken = answer, False
        if not same:
            if rule == "strong":
                final, taken = a2, True
            elif rule == "third":
                a3, _ = await run(recheck_model(model_name))
                if a3 is not None and norm(a3) == norm(a2):
                    final, taken = a2, True
                elif a3 is not None and norm(a3) == norm(answer):
                    final = answer
            else:
                v2 = judge.free_verdict(task, a2, dict(c2.extra, agree_other=False, other_answered=True))
                if v2 and v2["p"] > p_new:
                    final, taken = a2, True
        journal.event("verify_retry", task=task.task_id, action="other", model=m2, p_before=p, p_after=p_new, same_answer=same,
                      taken=taken, rule=rule)
        return final, p_new, taken
    m2 = recheck_model(model_name)
    a2, c2 = await run(m2)
    if a2 is None:
        return answer, p, False
    same = norm(a2) == norm(answer)
    try:
        v2 = await judge.verify_answer(c2, task, a2)
    except Exception:  # noqa: BLE001
        v2 = None
    take = bool(v2) and (v2["p"] > p if rule != "strong" else True)
    journal.event("verify_retry", task=task.task_id, action="recheck", model=m2, p_before=p, p_after=(v2 or {}).get("p"),
                  same_answer=same, taken=take and not same)
    return (a2, v2["p"], True) if take else (answer, p, False)


EARLY_ANSWER = None   # callable(task, answer) set by run_agent: the first answer is on disk BEFORE verification / recheck starts


def _early(task, answer) -> None:
    if EARLY_ANSWER is not None and answer:
        try:
            EARLY_ANSWER(task, answer)
        except Exception:  # noqa: BLE001 - never lose the run over a failed early write
            pass


async def resolve_disagree(ctx, task, model_name, pred, cfg, ledger, journal, answer, p):
    """HARNESS_VERIFY=other already ran the second model (ctx.extra["other_answer"]); no new run for it. Rule HARNESS_DISAGREE:
    higher_p (default) the answer with the higher verifier p; strong the second answer; third a third opinion by the strong
    model, majority of three (no majority: the first answer). The answer is always submitted."""
    import os

    from . import judge
    from .answers import normalize

    a2 = ctx.extra["other_answer"]
    norm = lambda a: normalize(task.check, a or "").strip().lower()
    if norm(a2) == norm(answer):
        return answer, p
    rule = os.environ.get("HARNESS_DISAGREE", "higher_p")
    final, taken = answer, False
    if rule == "strong":
        final, taken = a2, True
    elif rule == "third":
        from .llm import CallContext
        from .pipelines import PIPELINES
        c3 = CallContext(task, recheck_model(model_name), pred, cfg, ledger, journal)
        c3.extra["recheck"] = True
        try:
            res = await PIPELINES[task.task_type](c3)
            a3 = res[1] if res is not None and res[0] == "solved" else None
        except Exception:  # noqa: BLE001
            a3 = None
        if a3 is not None and norm(a3) == norm(a2):
            final, taken = a2, True
    else:
        v2 = judge.free_verdict(task, a2, dict(ctx.extra, agree_other=False, other_answered=True))
        if v2 and v2["p"] > p:
            final, taken = a2, True
    journal.event("verify_retry", task=task.task_id, action="other", model=ctx.extra.get("other_model"), p_before=p,
                  p_after=p, same_answer=False, taken=taken, rule=rule)
    return final, p


async def _verify_loop(ctx, pipeline, task, model_name, pred, cfg, ledger, journal, status, answer):
    """Verifier verdict and what follows (HARNESS_VERIFY_ACTION):
       log       only the `verify` event;
       retry     old rule: p below HARNESS_VERIFY_MIN -> same pipeline once more, keep the higher p;
       recheck   verdict "recheck" -> `recheck_one` (HARNESS_RECHECK_MODE strong|other, HARNESS_RECHECK_QUEUE inline|later,
                 HARNESS_RECHECK_TYPES = types that are rechecked, default all).
       verdict "unsolvable" (p below HARNESS_VERIFY_GIVEUP, default 0 = never): no more tokens; the answer is still
       submitted unless HARNESS_SUBMIT_UNSOLVABLE=off."""
    import os

    from . import judge
    from .llm import CallContext

    if status == "solved":
        _early(task, answer)   # a cut of the run during the verifier call / recheck must not lose the answer that is ready
    v = await judge.verify_answer(ctx, task, answer)
    if not v:
        return status, answer
    act = judge.action()
    best_p = v["p"]
    if act == "retry" and best_p < judge.min_p():
        ctx2 = CallContext(task, model_name, pred, cfg, ledger, journal)
        ctx2.extra["recheck"] = True
        try:
            res2 = await pipeline(ctx2)
        except Exception:  # noqa: BLE001 - the first answer stands (also on CensorStop / BudgetStop)
            res2 = None
        if res2 is not None and res2[0] == "solved":
            v2 = await judge.verify_answer(ctx2, task, res2[1])
            same = (res2[1] or "").strip().lower() == (answer or "").strip().lower()
            take = bool(v2) and v2["p"] > best_p
            journal.event("verify_retry", task=task.task_id, action=act, model=model_name, p_before=best_p,
                          p_after=(v2 or {}).get("p"), same_answer=same, taken=take and not same)
            if take:
                status, answer, best_p = res2[0], res2[1], v2["p"]
    elif act == "recheck" and judge.verdict(best_p) == "recheck" and task.task_type in {
            t.strip() for t in os.environ.get("HARNESS_RECHECK_TYPES", task.task_type).split(",")}:
        if ctx.extra.get("other_answer") is not None and recheck_queue() != "later":
            answer, best_p = await resolve_disagree(ctx, task, model_name, pred, cfg, ledger, journal, answer, best_p)
        elif recheck_queue() == "later":
            ctx.extra["defer"] = {"p": best_p, "answer": answer, "model": model_name, "extra": dict(ctx.extra)}
            journal.event("recheck_deferred", task=task.task_id, p=best_p, reward=task.reward)
        else:
            answer, best_p, _ = await recheck_one(task, model_name, pred, cfg, ledger, journal, answer, best_p, ctx.extra)
    if judge.verdict(best_p) == "unsolvable":
        journal.event("verdict_unsolvable", task=task.task_id, p=best_p)
        if os.environ.get("HARNESS_SUBMIT_UNSOLVABLE", "on").strip().lower() == "off":
            try:   # R-X10: the early write already put the answer on disk; "off" must remove it
                (task.dir / "answer.txt").unlink()
            except OSError:
                pass
            return "unsolvable", None
    return status, answer


def agent_retry_on() -> bool:
    """HARNESS_AGENT_RETRY=on (default) | off. The whole agent run is ONE attempt of `guarded`, and `guarded` never repeats
    a task that already paid tokens, so a 429 on step k of the agent used to end the task with an error (prod_fallback_late_60k:
    2 tasks, ~7k tokens lost). With `on` every single model call INSIDE the agent is repeated on 429 / 5xx only."""
    import os
    return os.environ.get("HARNESS_AGENT_RETRY", "on").strip().lower() != "off"


_STATUS_RE = re.compile(r"Error code:\s*(\d{3})")


def _rate_retry_rail(task_id: str, model_name: str, journal, ledger):
    """Repeats ONE model call inside the agent when the gateway refused it: 429 (key-parallel / rpm window; pause = the
    "retry in N s" of the message, 60 s by default, shared with every other call through llm._COOLDOWN_UNTIL) or 5xx
    (2, 4, 8 s), up to 3 times per call. A refused request generates nothing, so the repeat is not billed twice. Not
    repeated: timeouts and anything without such a status (the proxy may have billed them), and nothing after the hard cap.
    openjiuwen's own ModelAnomalyDetectionRail cannot do this: openjiuwen re-raises the SDK error as FrameworkError with the
    context suppressed, so the rail sees no status and no Retry-After and would repeat everything after 2-8 s."""
    import time

    from openjiuwen.core.single_agent.rail.base import AgentRail

    from . import llm, rpm

    class RateRetryRail(AgentRail):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        async def on_model_exception(self, ctx) -> None:
            exc = getattr(ctx, "exception", None)
            if exc is None or getattr(ledger, "capped", False):
                return
            text = type(exc).__name__ + " " + str(exc)
            m = _STATUS_RE.search(text)
            code = 429 if rpm.is_429(exc) else (int(m.group(1)) if m else None)
            if code is None or not (code == 429 or code >= 500) or self.count >= 3:
                self.count = 0
                return
            wait = llm._retry_after(exc) if code == 429 else 2.0 * (2 ** self.count)
            if code == 429:
                llm._COOLDOWN_UNTIL = max(llm._COOLDOWN_UNTIL, time.monotonic() + wait)
            self.count += 1
            try:
                journal.event("agent_retry", task=task_id, model=model_name, status=code, attempt=self.count, wait=wait)
            except Exception:  # noqa: BLE001
                pass
            ctx.request_retry(delay_seconds=wait)

        async def after_model_call(self, ctx) -> None:   # a call went through: the next refusal gets the full retry budget
            self.count = 0

    return RateRetryRail()


async def run_agent_task(task: TaskInfo, model_name: str, pred: Prediction, cfg: Config,
                         ledger: TokenLedger, journal: Journal, pause_base: float | None = None) -> ExecResult:
    """pause_base (late phase only): ledger cost of the task when the agent starts; the run may then pause at the
    threshold of its type (harness/agent_pause.py, HARNESS_AGENT_PAUSE) instead of going on or being cut."""
    from openjiuwen.core.foundation.llm.model import Model
    from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig
    from openjiuwen.core.session.agent import create_agent_session
    from openjiuwen.core.single_agent.schema.agent_card import AgentCard
    from openjiuwen.core.sys_operation import LocalWorkConfig, OperationMode, SysOperation, SysOperationCard
    from openjiuwen.harness.factory import create_deep_agent

    from .censor import CensorRail
    from .budget import BudgetStop
    from .llm import effective_max_tokens, guarded
    from . import agent_pause

    t0 = time.time()
    pausable = pause_base is not None and bool(agent_pause.thresholds())
    model_name = route_model(task, model_name, None)
    profile = load_profile(task.task_type)
    tmp_ws: str | None = None
    if task.workspace is not None:
        ws = task.workspace
    else:
        tmp_ws = tempfile.mkdtemp(prefix="ws_")
        ws = Path(tmp_ws)
    async def attempt() -> str:
        mtok = effective_max_tokens(model_name, int(profile.get("max_tokens", 1500)))
        if getattr(ledger, "cap", None) is not None:   # HARNESS_HARD_CAP: the reply may not run past the cap (the agent path too)
            mtok = ledger.clamp_output(mtok, (len(task.instruction) + len(profile["system_prompt"])) / 4.0)
        model = Model(
            model_client_config=ModelClientConfig(
                client_provider="OpenAI", api_key=cfg.api_key or "EMPTY", api_base=cfg.api_base,
                timeout=180, verify_ssl=False,
            ),
            model_config=ModelRequestConfig(model=model_name, max_tokens=mtok),
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
            rails=([agent_pause.make_rail(task.task_id, model_name)] if pausable else [])
                  + [_make_tools_rail(list(profile.get("tools", []))),
                     CensorRail(task.task_id, model_name, ledger, journal, pred)]
                  + ([_rate_retry_rail(task.task_id, model_name, journal, ledger)] if agent_retry_on() else []),
            sys_operation=sysop,
            cwd=str(ws),
            max_iterations=int(profile.get("max_iterations", 10)),
            enable_read_image_multimodal=False,  # otherwise DeepAgent sends a hidden probe request
            enable_security_rail=False,
            enable_model_anomaly_detection_rail=False,   # the default one also retries timeouts (may be billed twice); ours is below
            restrict_to_work_dir=True,
        )
        query = {"query": task.instruction + FOOTER}
        session = create_agent_session(session_id=f"s_{task.task_id}", card=card)
        await session.pre_run(inputs=query)
        if pausable:   # registered only once the agent really runs: a run still waiting for a call slot is not "active"
            agent_pause.PAUSER.register(task, pause_base)
        out = await agent.invoke(query, session=session)
        return str(out.get("output", "")) if isinstance(out, dict) else str(out)

    try:
        # the agent run goes through the same guard as every call: concurrency limit, 429 pause, hard cap
        text = await guarded(ledger, journal, task.task_id, model_name, attempt)
        final = extract_final(text)
        if final is None:
            return ExecResult("failed", None, time.time() - t0)
        if final.strip().upper().startswith("UNSOLVABLE"):
            return ExecResult("unsolvable", None, time.time() - t0)
        ans = normalize(task.check, final)
        _early(task, ans)
        return ExecResult("solved", ans, time.time() - t0)
    except BudgetStop as exc:   # the hard cap: no new call (agent_pause.PauseAbort is one too)
        journal.event("task_stopped", task=task.task_id, by=type(exc).__name__, detail=str(exc)[:120], stage="agent")
        return ExecResult("failed", None, time.time() - t0)
    except Exception as exc:  # noqa: BLE001 - openjiuwen may wrap the PauseAbort raised inside a rail
        if pausable and "paused agent given up" in str(exc):
            journal.event("task_stopped", task=task.task_id, by="PauseAbort", detail=str(exc)[:120], stage="agent")
            return ExecResult("failed", None, time.time() - t0)
        raise
    finally:
        if pausable:
            agent_pause.PAUSER.unregister(task.task_id)
        if tmp_ws:
            shutil.rmtree(tmp_ws, ignore_errors=True)


# Registry of LLM executors (the deterministic solvers are a separate registry: harness/solvers/).
# Interface: coroutine (task, model_name, pred, cfg, ledger, journal) -> ExecResult. Limits come from the censor
# (harness/limits.py) and the hard budget (harness/budget.py) inside every call; a refusal is ExecResult("failed"/"unsolvable").
EXECUTORS = {
    "simple": run_task,         # scripted pipeline per task type, tool agent as fallback (the "simple LLM executor")
    "agent": run_agent_task,    # tool agent (late tier / strong executor for rechecks)
}
