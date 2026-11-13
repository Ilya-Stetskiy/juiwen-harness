"""Pause of the tool agent in the late phase (HARNESS_FALLBACK=late): stop a run that went past its threshold, keep its state.

    HARNESS_AGENT_PAUSE="coding=6000,data_analysis=2500"   (default) agent tokens per type after which the run pauses;
                                                         a type not listed never pauses; "off" switches the block off.

Why (lab-results, 2 395 agent runs in 142 experiments, see FINDINGS_agent_budget.md): the value of letting an agent run go on
falls with what it has already spent. Coding: 0.68 points per 1k of the tokens still to come at 2k spent, 0.31 at 6k, 0.20
at 10k; data_analysis: 1.43 at 1k, 0.10 at 4k. Fixed caps that END the task (HARNESS_CENSOR) never gave a reliable gain in
the lab and cost points on data_analysis and search, so here nothing is thrown away: the run only waits.

How: a rail checks before every model call of the agent whether the tokens the agent itself spent (ledger cost of the task
minus its cost when the agent started) passed the threshold of its type. If so, the coroutine waits on an event; the whole
agent state (messages, files in the workspace) stays in memory. The run is released when
NO agent of the late phase is running any more (only paused ones are left or the others are still waiting for a
slot): then the remaining budget cannot go anywhere better. Released first: the highest reward x P(solved | still going),
P from the same lab data (a tie: the run that spent less). A released run gets one more step (the threshold of its type)
and then pauses again, so several paused runs take turns instead of the first one taking the whole rest. While a run is
paused the late phase does not reserve tokens for it (run_agent._late_worker), so queued agent tasks start first. Near the end of the run hour (HARNESS_LIMITS=on) paused runs are aborted, not resumed. The time a
run waits does not count against its per-task time limit (`run_active`).

Only the late phase pauses: in the main pass a paused agent would hold a worker that pipeline tasks need.
"""
from __future__ import annotations

import asyncio
import os
import time

from .budget import BudgetStop

DEFAULT = "coding=6000,data_analysis=2500"

# P(solved | the agent run is still going after s tokens), lab-results (FINDINGS_agent_budget.md); used only to order releases
_P_CONT = {
    "coding": ((0, 0.82), (2000, 0.81), (3000, 0.75), (4000, 0.68), (6000, 0.47), (8000, 0.41), (10000, 0.36)),
    "data_analysis": ((0, 0.67), (1000, 0.63), (2000, 0.36), (3000, 0.14), (4000, 0.09), (6000, 0.03)),
    "math": ((0, 0.33), (4000, 0.44), (6000, 0.71)),
    "search": ((0, 0.56), (1000, 0.52)),
}


class PauseAbort(BudgetStop):
    """A paused run was given up (end of the run hour): the executor treats it like the hard cap, the task fails cleanly."""


def thresholds() -> dict[str, float]:
    raw = os.environ.get("HARNESS_AGENT_PAUSE", DEFAULT).strip()
    if raw.lower() in ("", "off", "0", "none"):
        return {}
    out: dict[str, float] = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                if float(v) > 0:
                    out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def p_cont(task_type: str, spent: float) -> float:
    p = 0.5
    for s, v in _P_CONT.get(task_type, ()):
        if spent >= s:
            p = v
    return p


class _Run:
    def __init__(self, task, base: float) -> None:
        self.task = task
        self.base = base            # ledger cost of the task when the agent started (the pipeline part is not the agent's)
        self.paused_at: float | None = None
        self.paused_total = 0.0
        self.limit: float | None = None   # agent tokens after which the run pauses (next step after every release)
        self.aborted = False
        self.event = asyncio.Event()


class Pauser:
    """One per process. `register` / `unregister` frame every pausable agent run; `watch` releases paused runs."""

    def __init__(self) -> None:
        self.runs: dict[str, _Run] = {}
        self.last_start = 0.0
        self.journal = None
        self.ledger = None
        self.budget = 0.0

    def setup(self, ledger, journal, budget: float) -> None:
        self.ledger, self.journal, self.budget = ledger, journal, float(budget)

    def register(self, task, base: float) -> None:
        self.runs[task.task_id] = _Run(task, base)
        self.last_start = time.monotonic()

    def unregister(self, task_id: str) -> None:
        self.runs.pop(task_id, None)

    def paused_seconds(self, task_id: str) -> float:
        r = self.runs.get(task_id)
        if r is None:
            return 0.0
        cur = time.monotonic() - r.paused_at if r.paused_at is not None else 0.0
        return r.paused_total + cur

    def active(self) -> int:
        return sum(1 for r in self.runs.values() if r.paused_at is None)

    def paused(self) -> list[_Run]:
        return [r for r in self.runs.values() if r.paused_at is not None]

    def is_paused(self, task_id: str) -> bool:
        """The late phase drops the reservation of a paused run: new agent tasks get the free budget first."""
        r = self.runs.get(task_id)
        return r is not None and r.paused_at is not None

    async def maybe_pause(self, task_id: str, model: str) -> None:
        r = self.runs.get(task_id)
        if r is None or self.ledger is None:
            return
        cap = thresholds().get(r.task.task_type)
        if not cap:
            return
        if r.limit is None:
            r.limit = cap
        spent = self.ledger.task_cost(task_id) - r.base
        if spent <= r.limit or getattr(self.ledger, "capped", False):
            return
        r.paused_at = time.monotonic()
        self._event("agent_pause", task=task_id, model=model, type=r.task.task_type, agent_spent=round(spent, 1),
                    threshold=round(r.limit, 1), budget_left=round(self._left(), 1))
        r.event = asyncio.Event()
        await r.event.wait()
        r.paused_total += time.monotonic() - r.paused_at
        r.paused_at = None
        if r.aborted:
            raise PauseAbort("paused agent given up at the end of the run hour")

    def _left(self) -> float:
        return self.budget - float(self.ledger.spent())

    def _event(self, name: str, **kw) -> None:
        if self.journal is not None:
            try:
                self.journal.event(name, **kw)
            except Exception:  # noqa: BLE001
                pass

    def release_best(self) -> None:
        ps = self.paused()
        if not ps:
            return
        def spent_of(r: _Run) -> float:
            return self.ledger.task_cost(r.task.task_id) - r.base

        def prio(r: _Run):   # reward x P(solved | still going); on a tie the run that spent less goes first
            return (float(r.task.reward) * p_cont(r.task.task_type, spent_of(r)), -spent_of(r))
        r = max(ps, key=prio)
        step = thresholds().get(r.task.task_type) or 0.0
        r.limit = spent_of(r) + step   # one more step, then the paused runs take turns again
        self._event("agent_resume", task=r.task.task_id, waited=round(time.monotonic() - r.paused_at, 1),
                    next_threshold=round(r.limit, 1), paused_left=len(ps) - 1, budget_left=round(self._left(), 1))
        r.event.set()

    def abort_all(self) -> None:
        for r in self.paused():
            r.aborted = True
            self._event("agent_pause_abort", task=r.task.task_id)
            r.event.set()

    async def watch(self, done: asyncio.Event, out_of_time) -> None:
        """Runs beside the late workers. Releases one paused run when no agent is active for two ticks in a row and no run
        started in the last 2 s (a worker between popping a task and starting its agent must not trigger a release)."""
        idle = 0
        while not done.is_set():
            await asyncio.sleep(0.5)
            if not self.paused():
                idle = 0
                continue
            if out_of_time():
                self.abort_all()
                continue
            if self.active() == 0 and time.monotonic() - self.last_start > 2.0:
                idle += 1
                if idle >= 2:
                    self.release_best()
                    idle = 0
            else:
                idle = 0
        self.abort_all()   # nothing may stay blocked after the phase


PAUSER = Pauser()


async def run_active(coro, limit: float, task_id: str):
    """asyncio.wait_for that does not count the seconds the run spent paused."""
    t = asyncio.ensure_future(coro)
    t0 = time.monotonic()
    try:
        while True:
            done, _ = await asyncio.wait({t}, timeout=1.0)
            if done:
                return t.result()
            if time.monotonic() - t0 - PAUSER.paused_seconds(task_id) > limit:
                t.cancel()
                try:
                    await t
                except BaseException:  # noqa: BLE001
                    pass
                raise asyncio.TimeoutError()
    finally:
        if not t.done():
            t.cancel()


def make_rail(task_id: str, model: str):
    from openjiuwen.core.single_agent.rail.base import AgentRail

    class PauseRail(AgentRail):
        async def before_model_call(self, ctx) -> None:
            await PAUSER.maybe_pause(task_id, model)

    return PauseRail()
