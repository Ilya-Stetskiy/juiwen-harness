"""No API: the late-phase agent pause (harness/agent_pause.py) with fake agents that spend tokens step by step.

Checks: a run past its type threshold pauses and keeps going after the release; nothing is released while another agent
is active; the release order is reward x P(solved | still going); a type without a threshold never pauses; the paused
seconds do not count against the per-task time limit; paused runs are aborted at the end of the run hour.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["HARNESS_AGENT_PAUSE"] = "coding=6000,data_analysis=2500"

from harness import agent_pause  # noqa: E402
from harness.agent_pause import PAUSER, PauseAbort  # noqa: E402
from harness.ledger import TokenLedger  # noqa: E402


class T:
    def __init__(self, tid, typ, reward):
        self.task_id, self.task_type, self.reward = tid, typ, reward


class J:
    def __init__(self):
        self.ev = []

    def event(self, name, **kw):
        self.ev.append((time.monotonic(), name, kw))


def names(j, name):
    return [kw.get("task") for _, n, kw in j.ev if n == name]


async def fake_agent(task, ledger, steps, cost_out, log, delay=0.05):
    """`steps` model calls; each costs `cost_out` output tokens; the pause rail runs before every call."""
    base = ledger.task_cost(task.task_id)
    PAUSER.register(task, base)
    try:
        for i in range(steps):
            await PAUSER.maybe_pause(task.task_id, "m")
            await asyncio.sleep(delay)
            ledger.add(task.task_id, 0, cost_out)
        log.append(task.task_id)
        return "solved"
    except PauseAbort:
        log.append(task.task_id + ":aborted")
        return "aborted"
    finally:
        PAUSER.unregister(task.task_id)


async def scenario_release_order():
    ledger, j = TokenLedger(), J()
    PAUSER.__init__()
    PAUSER.setup(ledger, j, 100000)
    done, log = asyncio.Event(), []
    watcher = asyncio.ensure_future(PAUSER.watch(done, lambda: False))
    da_lo = T("da_low", "data_analysis", 1)       # pauses at > 2500
    da_hi = T("da_high", "data_analysis", 5)      # pauses at > 2500, higher reward -> released first
    code = T("code", "coding", 3)                 # 4 x 1000 = 4000 < 6000: never pauses, keeps the others paused
    srch = T("search", "search", 2)               # no threshold for search
    res = await asyncio.gather(fake_agent(da_lo, ledger, 5, 1000, log), fake_agent(da_hi, ledger, 5, 1000, log),
                               fake_agent(code, ledger, 4, 1000, log, delay=0.6), fake_agent(srch, ledger, 6, 1000, log))
    done.set()
    await watcher
    assert res == ["solved"] * 4, res
    assert set(names(j, "agent_pause")) == {"da_low", "da_high"}, j.ev
    assert names(j, "agent_resume") == ["da_high", "da_low"], names(j, "agent_resume")
    first_resume = min(t for t, n, _ in j.ev if n == "agent_resume")
    assert log.index("code") < log.index("da_high"), log   # no release while the coding agent was still active
    assert ledger.task_cost("da_low") == 5000 and ledger.task_cost("da_high") == 5000
    print("release order and resume: ok", log)


async def scenario_abort_and_timeout():
    ledger, j = TokenLedger(), J()
    PAUSER.__init__()
    PAUSER.setup(ledger, j, 100000)
    done, log = asyncio.Event(), []
    t_end = time.monotonic() + 1.5
    watcher = asyncio.ensure_future(PAUSER.watch(done, lambda: time.monotonic() > t_end))
    blocker = T("blocker", "coding", 1)   # active for ~2.4 s: the paused DA run is never released, then the hour ends
    da = T("da", "data_analysis", 4)
    t0 = time.monotonic()
    # the DA run pauses after ~0.15 s and waits; its active time stays far below the 1 s limit
    r = await asyncio.gather(agent_pause.run_active(fake_agent(da, ledger, 5, 1000, log), 1.0, "da"),
                             fake_agent(blocker, ledger, 4, 1000, log, delay=0.6))
    done.set()
    await watcher
    assert r[0] == "aborted", r
    assert "agent_pause_abort" in [n for _, n, _ in j.ev]
    assert time.monotonic() - t0 > 1.5   # it waited past 1 s without the active-time limit firing
    print("abort at the end of the hour, paused time not counted: ok", log)


async def scenario_timeout_still_works():
    PAUSER.__init__()
    try:
        await agent_pause.run_active(asyncio.sleep(5), 0.5, "x")
    except asyncio.TimeoutError:
        print("active-time limit still fires: ok")
        return
    raise AssertionError("no timeout")


def scenario_thresholds():
    assert agent_pause.thresholds() == {"coding": 6000.0, "data_analysis": 2500.0}
    os.environ["HARNESS_AGENT_PAUSE"] = "off"
    assert agent_pause.thresholds() == {}
    os.environ["HARNESS_AGENT_PAUSE"] = "coding=6000,data_analysis=2500"
    assert agent_pause.p_cont("data_analysis", 3500) == 0.14 and agent_pause.p_cont("coding", 500) == 0.82
    print("thresholds: ok")


async def scenario_take_turns():
    """Two long DA runs (8 x 1000 tokens, threshold 2500): after a release a run gets one more step, then pauses again,
    so the resumes alternate instead of the first run taking the whole rest."""
    ledger, j = TokenLedger(), J()
    PAUSER.__init__()
    PAUSER.setup(ledger, j, 100000)
    done, log = asyncio.Event(), []
    watcher = asyncio.ensure_future(PAUSER.watch(done, lambda: False))
    a, b = T("a", "data_analysis", 3), T("b", "data_analysis", 3)
    res = await asyncio.gather(fake_agent(a, ledger, 8, 1000, log), fake_agent(b, ledger, 8, 1000, log))
    done.set()
    await watcher
    assert res == ["solved", "solved"], res
    order = names(j, "agent_resume")
    assert len(order) >= 4 and order[0] != order[1] and order[1] != order[2], order
    assert PAUSER.is_paused("a") is False
    print("paused runs take turns: ok", order)


scenario_thresholds()
asyncio.run(scenario_release_order())
asyncio.run(scenario_abort_and_timeout())
asyncio.run(scenario_timeout_still_works())
asyncio.run(scenario_take_turns())
print("test_agent_pause: ok")
