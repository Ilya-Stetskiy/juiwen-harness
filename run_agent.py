"""Entry point: python /app/run_agent.py --tasks-dir /data/tasks

Pipeline: deterministic solvers -> predictor -> scheduler (greedy knapsack with replanning)
-> pool of workers -> executor (+ censor rail) -> answer.txt written by the harness.
Never raises: every failure is logged and the run continues with the next task.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
import traceback

from harness.answers import write_answer
from harness.config import MODELS, Config
from harness.journal import Journal
from harness.ledger import TokenLedger
from harness.predictor import Predictor
from harness.solvers import try_solve
from harness.sorter import Scheduler
from harness.tasks import load_tasks


async def _worker(wid: int, sched: Scheduler, cfg: Config, ledger: TokenLedger, journal: Journal,
                  t_start: float, state: dict) -> None:
    from harness.executor import run_task

    while True:
        if cfg.limits_on and time.time() - t_start > cfg.run_seconds - cfg.max_task_seconds - 60:
            journal.event("time_stop", worker=wid)
            return
        nxt = sched.next(ledger.spent())
        if nxt is None:
            return
        task, model, pred = nxt
        before = ledger.task_cost(task.task_id)
        journal.event("task_start", task=task.task_id, type=task.task_type, model=model,
                      est_p50=round(pred.tokens_p50, 1), worker=wid)
        status, answer = "error", None
        try:
            coro = run_task(task, model, pred, cfg, ledger, journal)
            res = await (asyncio.wait_for(coro, cfg.max_task_seconds) if cfg.limits_on else coro)
            status, answer = res.status, res.answer
            if status == "solved" and answer:
                write_answer(task.dir, answer)
                state["solved"] += 1
        except asyncio.TimeoutError:
            status = "timeout"
        except Exception as exc:  # noqa: BLE001 - the run must go on
            journal.event("task_error", task=task.task_id, error=type(exc).__name__, detail=str(exc)[:200])
            traceback.print_exc(file=sys.stderr)
        actual = ledger.task_cost(task.task_id) - before
        sched.report(task, model, actual)
        if status == "deferred":
            state["deferred"].append((task, model, pred))
            journal.event("task_deferred", task=task.task_id, cost=round(actual, 1), reward=task.reward)
            continue
        journal.event("task_end", task=task.task_id, status=status, cost=round(actual, 1),
                      total=round(ledger.spent(), 1), reward=task.reward)


async def _late_worker(wid: int, queue: list, cfg: Config, ledger: TokenLedger, journal: Journal,
                       t_start: float, state: dict) -> None:
    """Second tier: tasks the pipeline could not solve go to the tool agent, highest reward first, only while the
    remaining budget covers the agent's expected cost (in-flight runs are reserved)."""
    from harness.executor import fallback_estimate, run_agent_task

    running = state["running"]  # task id -> (expected cost, ledger cost at start)

    def promised() -> float:
        """What running agents are still expected to spend: their estimate minus what is already in the ledger."""
        return sum(max(est_ - (ledger.task_cost(tid) - b0), 0.0) for tid, (est_, b0) in running.items())

    while queue:
        task, model, pred = queue[0]
        est = fallback_estimate(task.task_type)
        left = cfg.budget - ledger.spent()
        out_of_time = cfg.limits_on and time.time() - t_start > cfg.run_seconds - cfg.max_task_seconds - 60
        if out_of_time or left < est or (left - promised() < est and not running):
            queue.pop(0)  # the budget is really exhausted (nothing running that could free it)
            journal.event("task_end", task=task.task_id, status="unfunded", cost=0.0,
                          total=round(ledger.spent(), 1), reward=task.reward)
            continue
        if left - promised() < est:  # the money is promised to agents still running: wait for them
            await asyncio.sleep(1.0)
            continue
        queue.pop(0)
        before = ledger.task_cost(task.task_id)
        running[task.task_id] = (est, before)
        journal.event("task_resume", task=task.task_id, worker=wid, est=est, reward=task.reward)
        status, answer = "error", None
        try:
            coro = run_agent_task(task, model, pred, cfg, ledger, journal)
            res = await (asyncio.wait_for(coro, cfg.max_task_seconds) if cfg.limits_on else coro)
            status, answer = res.status, res.answer
            if status == "solved" and answer:
                write_answer(task.dir, answer)
                state["solved"] += 1
        except asyncio.TimeoutError:
            status = "timeout"
        except Exception as exc:  # noqa: BLE001 - the run must go on
            journal.event("task_error", task=task.task_id, error=type(exc).__name__, detail=str(exc)[:200])
            traceback.print_exc(file=sys.stderr)
        finally:
            running.pop(task.task_id, None)
        journal.event("task_end", task=task.task_id, status=status,
                      cost=round(ledger.task_cost(task.task_id) - before, 1),
                      total=round(ledger.spent(), 1), reward=task.reward)


async def amain(tasks_dir: str) -> None:
    cfg = Config.from_env()
    journal = Journal(cfg.log_dir)
    ledger = TokenLedger()
    t_start = time.time()
    tasks = load_tasks(tasks_dir)
    journal.event("run_start", n_tasks=len(tasks), budget=cfg.budget, workers=cfg.workers,
                  limits_on=cfg.limits_on, force_model=cfg.force_model)

    rest = []
    n_det = 0
    for t in tasks:
        ans = try_solve(t)
        if ans is not None:
            try:
                write_answer(t.dir, ans)
                n_det += 1
                journal.event("solver_ok", task=t.task_id, reward=t.reward)
                continue
            except OSError:
                pass
        rest.append(t)

    models = [cfg.force_model] if cfg.force_model else list(MODELS)
    sched = Scheduler(rest, models, Predictor(), float(cfg.budget))
    state = {"solved": 0, "deferred": [], "running": {}}
    await asyncio.gather(*[
        _worker(i, sched, cfg, ledger, journal, t_start, state) for i in range(cfg.workers)
    ])
    if state["deferred"]:
        queue = list(state["deferred"])  # HARNESS_LATE_ORDER=fifo keeps the order in which the pipeline gave up
        if os.environ.get("HARNESS_LATE_ORDER", "reward").lower() != "fifo":
            queue.sort(key=lambda x: -x[0].reward)
        journal.event("late_phase", n=len(queue), spent=round(ledger.spent(), 1), budget=cfg.budget)
        await asyncio.gather(*[
            _late_worker(i, queue, cfg, ledger, journal, t_start, state) for i in range(cfg.workers)
        ])
    journal.event("run_end", solved_llm=state["solved"], solved_det=n_det, spent=round(ledger.spent(), 1),
                  calls=ledger.calls, skipped=sched.skipped, correction=sched.correction(),
                  seconds=round(time.time() - t_start, 1))
    journal.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--out", default=None, help="ignored (organizers: OUT is not used)")
    args, _ = ap.parse_known_args()
    tasks_dir = os.path.abspath(args.tasks_dir)
    # openjiuwen writes ./logs relative to the working directory, but /app is read-only in the
    # competition container. Switch to a writable directory BEFORE the first openjiuwen import.
    for cand in (os.environ.get("HARNESS_CWD") or "/tmp/harness_cwd", tempfile.gettempdir()):
        try:
            os.makedirs(cand, exist_ok=True)
            os.chdir(cand)
            break
        except OSError:
            continue
    try:
        asyncio.run(amain(tasks_dir))
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
