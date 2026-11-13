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
from harness import batching, guard, guess
from harness.predictor import make_predictor
from harness.solvers import dispatch
from harness.sorter import make_scheduler
from harness.tasks import load_tasks


async def _worker(wid: int, sched: Scheduler, cfg: Config, ledger: TokenLedger, journal: Journal,
                  t_start: float, state: dict) -> None:
    from harness.executor import run_task

    while True:
        if cfg.limits_on and time.time() - t_start > cfg.run_seconds - cfg.max_task_seconds - 60:
            journal.event("time_stop", worker=wid)
            return
        if state.get("adaptive") is not None:   # HARNESS_RUN_MODE=adaptive: the mode for the next task
            state["adaptive"].update(ledger.spent())
        nxt = sched.next(ledger.spent())
        if nxt is None:
            # R-X3 (HARNESS_WORKER_WAIT=on, default off): nothing fits while others are in flight; their reservations are freed
            # when they finish, so wait instead of leaving for good.
            if os.environ.get("HARNESS_WORKER_WAIT", "off").strip().lower() == "on":
                n_left, n_fly = sched.pending()
                if n_left > 0 and n_fly > 0:
                    await asyncio.sleep(0.5)
                    continue
            return
        task, model, pred = nxt
        from harness import skip_psucc
        sk = skip_psucc.should_skip(sched, pred, float(cfg.budget), ledger.spent())
        if sk is not None:   # HARNESS_SKIP_PSUCC>0
            journal.event("skip_psucc", task=task.task_id, reward=task.reward, **sk)
            try:
                g = await skip_psucc.free_answer(task, model, pred, cfg, ledger, journal)
                if g:
                    write_answer(task.dir, guard.scrub_answer(g, task.task_id, journal))
            except Exception as exc:  # noqa: BLE001
                journal.event("task_error", task=task.task_id, error=type(exc).__name__, detail=str(exc)[:200])
            try:
                sched.report(task, model, 0.0)
            except Exception:  # noqa: BLE001
                pass
            journal.event("task_end", task=task.task_id, status="skipped", cost=0.0,
                          total=round(ledger.spent(), 1), reward=task.reward)
            continue
        if await batching.maybe_batch(task, model, pred, sched, cfg, ledger, journal, state, wid):  # HARNESS_BATCH_SIZE>1
            continue
        before = ledger.task_cost(task.task_id)
        journal.event("task_start", task=task.task_id, type=task.task_type, model=model,
                      est_p50=round(pred.tokens_p50, 1), worker=wid,
                      mode=os.environ.get("HARNESS_RUN_MODE_EFFECTIVE", ""))   # R-X11: adaptive switches are process-wide; the mode at the start of each task is logged so a task that saw a switch is visible
        status, answer = "error", None
        try:
            coro = run_task(task, model, pred, cfg, ledger, journal)
            res = await (asyncio.wait_for(coro, cfg.max_task_seconds) if cfg.limits_on else coro)
            status, answer = res.status, res.answer
            if status == "solved" and answer:
                write_answer(task.dir, guard.scrub_answer(answer, task.task_id, journal))
                state["solved"] += 1
                if res.recheck:   # HARNESS_RECHECK_QUEUE=later: back into the queue, ranked again after the main pass
                    state["recheck"].append((task, model, pred, res.recheck))
            elif status == "unsolvable" and guess.mode() != "off":   # HARNESS_GUESS_UNSOLVABLE
                g = await guess.guess(task, model, pred, cfg, ledger, journal)
                if g:
                    write_answer(task.dir, guard.scrub_answer(g, task.task_id, journal))
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


async def _recheck_worker(queue: list, cfg: Config, ledger: TokenLedger, journal: Journal, t_start: float) -> None:
    """Second look at doubtful answers, best expected gain first: reward x (1 - p) / expected cost. Runs while the budget
    (stop line 97%) covers twice the expected cost of a recheck."""
    from harness.executor import recheck_one

    while queue:
        if cfg.limits_on and time.time() - t_start > cfg.run_seconds - cfg.max_task_seconds - 60:   # R-X2: the same deadline as the main phase
            journal.event("time_stop", worker="recheck")
            return
        task, model, pred, info = queue.pop(0)
        est = 2.0 * max(pred.tokens_p50, 300.0)
        if cfg.budget * 0.97 - ledger.spent() < est:
            journal.event("recheck_skipped", task=task.task_id, reason="budget", left=round(cfg.budget * 0.97 - ledger.spent(), 1))
            continue
        before = ledger.task_cost(task.task_id)
        try:
            coro = recheck_one(task, model, pred, cfg, ledger, journal, info["answer"], info["p"], info["extra"])
            ans, p, changed = await (asyncio.wait_for(coro, cfg.max_task_seconds) if cfg.limits_on else coro)
        except Exception as exc:  # noqa: BLE001 (a timeout too: the answer that is already on disk stays)
            journal.event("task_error", task=task.task_id, error=type(exc).__name__, detail=str(exc)[:200])
            continue
        if changed and ans:
            write_answer(task.dir, guard.scrub_answer(ans, task.task_id, journal))
        journal.event("recheck_done", task=task.task_id, changed=changed, p=round(p, 4), cost=round(ledger.task_cost(task.task_id) - before, 1))


async def _late_worker(wid: int, queue: list, cfg: Config, ledger: TokenLedger, journal: Journal,
                       t_start: float, state: dict) -> None:
    """Second tier: tasks the pipeline could not solve go to the tool agent, highest reward first, only while the
    remaining budget covers the agent's expected cost (in-flight runs are reserved)."""
    from harness import agent_pause
    from harness.executor import fallback_estimate, run_agent_task

    running = state["running"]  # task id -> (expected cost, ledger cost at start)

    def promised() -> float:
        """What running agents are still expected to spend: their estimate minus what is already in the ledger."""
        return sum(max(est_ - (ledger.task_cost(tid) - b0), 0.0) for tid, (est_, b0) in running.items()
                   if not agent_pause.PAUSER.is_paused(tid))   # a paused run reserves nothing: queued tasks go first

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
            coro = run_agent_task(task, model, pred, cfg, ledger, journal, pause_base=before)
            # the per-task time limit counts only the seconds the agent was not paused (harness/agent_pause.py)
            res = await (agent_pause.run_active(coro, cfg.max_task_seconds, task.task_id) if cfg.limits_on else coro)
            status, answer = res.status, res.answer
            if status == "solved" and answer:
                write_answer(task.dir, guard.scrub_answer(answer, task.task_id, journal))   # the same rail as the main pass
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


def _prod_switches() -> dict:
    """The switches that decide the behaviour of a run (journal `run_start`): limits, sandbox, mode, predictor, tokenizer."""
    keys = ("HARNESS_LIMITS", "HARNESS_SANDBOX", "HARNESS_RUN_MODE", "HARNESS_PREDICTOR", "HARNESS_TOKENIZER",
            "HARNESS_VERIFY", "HARNESS_SORTER", "HARNESS_BATCH_SIZE", "HARNESS_FALLBACK", "HARNESS_HARD_CAP")
    return {k.replace("HARNESS_", "").lower(): os.environ.get(k, "") for k in keys}


async def amain(tasks_dir: str) -> None:
    cfg = Config.from_env()
    hidden = guard.hide_secrets()   # S12 sandbox layer: the API key leaves os.environ, only cfg keeps it
    journal = Journal(cfg.log_dir)
    if hidden:
        journal.event("secrets_hidden", names=hidden)
    ledger = TokenLedger()
    t_start = time.time()
    tasks = load_tasks(tasks_dir)
    from harness import mode as run_mode
    try:
        mrec = run_mode.apply(tasks, float(cfg.budget))   # HARNESS_RUN_MODE=legacy (default: nothing) | accuracy | economy | auto
    except Exception as exc:  # noqa: BLE001 - a broken profile must not leave the run without a single answer: go on as legacy
        traceback.print_exc(file=sys.stderr)
        mrec = {"requested": os.environ.get("HARNESS_RUN_MODE", ""), "mode": "legacy", "applied": {}, "overridden": []}
        journal.event("config_error", what="run mode failed, running as legacy", error=type(exc).__name__, detail=str(exc)[:200])
    journal.event("run_start", n_tasks=len(tasks), budget=cfg.budget, workers=cfg.workers,
                  limits_on=cfg.limits_on, force_model=cfg.force_model, run_mode=mrec["mode"], run_mode_requested=mrec["requested"],
                  mode_auto=mrec.get("auto"), mode_applied=mrec["applied"], mode_overridden=mrec["overridden"],
                  prod=_prod_switches())   # R-K1: what really runs (a legacy/off line in a boevoi run is a mistake of the launch)
    if not cfg.api_base:
        journal.event("config_warning", what="LLM_BASE_URL is empty: every model call will fail")   # R-X12
        print("[harness] WARNING: LLM_BASE_URL is empty", file=sys.stderr)
    if tasks and all(t.reward == 1.0 for t in tasks):
        journal.event("config_warning", what="all rewards are 1.0: the reward field is missing or has another name?")   # R-X8
        print("[harness] WARNING: all task rewards are 1.0", file=sys.stderr)
    if mrec["requested"] != "legacy":
        print(f"[harness] run mode {mrec['requested']} -> {mrec['mode']} {mrec.get('auto') or ''}", file=sys.stderr)

    rest = []
    n_det = 0
    for t in tasks:
        try:
            det = dispatch(t)
        except Exception as exc:  # noqa: BLE001 - a solver bug costs one task, not the run
            journal.event("solver_error", task=t.task_id, error=type(exc).__name__, detail=str(exc)[:200])
            rest.append(t)
            continue
        if det.kind == "unsolvable":   # deterministic verdict: no tokens are spent on the task (a guess may follow)
            journal.event("solver_unsolvable", task=t.task_id, solver=det.solver, reward=t.reward)
            g = await guess.guess(t, MODELS[0], None, cfg, ledger, journal) if guess.mode() != "off" else None
            if g:
                write_answer(t.dir, guard.scrub_answer(g, t.task_id, journal))
            continue
        ans = det.answer if det.kind == "answer" else None
        if ans is not None:
            try:
                write_answer(t.dir, ans)
                n_det += 1
                journal.event("solver_ok", task=t.task_id, reward=t.reward)
                continue
            except OSError:
                pass
        rest.append(t)

    from harness import executor
    executor.EARLY_ANSWER = lambda t, a: write_answer(t.dir, guard.scrub_answer(a, t.task_id, journal))
    models = [cfg.force_model] if cfg.force_model else list(MODELS)
    try:
        sched = make_scheduler(rest, models, make_predictor(), float(cfg.budget))
    except Exception as exc:  # noqa: BLE001 - a broken predictor (model file, switch): the placeholder predictor still plans the run
        traceback.print_exc(file=sys.stderr)
        journal.event("config_error", what="predictor failed, using the placeholder", error=type(exc).__name__, detail=str(exc)[:200])
        from harness.predictor import Predictor
        from harness.sorter import Scheduler
        sched = Scheduler(rest, models, Predictor(), float(cfg.budget))
    state = {"solved": 0, "deferred": [], "running": {}, "recheck": []}
    from harness import adaptive
    state["adaptive"] = adaptive.make(sched, float(cfg.budget), journal, mrec)
    phases = asyncio.ensure_future(_run_phases(sched, cfg, ledger, journal, t_start, state))
    _arm_hard_cap(ledger, cfg, journal, phases)
    try:
        await phases
    except asyncio.CancelledError:
        if not ledger.capped:
            raise
        # the hard cap cancelled every worker; the answers written so far stay (they are written before verification)
        journal.event("hard_cap_stop", spent=round(ledger.spent(), 1), cap=round(ledger.cap or 0.0, 1),
                      budget=cfg.budget, seconds=round(time.time() - t_start, 1))
    journal.event("run_end", solved_llm=state["solved"], solved_det=n_det, spent=round(ledger.spent(), 1),
                  calls=ledger.calls, skipped=sched.skipped, correction=sched.correction(),
                  seconds=round(time.time() - t_start, 1), hard_cap=ledger.capped)
    journal.close()


def _arm_hard_cap(ledger: TokenLedger, cfg: Config, journal: Journal, phases: "asyncio.Future") -> None:
    """HARNESS_HARD_CAP=on|off (default off): as soon as the cost reported by the API reaches HARNESS_HARD_CAP_LINE (default 1.0)
    x budget, nothing new starts (llm._ask_once raises BudgetStop) and every worker in flight is cancelled. The organizers only
    truncate the output on an overrun, so what is spent after the budget buys nothing."""
    if os.environ.get("HARNESS_HARD_CAP", "off").strip().lower() != "on":
        return
    try:
        line = float(os.environ.get("HARNESS_HARD_CAP_LINE", "1.0") or 1.0)
    except ValueError:
        line = 1.0
    loop = asyncio.get_running_loop()
    ledger.cap = float(cfg.budget) * line
    try:   # near the cap the calls run one at a time, so the clamp of max_tokens is exact (0 = never)
        serial = float(os.environ.get("HARNESS_HARD_CAP_SERIAL", "0.98") or 0.0)
    except ValueError:
        serial = 0.98
    ledger.serial_at = ledger.cap * serial if serial > 0 else None

    def on_cap(spent: float) -> None:
        journal.event("hard_cap_hit", spent=round(spent, 1), cap=round(ledger.cap or 0.0, 1), budget=cfg.budget)
        loop.call_soon_threadsafe(phases.cancel)

    ledger.on_cap = on_cap


async def _run_phases(sched, cfg: Config, ledger: TokenLedger, journal: Journal, t_start: float, state: dict) -> None:
    await asyncio.gather(*[
        _worker(i, sched, cfg, ledger, journal, t_start, state) for i in range(cfg.workers)
    ])
    if state["recheck"]:
        queue = sorted(state["recheck"], key=lambda x: -x[0].reward * (1.0 - x[3]["p"]) / max(x[2].tokens_p50, 100.0))
        journal.event("recheck_phase", n=len(queue), spent=round(ledger.spent(), 1), budget=cfg.budget)
        await asyncio.gather(*[_recheck_worker(queue, cfg, ledger, journal, t_start) for _ in range(cfg.workers)])
    if state["deferred"]:
        queue = list(state["deferred"])  # HARNESS_LATE_ORDER=fifo keeps the order in which the pipeline gave up
        if os.environ.get("HARNESS_LATE_ORDER", "reward").lower() != "fifo":
            queue.sort(key=lambda x: -x[0].reward)
        journal.event("late_phase", n=len(queue), spent=round(ledger.spent(), 1), budget=cfg.budget)
        from harness import agent_pause
        agent_pause.PAUSER.setup(ledger, journal, cfg.budget)
        done = asyncio.Event()
        out_of_time = lambda: cfg.limits_on and time.time() - t_start > cfg.run_seconds - 60   # noqa: E731
        watcher = asyncio.ensure_future(agent_pause.PAUSER.watch(done, out_of_time))
        try:
            await asyncio.gather(*[
                _late_worker(i, queue, cfg, ledger, journal, t_start, state) for i in range(cfg.workers)
            ])
        finally:
            done.set()
            await watcher


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
