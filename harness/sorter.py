"""Scheduler: greedy knapsack by reward * p_success / tokens, replanned after every task.

Not an agent. After each finished task the ratio actual/predicted updates a correction
factor per (task_type, model) which is applied to all remaining predictions. Tokens of
tasks in flight are reserved so five parallel workers do not overspend the budget.
"""
from __future__ import annotations

import os
import threading

from .predictor import Predictor, Prediction
from .tasks import TaskInfo

UNSOLVABLE_SKIP = 0.5
EMA_ALPHA = 0.3


def expected_basis(p: Prediction) -> Prediction:
    """R-X4. HARNESS_COST_BASIS=p50 (default): plan on the median, as before. mean: plan on the expected cost: the cost is heavy-tailed
    (sum of p50 over baseline0 32.7k, real spend 64-69k), so the median under-counts. The mean of a lognormal with the median p50 and the
    90th percentile p90 is p50 * exp(s^2 / 2), s = ln(p90 / p50) / 1.2816, s capped at HARNESS_COST_SIGMA_MAX (default 1.0, the
    multiplier 1.65) because the quantile range of the gbm model is wider than the observed spread. Stored in `tokens_p50`, so the
    sorter, the correction factor and the adaptive controller all see the same number."""
    if os.environ.get("HARNESS_COST_BASIS", "p50").strip().lower() != "mean":
        return p
    import math
    try:
        smax = float(os.environ.get("HARNESS_COST_SIGMA_MAX", "1.0") or 1.0)
    except ValueError:
        smax = 1.0
    if p.tokens_p50 <= 0 or p.tokens_p90 <= p.tokens_p50:
        return p
    s = min(math.log(p.tokens_p90 / p.tokens_p50) / 1.2816, max(smax, 0.0))
    return Prediction(p.tokens_p50 * math.exp(s * s / 2.0), p.tokens_p90, p.p_success, p.tokens_p10)


class Scheduler:
    """Greedy knapsack by reward * p_success / tokens (the default, HARNESS_SORTER=greedy)."""

    def _score(self, order: int, t: TaskInfo, p: Prediction):
        return t.reward * p.p_success / max(p.tokens_p50, 1.0)

    def __init__(self, tasks: list[TaskInfo], models: list[str], predictor: Predictor, budget: float) -> None:
        self._lock = threading.Lock()
        self._predictor = predictor
        self._models = list(models)
        self._budget = budget
        self._remaining: dict[str, TaskInfo] = {}
        self._order = {t.task_id: i for i, t in enumerate(tasks)}
        self._preds: dict[tuple[str, str], Prediction] = {}
        self._k: dict[tuple[str, str], float] = {}
        self._inflight: dict[str, float] = {}  # task_id -> reserved estimate
        self.skipped: list[tuple[str, str]] = []
        for t in tasks:
            if predictor.p_unsolvable(t) > UNSOLVABLE_SKIP:
                self.skipped.append((t.task_id, "p_unsolvable"))
                continue
            self._remaining[t.task_id] = t
            for m in self._models:
                self._preds[(t.task_id, m)] = expected_basis(predictor.predict(t, m))

    def _corrected(self, task: TaskInfo, model: str) -> Prediction:
        p = self._preds[(task.task_id, model)]
        k = self._k.get((task.task_type, model), 1.0)
        return Prediction(p.tokens_p50 * k, p.tokens_p90 * k, p.p_success)

    def estimate(self, task: TaskInfo, model: str) -> Prediction:
        with self._lock:
            return self._corrected(task, model)

    def next(self, spent: float) -> tuple[TaskInfo, str, Prediction] | None:
        """Best (task, model) that still fits into budget - spent - reserved; None if nothing fits."""
        with self._lock:
            free = self._budget - spent - sum(self._inflight.values())
            best = None
            for t in self._remaining.values():
                for m in self._models:
                    p = self._corrected(t, m)
                    if p.tokens_p50 > free:
                        continue
                    ratio = self._score(self._order[t.task_id], t, p)
                    if best is None or ratio > best[0]:
                        best = (ratio, t, m, p)
            if best is None:
                return None
            _, t, m, p = best
            del self._remaining[t.task_id]
            self._inflight[t.task_id] = p.tokens_p50
            return t, m, p

    def pending(self) -> tuple[int, int]:
        """(tasks not started, tasks in flight): a worker that finds nothing that fits may wait while others are in flight (R-X3)."""
        with self._lock:
            return len(self._remaining), len(self._inflight)

    def take_more(self, task_type: str, model: str, k: int, spent: float) -> list[tuple[TaskInfo, Prediction]]:
        """Up to k more tasks of one type for a batch (S11): best score first, only while they fit into the budget."""
        out: list[tuple[TaskInfo, Prediction]] = []
        with self._lock:
            free = self._budget - spent - sum(self._inflight.values())
            cand = sorted((t for t in self._remaining.values() if t.task_type == task_type),
                          key=lambda t: self._score(self._order[t.task_id], t, self._corrected(t, model)), reverse=True)
            for t in cand:
                if len(out) >= k:
                    break
                p = self._corrected(t, model)
                if p.tokens_p50 > free:
                    continue
                free -= p.tokens_p50
                del self._remaining[t.task_id]
                self._inflight[t.task_id] = p.tokens_p50
                out.append((t, p))
        return out

    def report(self, task: TaskInfo, model: str, actual_cost: float) -> None:
        with self._lock:
            self._inflight.pop(task.task_id, None)
            base = self._preds.get((task.task_id, model))
            if base and base.tokens_p50 > 0 and actual_cost > 0:
                key = (task.task_type, model)
                ratio = actual_cost / base.tokens_p50
                self._k[key] = (1 - EMA_ALPHA) * self._k.get(key, 1.0) + EMA_ALPHA * ratio

    def queue_estimate(self) -> dict:
        """Cost of the tasks not started yet on the cheapest model (corrected by the observed ratios) and the tokens in flight:
        {n, p50, p90, inflight}. Used by the adaptive mode controller."""
        with self._lock:
            tot50 = tot90 = 0.0
            for t in self._remaining.values():
                ps = [self._corrected(t, m) for m in self._models]
                best = min(ps, key=lambda p: p.tokens_p50)
                tot50 += best.tokens_p50
                tot90 += best.tokens_p90
            return {"n": len(self._remaining), "p50": tot50, "p90": tot90, "inflight": sum(self._inflight.values())}

    def correction(self) -> dict[str, float]:
        with self._lock:
            return {f"{t}|{m}": round(v, 3) for (t, m), v in self._k.items()}


class FifoScheduler(Scheduler):
    """Variant: tasks in the order they were given, on the cheapest model (HARNESS_SORTER=fifo)."""

    def _score(self, order, t, p):
        return (-order, -p.tokens_p50)


class RewardScheduler(Scheduler):
    """Variant: highest reward first, on the cheapest model, price ignored (HARNESS_SORTER=reward)."""

    def _score(self, order, t, p):
        return (t.reward, -p.tokens_p50)


SCHEDULERS = {"greedy": Scheduler, "fifo": FifoScheduler, "reward": RewardScheduler}


def make_scheduler(tasks, models, predictor, budget):
    """HARNESS_SORTER=greedy (default) | fifo | reward."""
    cls = SCHEDULERS.get(os.environ.get("HARNESS_SORTER", "greedy").strip().lower(), Scheduler)
    return cls(tasks, models, predictor, budget)
