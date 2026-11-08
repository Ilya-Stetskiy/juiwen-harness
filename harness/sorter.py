"""Scheduler: greedy knapsack by reward * p_success / tokens, replanned after every task.

Not an agent. After each finished task the ratio actual/predicted updates a correction
factor per (task_type, model) which is applied to all remaining predictions. Tokens of
tasks in flight are reserved so five parallel workers do not overspend the budget.
"""
from __future__ import annotations

import threading

from .predictor import Predictor, Prediction
from .tasks import TaskInfo

UNSOLVABLE_SKIP = 0.5
EMA_ALPHA = 0.3


class Scheduler:
    def __init__(self, tasks: list[TaskInfo], models: list[str], predictor: Predictor, budget: float) -> None:
        self._lock = threading.Lock()
        self._predictor = predictor
        self._models = list(models)
        self._budget = budget
        self._remaining: dict[str, TaskInfo] = {}
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
                self._preds[(t.task_id, m)] = predictor.predict(t, m)

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
                    ratio = t.reward * p.p_success / max(p.tokens_p50, 1.0)
                    if best is None or ratio > best[0]:
                        best = (ratio, t, m, p)
            if best is None:
                return None
            _, t, m, p = best
            del self._remaining[t.task_id]
            self._inflight[t.task_id] = p.tokens_p50
            return t, m, p

    def report(self, task: TaskInfo, model: str, actual_cost: float) -> None:
        with self._lock:
            self._inflight.pop(task.task_id, None)
            base = self._preds.get((task.task_id, model))
            if base and base.tokens_p50 > 0 and actual_cost > 0:
                key = (task.task_type, model)
                ratio = actual_cost / base.tokens_p50
                self._k[key] = (1 - EMA_ALPHA) * self._k.get(key, 1.0) + EMA_ALPHA * ratio

    def correction(self) -> dict[str, float]:
        with self._lock:
            return {f"{t}|{m}": round(v, 3) for (t, m), v in self._k.items()}
