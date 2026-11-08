"""Predictor v0: a PLACEHOLDER table, not measured data.

Interface is final (see harness-spec.md 3.1); numbers are guesses to be replaced by a
model trained on the generated datasets. Output per (task, model): tokens_p50,
tokens_p90, p_success; per task: p_unsolvable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tasks import TaskInfo

SOURCE = "placeholder"

# tokens_p50 per (task_type, thinking?) for the scripted pipeline. Rough: one sample of 8 examples,
# noreason model only (measured costs: math 45-67, data_analysis ~80, search 167-305, coding 160-400).
_P50_NOREASON = {"math": 60, "search": 250, "data_analysis": 100, "coding": 300}
_P50_REASON_MULT = 4.0
_P_SUCCESS_NOREASON = {"math": 0.6, "search": 0.8, "data_analysis": 0.8, "coding": 0.6}
_P_SUCCESS_REASON = {"math": 0.85, "search": 0.85, "data_analysis": 0.9, "coding": 0.8}


@dataclass(frozen=True)
class Prediction:
    tokens_p50: float
    tokens_p90: float
    p_success: float


class Predictor:
    def predict(self, task: TaskInfo, model: str) -> Prediction:
        thinking = not model.endswith("-noreason")
        base = _P50_NOREASON.get(task.task_type, 1500)
        p50 = base * (_P50_REASON_MULT if thinking else 1.0)
        # larger workspaces cost more input tokens per step
        p50 += task.workspace_bytes / 40.0 * (1.0 if not thinking else 1.0)
        table = _P_SUCCESS_REASON if thinking else _P_SUCCESS_NOREASON
        return Prediction(p50, p50 * 2.5, table.get(task.task_type, 0.7))

    def p_unsolvable(self, task: TaskInfo) -> float:
        return 0.0
