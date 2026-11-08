"""Predictor v0: a PLACEHOLDER table, not measured data.

Interface is final (see harness-spec.md 3.1); numbers are guesses to be replaced by a
model trained on the generated datasets. Output per (task, model): tokens_p50,
tokens_p90, p_success; per task: p_unsolvable.
"""
from __future__ import annotations

import os
import re
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
        """HARNESS_PUNSOLV=contradiction: coding tasks whose tests contradict each other (static detector, no tokens)
        get 1.0 and are skipped by the scheduler without a model call. Default: 0 for every task."""
        if os.environ.get("HARNESS_PUNSOLV", "off").lower() == "contradiction" and task.task_type == "coding":
            return 1.0 if _coding_contradiction(task) else 0.0
        return 0.0


def _coding_contradiction(task: TaskInfo) -> bool:
    from .contradiction import find_contradictions

    ws = task.workspace
    m = re.search(r"python3?\s+(\S+\.py)", task.instruction)
    if ws is None or not m or not (ws / m.group(1)).is_file():
        return False
    return bool(find_contradictions((ws / m.group(1)).read_text(encoding="utf-8", errors="replace")))


class FlatPredictor(Predictor):
    """Variant: every task costs the same (500 tokens, 2000 with reasoning): the sorter degenerates to reward order."""

    def predict(self, task: TaskInfo, model: str) -> Prediction:
        p50 = 500.0 * (1.0 if model.endswith("-noreason") else _P50_REASON_MULT)
        return Prediction(p50, p50 * 2.5, 0.7)


# Medians measured on the scripted pipelines (SCRIPTS.md section 4, hybrid cost incl. fallback), tokens per task, noreason.
_MEASURED = {"math": 289.0, "data_analysis": 335.0, "search": 160.0, "coding": 810.0}


class MeasuredPredictor(Predictor):
    """Variant: the placeholder table replaced by measured medians (still one number per type, no task features)."""

    def predict(self, task: TaskInfo, model: str) -> Prediction:
        thinking = not model.endswith("-noreason")
        p50 = _MEASURED.get(task.task_type, 1500.0) * (3.0 if thinking else 1.0) + task.workspace_bytes / 40.0
        table = _P_SUCCESS_REASON if thinking else _P_SUCCESS_NOREASON
        return Prediction(p50, p50 * 2.5, table.get(task.task_type, 0.7))


PREDICTORS = {"placeholder": Predictor, "flat": FlatPredictor, "measured": MeasuredPredictor}


def make_predictor() -> Predictor:
    """HARNESS_PREDICTOR=placeholder (default) | flat | measured."""
    return PREDICTORS.get(os.environ.get("HARNESS_PREDICTOR", "placeholder").strip().lower(), Predictor)()
