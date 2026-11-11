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


_PS = None


def p_success_for(task_type: str, model: str, default: float) -> float:
    """HARNESS_PSUCCESS=table (default: the placeholder tables above) | measured (profiles/p_success.json, per type and model, from the
    token stream; review R-X5: the placeholders put data_analysis and search at 0.8 and coding at 0.6, the stream says 0.59 / 0.62 / 0.83)."""
    import json
    from pathlib import Path
    global _PS
    if os.environ.get("HARNESS_PSUCCESS", "table").strip().lower() != "measured":
        return default
    if _PS is None:
        try:
            _PS = json.loads((Path(__file__).resolve().parent.parent / "profiles" / "p_success.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _PS = {}
    v = (_PS.get(task_type) or {}).get(model)
    return float(v) if v else default


@dataclass(frozen=True)
class Prediction:
    tokens_p50: float
    tokens_p90: float
    p_success: float
    tokens_p10: float = 0.0     # filled by the gbm predictor (range); 0 = unknown


class Predictor:
    def predict(self, task: TaskInfo, model: str) -> Prediction:
        thinking = not model.endswith("-noreason")
        base = _P50_NOREASON.get(task.task_type, 1500)
        p50 = base * (_P50_REASON_MULT if thinking else 1.0)
        # larger workspaces cost more input tokens per step
        p50 += task.workspace_bytes / 40.0 * (1.0 if not thinking else 1.0)
        table = _P_SUCCESS_REASON if thinking else _P_SUCCESS_NOREASON
        return Prediction(p50, p50 * 2.5, p_success_for(task.task_type, model, table.get(task.task_type, 0.7)))

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
        return Prediction(p50, p50 * 2.5, p_success_for(task.task_type, model, table.get(task.task_type, 0.7)))


_TYPES = ("math", "search", "data_analysis", "coding")


class GbmPredictor(Predictor):
    """Variant: gradient-boosted trees (profiles/cost_model.json, tools/predictor_export.py) on runtime features of the task:
    type, thinking, model family, workspace bytes, number of files, exact Qwen token counts of the instruction and of the
    prompt material. Gives a range p10 / p50 / p90. Cross-validated on the stream results (19 972 rows): median APE 0.27
    against 0.78 of the placeholder. Without the model file it falls back to the placeholder table."""

    def __init__(self) -> None:
        from . import costmodel
        self._model = costmodel.load()
        self._mat: dict[str, tuple[int, int, int]] = {}

    def _task_features(self, task: TaskInfo) -> tuple[int, int, int]:
        """(tokens of the instruction, tokens of the prompt material, number of files), cached per task."""
        hit = self._mat.get(task.task_id)
        if hit is not None:
            return hit
        from . import tokcount
        from .pipelines import clean_instruction
        q = clean_instruction(task.instruction)
        material = q
        n_files = 0
        ws = task.workspace
        try:
            if ws is not None:
                n_files = sum(1 for p in ws.rglob("*") if p.is_file())
                if task.task_type == "search":
                    from .pipelines import _read_kb
                    material = q + "\n\nDocuments:\n" + (_read_kb(ws, q) or "")
                elif task.task_type == "data_analysis":
                    from .pipelines import _data_overview
                    material = q + "\n\n" + _data_overview(ws)
        except Exception:  # noqa: BLE001 - the estimate falls back to the instruction
            material = q
        out = (tokcount.count(q), tokcount.count(material), n_files)
        self._mat[task.task_id] = out
        return out

    def predict(self, task: TaskInfo, model: str) -> Prediction:
        if self._model is None or task.task_type not in _TYPES:
            return super().predict(task, model)
        tq, tm, nf = self._task_features(task)
        x = [float(task.task_type == t) for t in _TYPES] + [float(not model.endswith("-noreason")), float(model.startswith("qwen3.8")),
                                                            float(task.workspace_bytes), float(nf), float(tq), float(tm)]
        r = self._model.predict(x)
        table = _P_SUCCESS_REASON if not model.endswith("-noreason") else _P_SUCCESS_NOREASON
        return Prediction(r["p50"], r["p90"], p_success_for(task.task_type, model, table.get(task.task_type, 0.7)), r["p10"])


PREDICTORS = {"placeholder": Predictor, "flat": FlatPredictor, "measured": MeasuredPredictor, "gbm": GbmPredictor}


def make_predictor() -> Predictor:
    """HARNESS_PREDICTOR=placeholder (default) | flat | measured | gbm."""
    return PREDICTORS.get(os.environ.get("HARNESS_PREDICTOR", "placeholder").strip().lower(), Predictor)()
