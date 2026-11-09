"""Deterministic solvers (no LLM, zero tokens): the first block of the chain, run on ALL tasks before anything costs tokens.

Interface: a solver is `solver(task) -> Result | str | None`.
    Result("answer", text, trust)   an answer; trust "verify" = still goes through the verifier, "final" = accepted as is
    Result("unsolvable", None, "final")   the task cannot be solved (e.g. contradicting tests): no tokens are spent on it
    None / Result("unsure")         "not able": the task goes on to the LLM executors
A bare string is an answer with trust "verify". When in doubt a solver returns None: a wrong answer here also takes the
task away from the LLM executors.
The LLM executors (scripted pipelines, tool agent) are a separate registry: harness/executor.py `EXECUTORS`.
HARNESS_SOLVERS=arith,contradiction switches solvers on (default: none, the run is unchanged).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from ..tasks import TaskInfo


@dataclass
class Result:
    kind: str                  # answer | unsure | unsolvable
    answer: str | None = None
    trust: str = "verify"      # verify | final
    solver: str = ""


Solver = Callable[[TaskInfo], "Result | str | None"]

REGISTRY: list[tuple[str, Solver]] = []   # solvers that are always on (none yet)


def _contradiction(task: TaskInfo):
    if task.task_type != "coding":
        return None
    from ..predictor import _coding_contradiction
    return Result("unsolvable", None, "final") if _coding_contradiction(task) else None


def available() -> dict[str, Solver]:
    from . import arith
    return {"arith": arith.solve, "contradiction": _contradiction}


def active() -> list[tuple[str, Solver]]:
    names = [n.strip() for n in os.environ.get("HARNESS_SOLVERS", "").split(",") if n.strip()]
    reg = available()
    return REGISTRY + [(n, reg[n]) for n in names if n in reg]


def dispatch(task: TaskInfo) -> Result:
    """First solver that is sure wins. Always returns a Result (kind "unsure" when nobody is)."""
    for name, solver in active():
        try:
            r = solver(task)
        except Exception:  # a broken solver must not break the run
            r = None
        if isinstance(r, str):
            r = Result("answer", r, "verify") if r.strip() else None
        if r is not None and r.kind in ("answer", "unsolvable") and (r.kind == "unsolvable" or (r.answer or "").strip()):
            r.solver = name
            if r.answer:
                r.answer = str(r.answer).strip()
            return r
    return Result("unsure")


def try_solve(task: TaskInfo) -> str | None:
    """Compatibility: the answer of the first solver, or None."""
    r = dispatch(task)
    return r.answer if r.kind == "answer" else None
