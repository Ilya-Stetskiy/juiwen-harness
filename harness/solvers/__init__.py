"""Deterministic solvers (no LLM, zero tokens).

A solver is a callable `solver(task) -> str | None`; None means "not sure or template
not recognised". The registry is empty in v0: solvers are developed against the
generated datasets and added here. When in doubt a solver must return None, because
a wrong solver answer also takes the task away from the LLM executor.
"""
from __future__ import annotations

from typing import Callable

from ..tasks import TaskInfo

Solver = Callable[[TaskInfo], "str | None"]

REGISTRY: list[Solver] = []   # solvers that are always on (none in v0)


def available() -> dict[str, Solver]:
    from . import arith
    return {"arith": arith.solve}


def active() -> list[Solver]:
    """HARNESS_SOLVERS=arith[,name...] switches registered solvers on (default: none, the run is unchanged)."""
    import os
    names = [n.strip() for n in os.environ.get("HARNESS_SOLVERS", "").split(",") if n.strip()]
    reg = available()
    return REGISTRY + [reg[n] for n in names if n in reg]


def try_solve(task: TaskInfo) -> str | None:
    for solver in active():
        try:
            ans = solver(task)
        except Exception:  # a broken solver must not break the run
            ans = None
        if ans is not None and str(ans).strip():
            return str(ans).strip()
    return None
