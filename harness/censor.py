"""Censor v1 (Rail): counts tokens and logs deviations. It stops and switches NOTHING.

Counters come from response.usage_metadata (response.usage is None in openjiuwen
0.1.20). Extension points for later versions (see harness-spec.md 3.4, 8.1):
before_model_call (clamp max_tokens), ctx.request_force_finish, ctx.request_retry,
set_llm on the inner ReActAgent.
"""
from __future__ import annotations

import time

from openjiuwen.core.single_agent.rail.base import AgentRail, ModelCallInputs

from .journal import Journal
from .ledger import TokenLedger
from .predictor import Prediction


class CensorRail(AgentRail):
    def __init__(self, task_id: str, model: str, ledger: TokenLedger, journal: Journal, pred: Prediction,
                 flagged: set[str] | None = None) -> None:
        super().__init__()
        self._task_id = task_id
        self._model = model
        self._ledger = ledger
        self._journal = journal
        self._pred = pred
        # shared per task, so a pipeline with several calls logs each deviation level once
        self._flagged: set[str] = flagged if flagged is not None else set()
        self._t0 = 0.0
        self._n = 0

    async def before_model_call(self, ctx) -> None:
        self._t0 = time.time()

    async def after_model_call(self, ctx) -> None:
        inputs = ctx.inputs
        resp = inputs.response if isinstance(inputs, ModelCallInputs) else None
        usage = getattr(resp, "usage_metadata", None)
        self._n += 1
        if usage is None:
            self._journal.event("usage_missing", task=self._task_id, model=self._model, call=self._n)
            return
        tin = int(getattr(usage, "input_tokens", 0) or 0)
        tout = int(getattr(usage, "output_tokens", 0) or 0)
        cost = self._ledger.add(self._task_id, tin, tout)
        spent = self._ledger.task_cost(self._task_id)
        self._journal.event(
            "llm_call", task=self._task_id, model=self._model, call=self._n,
            input_tokens=tin, output_tokens=tout, cost=round(cost, 1),
            task_cost=round(spent, 1), latency=round(time.time() - self._t0, 2),
        )
        for label, limit in (("p50", self._pred.tokens_p50), ("p90", self._pred.tokens_p90)):
            if spent > limit and label not in self._flagged:
                self._flagged.add(label)
                self._journal.event("deviation", task=self._task_id, model=self._model,
                                    level=label, predicted=round(limit, 1), task_cost=round(spent, 1))
