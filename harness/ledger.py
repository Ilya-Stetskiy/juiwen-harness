"""Shared token ledger. Cost formula of the competition: input/10 + output.

Reasoning tokens are NOT added separately: on the gateway they are already inside
completion_tokens (checked 2026-11-07). Cache is ignored by the rules.
"""
from __future__ import annotations

import threading
from collections import defaultdict


def cost_of(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 10.0 + output_tokens


class TokenLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0.0
        self.calls = 0
        self._per_task: dict[str, float] = defaultdict(float)

    def add(self, task_id: str, input_tokens: int, output_tokens: int) -> float:
        c = cost_of(input_tokens, output_tokens)
        with self._lock:
            self.total += c
            self.calls += 1
            self._per_task[task_id] += c
        return c

    def task_cost(self, task_id: str) -> float:
        with self._lock:
            return self._per_task[task_id]

    def spent(self) -> float:
        with self._lock:
            return self.total
