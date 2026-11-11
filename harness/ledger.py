"""Shared token ledger. Cost formula of the competition: input/10 + output.

Reasoning tokens are NOT added separately: on the gateway they are already inside
completion_tokens (checked 2026-11-07). Cache is ignored by the rules.
"""
from __future__ import annotations

import os
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
        # Hard cap (HARNESS_HARD_CAP=on): the moment the cost reported by the API reaches cap_line * budget, `capped` is set and
        # `on_cap` is called once (from any thread). Configured by the run (run_agent.amain); None = no cap.
        self.cap: float | None = None
        self.capped = False
        self.cap_spent = 0.0
        self.on_cap = None
        self.active = 0   # calls in flight (for the per-call output clamp near the cap)
        self.serial_at: float | None = None   # from this cost on LLM calls run one at a time (HARNESS_HARD_CAP_SERIAL x cap)

    def serial_needed(self) -> bool:
        return self.serial_at is not None and self.total >= self.serial_at

    def clamp_output(self, max_tokens: int, prompt_tokens: float) -> int:
        """Near the cap: the output limit of a call is the free budget (after the prompt, which costs prompt/10) shared between the
        calls in flight, so the provider itself cuts the reply at the cap and the overrun stays a few tokens, not a whole reply.
        Without a cap, or while the free budget is larger than max_tokens, max_tokens is returned unchanged."""
        if self.cap is None:
            return max_tokens
        with self._lock:
            free = self.cap - self.total - prompt_tokens / 10.0
            share = free / max(self.active, 1)
        return int(max(min(max_tokens, share), 16))

    def add(self, task_id: str, input_tokens: int, output_tokens: int) -> float:
        c = cost_of(input_tokens, output_tokens)
        fire = False
        with self._lock:
            self.total += c
            self.calls += 1
            self._per_task[task_id] += c
            if self.cap is not None and not self.capped and self.total >= self.cap:
                self.capped = True
                self.cap_spent = self.total
                fire = True
        if fire and self.on_cap is not None:
            try:
                self.on_cap(self.cap_spent)
            except Exception:  # noqa: BLE001 - the cap must never break the call that tripped it
                pass
        return c

    def task_cost(self, task_id: str) -> float:
        with self._lock:
            return self._per_task[task_id]

    def spent(self) -> float:
        with self._lock:
            return self.total
