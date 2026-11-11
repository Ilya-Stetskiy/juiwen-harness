"""Adaptive run mode (HARNESS_RUN_MODE=adaptive; default off, legacy is unchanged).

Before every task the controller compares the estimated cost of the queue that is not started yet (scheduler.queue_estimate: gbm
p50 on the cheapest model, corrected by the observed actual/predicted ratios, plus the tokens in flight) with the remaining budget:

    ratio = (queue p50 + in flight) / (budget - spent)
    ratio > economy_above  -> economy values of the switches that differ between the profiles (batches, no verification)
    ratio < accuracy_below -> accuracy values (second-model verification)
    in between             -> the mode stays (hysteresis); the start in the middle band is economy

The switches are process-wide environment variables read at call time, so a switch applies to the tasks that start afterwards;
the ones already running finish as they started. A switch is logged as the event `mode_switch` (reason, remaining, estimate,
ratio). A variable set explicitly in the environment of the run is never touched. The order of tasks (reward per token) is the
same in both modes: the sorter is shared.
"""
from __future__ import annotations

import os

from . import mode as run_mode


class Controller:
    def __init__(self, sched, budget: float, journal) -> None:
        cfg = run_mode._yaml("mode_auto.yaml").get("adaptive") or {}
        self.hi = float(cfg.get("economy_above", 0.8))
        self.lo = float(cfg.get("accuracy_below", 0.5))
        self.sched, self.budget, self.journal = sched, float(budget), journal
        self.profiles = {m: run_mode.load_profile(m) for m in ("accuracy", "economy")}
        self.keys = [k for k in run_mode.switchable_keys() if k not in os.environ]   # explicit settings stay
        self.mode: str | None = None
        self.switches = 0

    def ratio(self, spent: float) -> tuple[float, dict]:
        q = self.sched.queue_estimate()
        remaining = max(self.budget - spent, 1.0)
        return (q["p50"] + q["inflight"]) / remaining, q

    def _apply(self, mode: str) -> None:
        prof = self.profiles[mode]
        for k in self.keys:
            if k in prof:
                os.environ[k] = prof[k]
            else:
                os.environ.pop(k, None)        # absent in this profile: the default
        os.environ["HARNESS_RUN_MODE_EFFECTIVE"] = mode

    def update(self, spent: float) -> None:
        r, q = self.ratio(spent)
        new, reason = self.mode, ""
        if self.mode is None:
            new, reason = ("accuracy" if r < self.lo else "economy"), "start"
        elif self.mode == "accuracy" and r > self.hi:
            new, reason = "economy", "queue does not fit"
        elif self.mode == "economy" and r < self.lo:
            new, reason = "accuracy", "budget to spare"
        if new != self.mode:
            old, self.mode = self.mode, new
            self._apply(new)
            self.switches += 1
            self.journal.event("mode_switch", old=old, mode=new, reason=reason, ratio=round(r, 3), remaining=round(self.budget - spent),
                               queue_p50=round(q["p50"]), queue_p90=round(q["p90"]), inflight=round(q["inflight"]), queue_n=q["n"])


def make(sched, budget: float, journal, mrec: dict):
    """The controller when HARNESS_RUN_MODE=adaptive, else None."""
    if mrec.get("mode") != "adaptive":
        return None
    return Controller(sched, budget, journal)
