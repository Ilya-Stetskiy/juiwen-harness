"""Requests per minute counted next to the token ledger (B9). OFF by default.

    HARNESS_RPM_LIMIT=0|<n>    0 (default): requests are only counted and logged; n: a call that would be the (n+1)-th request
                               in the last 60 seconds waits (async sleep) until a slot is free. The gateway allows 60 per
                               minute for the key, the lab proxy spaces requests itself (--rpm 55); inside the competition
                               container the harness is the only place that can do this.
Events: `rate_wait` (waited seconds), `rate_429` (the gateway answered 429; the call is not retried here).
The counter is per process: parallel sessions need the shared file of harness/budget.py (open question H7).
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import deque

_stamps: deque = deque()
_lock = asyncio.Lock() if False else None  # created lazily inside the running loop


def limit() -> int:
    try:
        return max(int(os.environ.get("HARNESS_RPM_LIMIT", "0") or 0), 0)
    except ValueError:
        return 0


def count_last_minute() -> int:
    now = time.time()
    while _stamps and now - _stamps[0] > 60.0:
        _stamps.popleft()
    return len(_stamps)


async def acquire(journal=None, task_id: str = "") -> None:
    n = limit()
    waited = 0.0
    while n and count_last_minute() >= n:
        sleep = max(60.0 - (time.time() - _stamps[0]), 0.05) + 0.05
        waited += sleep
        await asyncio.sleep(sleep)
    _stamps.append(time.time())
    if waited and journal is not None:
        journal.event("rate_wait", task=task_id, seconds=round(waited, 2), limit=n)


def is_429(exc: BaseException) -> bool:
    t = f"{type(exc).__name__} {exc}"
    return "429" in t or "RateLimit" in t
