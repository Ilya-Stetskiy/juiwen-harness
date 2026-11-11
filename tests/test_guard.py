"""Run: python tests/test_guard.py  (no API needed). Checks harness.llm.guarded: concurrency limit, 429 pause, no retry after billing,
serial mode and the hard cap."""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["HARNESS_MAX_CONCURRENT"] = "3"

from harness import llm  # noqa: E402
from harness.budget import BudgetStop  # noqa: E402
from harness.ledger import TokenLedger  # noqa: E402


class J:
    def __init__(self):
        self.ev = []

    def event(self, name, **kw):
        self.ev.append((name, kw))


async def main():
    led, j = TokenLedger(), J()
    cur, peak = [0], [0]

    async def work():
        cur[0] += 1
        peak[0] = max(peak[0], cur[0])
        await asyncio.sleep(0.03)
        cur[0] -= 1
        return "ok"

    await asyncio.gather(*[llm.guarded(led, j, f"t{i}", "m", work) for i in range(10)])
    assert peak[0] == 3, peak

    calls = []

    async def rate_limited():
        calls.append(time.monotonic())
        if len(calls) <= 2:
            raise Exception("Error code: 429 - retry in 1 s")
        return "done"

    t = time.monotonic()
    out = await asyncio.gather(llm.guarded(led, j, "a", "m", rate_limited), llm.guarded(led, j, "b", "m", rate_limited))
    assert out == ["done", "done"] and time.monotonic() - t >= 1.0

    async def billed():
        led.add("c", 0, 10)
        raise Exception("429")

    try:
        await llm.guarded(led, j, "c", "m", billed)
        raise AssertionError("a billed call must not be repeated")
    except Exception as exc:  # noqa: BLE001
        assert "429" in str(exc)

    led2 = TokenLedger()
    led2.cap, led2.serial_at = 1000.0, 98.0
    led2.add("x", 0, 99)
    peak[0] = 0
    await asyncio.gather(*[llm.guarded(led2, j, f"s{i}", "m", work) for i in range(6)])
    assert peak[0] == 1, peak

    led2.capped = True
    try:
        await llm.guarded(led2, j, "z", "m", work)
        raise AssertionError("no call after the cap")
    except BudgetStop:
        pass
    print("test_guard: ok")


asyncio.run(main())
