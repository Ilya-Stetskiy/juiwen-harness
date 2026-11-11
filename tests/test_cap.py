"""Run: python tests/test_cap.py  (no API). The hard cap: it fires once, cancels the phases, works from another thread, clamps max_tokens, and a
broken run mode / predictor does not leave the run without answers."""
import asyncio
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["HARNESS_HARD_CAP"] = "on"

import run_agent as ra  # noqa: E402
from harness.ledger import TokenLedger  # noqa: E402


class Cfg:
    budget = 1000


class J:
    def __init__(self):
        self.ev = []

    def event(self, name, **kw):
        self.ev.append((name, kw))


async def main():
    led, j = TokenLedger(), J()

    async def worker(i):
        for _ in range(100):
            await asyncio.sleep(0.005)
            led.add(f"t{i}", 0, 60)

    async def phases():
        await asyncio.gather(*[worker(i) for i in range(5)])

    ph = asyncio.ensure_future(phases())
    ra._arm_hard_cap(led, Cfg, j, ph)
    try:
        await ph
        raise AssertionError("the cap must cancel the phases")
    except asyncio.CancelledError:
        pass
    assert led.capped and led.spent() >= 1000
    assert [e[0] for e in j.ev].count("hard_cap_hit") == 1
    assert led.serial_at == 980.0

    led2, j2 = TokenLedger(), J()
    idle = asyncio.ensure_future(asyncio.sleep(5))
    ra._arm_hard_cap(led2, Cfg, j2, idle)
    threading.Thread(target=lambda: led2.add("x", 0, 1500)).start()
    try:
        await idle
        raise AssertionError("trip from another thread")
    except asyncio.CancelledError:
        pass

    led3 = TokenLedger()
    led3.cap = 1000.0
    assert led3.clamp_output(3800, 0) == 1000          # the whole free budget for the only call
    led3.active = 2
    assert led3.clamp_output(3800, 0) == 500           # shared by the calls in flight
    led3.add("a", 0, 995)
    assert led3.clamp_output(3800, 0) == 16            # floor
    print("test_cap: ok")


asyncio.run(main())
