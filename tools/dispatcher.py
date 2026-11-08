#!/usr/bin/env python3
"""Gateway dispatcher (lab tool): one process decides who uses the gateway, and when data collection must stop.

    python3 dispatcher.py run          the loop (detach it like the queue runner: setsid nohup ... &)
    python3 dispatcher.py once         one decision, prints what it would do, starts nothing
    python3 dispatcher.py status       state file + gateway fractions + thresholds
    python3 dispatcher.py selftest     decision logic and the loop with stub commands, no gateway calls

Rules
  1. Experiments of the queue (`lab.py queue`) have priority. When experiments are pending the dispatcher starts the queue
     runner (once the gateway is ready) and the filler waits.
  2. When the queue is empty the dispatcher runs the FILLER, data collection: `stream_run.py --max-chunks 1` with a small
     chunk (default 8 tasks), then looks at the queue again. An experiment that arrives waits for the end of the running
     chunk (about 1-2 minutes with 8 tasks; a retry pass of the broken-cell bucket before the chunk can take longer).
  3. ALL stop thresholds are in one file, `<lab>/dispatcher.yaml` (env DISPATCH_CONFIG), see DEFAULTS. The filler stops at
     `filler_week_stop_pct` of the weekly limit or `filler_day_stop_pct` of the daily capacity and resumes when the day
     share falls below `filler_day_resume_pct`; experiments may use more (`exp_*`). stream_run's own guard (--week-guard) and
     the proxy's capacity-halt stay as a last line; the dispatcher passes stream_run a huge --week-guard so it never touches
     STREAM_STOP by itself.
  4. Readiness: before each filler chunk and before each queue run the dispatcher checks the proxy (`gateway_ready` of
     stream_run: starts it if it is not up, restarts it after a halt once the limits recovered) and the thresholds; it does
     not start work on a halted or not-yet-ready proxy (the first stream chunk lost 152 cells to exactly that).
  5. It never starts a filler while another stream_run (gateway backend) is running: stop the old stream with
     `touch STREAM_STOP` first, it ends after its chunk.
What to collect when nothing is to be checked (scarce: wrong answers on search and data_analysis, fresh tasks for the lab) is
NOT implemented: stream_run takes types in a fixed rotation. `filler: stream` runs the ordinary stream in small portions.
Stop the dispatcher: touch <lab>/DISPATCHER_STOP (it finishes the running portion).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stream_run as sr  # noqa: E402

LAB = sr.LAB
CONFIG = Path(os.environ.get("DISPATCH_CONFIG") or LAB / "dispatcher.yaml")
QUEUE = Path(os.environ.get("LAB_EXPERIMENTS") or LAB / "experiments") / "queue"
STATE = LAB / "dispatcher_state.json"
LOG = LAB / "dispatcher.log"
STOP = LAB / "DISPATCHER_STOP"

DEFAULTS = {
    "filler": "stream",                 # stream | off
    "filler_week_stop_pct": 60,         # % of chat.week.limit: data collection stops
    "filler_day_stop_pct": 72,          # % of daily_capacity.pct_used: data collection pauses
    "filler_day_resume_pct": 65,        # ... and resumes below this
    "filler_session_min_remaining": 3000,   # tokens left in the 3 h window below which the filler pauses
    "exp_week_stop_pct": 90,            # experiments (the lab) may use more
    "exp_day_stop_pct": 88,
    "exp_session_min_remaining": 1500,
    "chunk": 8,                         # tasks per filler portion (stream default is 40)
    "seed": 11,
    "rpm": 55,
    "models": "",                       # empty = all four
    "poll_seconds": 20,
    "idle_when_blocked_seconds": 120,
    "allow_external_stream": False,     # tests only: run the filler although another stream_run is running
    "queue_cmd": None,                  # override for tests
    "filler_cmd": None,
}


def load_cfg() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG.exists():
        import yaml
        cfg.update(yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {})
    return cfg


def log(msg: str) -> None:
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------ gateway numbers and the rules
def fractions(lim: dict | None) -> dict | None:
    """Weekly share, daily-capacity share, session tokens left, abuse cooldown, from /v1/limits."""
    if lim is None:
        return None
    wk = (lim.get("chat") or {}).get("week") or {}
    se = (lim.get("chat") or {}).get("session") or {}
    d = lim.get("daily_capacity") or {}
    limit = wk.get("limit") or 0
    return {"week_pct": 100.0 * wk.get("used", 0) / limit if limit else 0.0, "week_used": wk.get("used", 0), "week_limit": limit,
            "day_pct": float(d.get("pct_used", 100)), "day_exhausted": bool(d.get("exhausted")),
            "session_left": se.get("remaining", 0), "abuse": lim.get("abuse_cooldown_sec") or 0}


def filler_gate(cfg: dict, fr: dict | None, paused: bool) -> tuple[bool, str]:
    """(allowed, reason). `paused`: the filler was stopped by the daily threshold and waits for the resume level."""
    if fr is None:
        return False, "gateway limits unreadable"
    if fr["abuse"]:
        return False, f"abuse cooldown {fr['abuse']} s"
    if fr["week_pct"] >= cfg["filler_week_stop_pct"]:
        return False, f"week {fr['week_pct']:.0f}% >= {cfg['filler_week_stop_pct']}%"
    if fr["day_exhausted"] or fr["day_pct"] >= cfg["filler_day_stop_pct"]:
        return False, f"day capacity {fr['day_pct']:.0f}% >= {cfg['filler_day_stop_pct']}%"
    if paused and fr["day_pct"] >= cfg["filler_day_resume_pct"]:
        return False, f"day capacity {fr['day_pct']:.0f}% waits for {cfg['filler_day_resume_pct']}%"
    if fr["session_left"] < cfg["filler_session_min_remaining"]:
        return False, f"3 h window has {fr['session_left']} tokens left"
    return True, "ok"


def exp_gate(cfg: dict, fr: dict | None) -> tuple[bool, str]:
    if fr is None:
        return False, "gateway limits unreadable"
    if fr["abuse"]:
        return False, f"abuse cooldown {fr['abuse']} s"
    if fr["week_pct"] >= cfg["exp_week_stop_pct"]:
        return False, f"week {fr['week_pct']:.0f}% >= {cfg['exp_week_stop_pct']}%"
    if fr["day_exhausted"] or fr["day_pct"] >= cfg["exp_day_stop_pct"]:
        return False, f"day capacity {fr['day_pct']:.0f}% >= {cfg['exp_day_stop_pct']}%"
    if fr["session_left"] < cfg["exp_session_min_remaining"]:
        return False, f"3 h window has {fr['session_left']} tokens left"
    return True, "ok"


def queue_pending() -> list[str]:
    out = []
    for d in ("running", "pending"):
        p = QUEUE / d
        if p.is_dir():
            out += [f"{d}/{f.name}" for f in sorted(p.glob("*.yaml"))]
    return out


def queue_runner_alive() -> bool:
    try:
        pid = int((QUEUE / "runner.pid").read_text() or 0)
        os.kill(pid, 0)
        return pid != os.getpid()
    except (OSError, ValueError):
        return False


def external_stream() -> list[str]:
    """Gateway-backend stream_run processes that are not ours (they would share the proxy with the filler)."""
    try:
        out = subprocess.run(["pgrep", "-af", "stream_run.py"], capture_output=True, text=True).stdout
    except OSError:
        return []
    return [ln for ln in out.splitlines() if "stream_run.py" in ln and "--backend local" not in ln and "pgrep" not in ln
            and f" {os.getpid()} " not in ln and "--selftest" not in ln and "--backfill" not in ln]


def proxy_ready(cfg: dict) -> tuple[bool, str]:
    """Proxy up and not halted; starts it / restarts after a halt through stream_run.gateway_ready (week guard disabled here)."""
    try:
        ok = sr.gateway_ready(float(cfg["rpm"]), 10 ** 12)
    except Exception as ex:  # noqa: BLE001
        return False, f"{type(ex).__name__}: {ex}"
    return (ok, "ok" if ok else "proxy not ready (halted, limits not recovered)")


def write_state(**kw) -> None:
    try:
        st = json.loads(STATE.read_text()) if STATE.exists() else {}
    except (OSError, ValueError):
        st = {}
    st.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    STATE.write_text(json.dumps(st, indent=1))


# ------------------------------------------------------------------ commands to run
def queue_command(cfg: dict) -> list[str]:
    return cfg["queue_cmd"] or [sys.executable, str(LAB / "lab.py"), "queue", "run"]


def filler_command(cfg: dict) -> list[str]:
    if cfg["filler_cmd"]:
        return list(cfg["filler_cmd"])
    cmd = [sys.executable, str(LAB / "stream_run.py"), "--seed", str(cfg["seed"]), "--chunk", str(cfg["chunk"]),
           "--max-chunks", "1", "--week-guard", str(10 ** 12), "--rpm", str(cfg["rpm"])]
    if cfg["models"]:
        cmd += ["--models", str(cfg["models"])]
    return cmd


def decide(cfg: dict, fr: dict | None, paused: bool, pending: list[str], runner_alive: bool, ext: list[str]) -> tuple[str, str]:
    """Pure decision: ('queue'|'filler'|'wait', reason). No side effects, easy to test."""
    if runner_alive:
        return "wait", "queue runner is running"
    if pending:
        ok, why = exp_gate(cfg, fr)
        return ("queue", f"{len(pending)} experiment(s) pending") if ok else ("wait", f"experiments pending but gateway gate: {why}")
    if cfg["filler"] != "stream":
        return "wait", "queue empty, filler off"
    if ext and not cfg["allow_external_stream"]:
        return "wait", "another stream_run is running (stop it with `touch STREAM_STOP`)"
    ok, why = filler_gate(cfg, fr, paused)
    return ("filler", "queue empty") if ok else ("wait", f"filler blocked: {why}")


# ------------------------------------------------------------------ the loop
def step(cfg: dict, dry: bool = False) -> tuple[str, str]:
    lim = sr.limits()
    fr = fractions(lim)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    paused = bool(st.get("filler_paused"))
    if fr and not paused and (fr["day_exhausted"] or fr["day_pct"] >= cfg["filler_day_stop_pct"]):
        paused = True
    if fr and paused and fr["day_pct"] < cfg["filler_day_resume_pct"] and not fr["day_exhausted"]:
        paused = False
    action, why = decide(cfg, fr, paused, queue_pending(), queue_runner_alive(), external_stream())
    write_state(filler_paused=paused, last_decision=action, last_reason=why, gateway=fr)
    if dry or action == "wait":
        return action, why
    ready, rwhy = proxy_ready(cfg)
    if not ready:
        write_state(last_decision="wait", last_reason=f"proxy: {rwhy}")
        return "wait", f"proxy: {rwhy}"
    cmd = queue_command(cfg) if action == "queue" else filler_command(cfg)
    log(f"{action}: {why}; running {' '.join(cmd[:4])}...")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(LAB), stdin=subprocess.DEVNULL).returncode
    log(f"{action} finished in {time.time() - t0:.0f}s, exit {rc}")
    write_state(last_run=action, last_run_seconds=round(time.time() - t0), last_exit=rc)
    return action, why


def run_loop() -> None:
    log("dispatcher start")
    last = None
    while not STOP.exists():
        cfg = load_cfg()
        action, why = step(cfg)
        if (action, why) != last:
            log(f"decision: {action} ({why})")
            last = (action, why)
        if action == "wait":
            time.sleep(cfg["idle_when_blocked_seconds"] if "blocked" in why or "proxy" in why else cfg["poll_seconds"])
        else:
            time.sleep(2)
    log("dispatcher stop (DISPATCHER_STOP present)")


def selftest() -> None:
    cfg = dict(DEFAULTS)
    fr = lambda w=10, d=10, s=19000, ab=0, ex=False: {"week_pct": w, "day_pct": d, "session_left": s, "abuse": ab, "day_exhausted": ex}
    assert decide(cfg, fr(), False, [], False, [])[0] == "filler"
    assert decide(cfg, fr(), False, ["pending/a.yaml"], False, [])[0] == "queue", "experiment first"
    assert decide(cfg, fr(), False, ["pending/a.yaml"], True, [])[0] == "wait", "runner already running"
    assert decide(cfg, fr(w=61), False, [], False, [])[0] == "wait", "filler stops at the weekly threshold"
    assert decide(cfg, fr(w=61), False, ["pending/a.yaml"], False, [])[0] == "queue", "experiments may use more"
    assert decide(cfg, fr(w=91), False, ["pending/a.yaml"], False, [])[0] == "wait"
    assert decide(cfg, fr(d=73), False, [], False, [])[0] == "wait", "daily threshold"
    assert decide(cfg, fr(d=68), True, [], False, [])[0] == "wait", "hysteresis: paused until the resume level"
    assert decide(cfg, fr(d=60), True, [], False, [])[0] == "filler"
    assert decide(cfg, fr(), False, [], False, ["python3 stream_run.py --seed 11"])[0] == "wait", "foreign stream"
    assert decide(cfg, None, False, [], False, [])[0] == "wait", "limits unreadable"
    assert decide(dict(cfg, filler="off"), fr(), False, [], False, [])[0] == "wait"
    lim = {"chat": {"week": {"used": 600000, "limit": 1000000}, "session": {"remaining": 5000}},
           "daily_capacity": {"pct_used": 46.6, "exhausted": False}, "abuse_cooldown_sec": 0}
    f = fractions(lim)
    assert abs(f["week_pct"] - 60.0) < 1e-9 and not filler_gate(cfg, f, False)[0]
    print("dispatcher selftest ok")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "selftest":
        selftest()
    elif cmd == "run":
        run_loop()
    elif cmd == "once":
        cfg = load_cfg()
        print(step(cfg, dry=True))
    elif cmd == "status":
        cfg = load_cfg()
        print(json.dumps({"config": cfg, "state": json.loads(STATE.read_text()) if STATE.exists() else None,
                          "gateway_now": fractions(sr.limits()), "pending": queue_pending(),
                          "queue_runner_alive": queue_runner_alive(), "external_stream": external_stream()}, indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
