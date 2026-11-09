#!/usr/bin/env python3
"""Endless interleaved evaluation stream for the harness (lab tool, runs on the GPU host).

Every chunk has the same number of tasks of each type in a fixed rotation (math, search, data_analysis, coding),
drawn without repetition from the unified datasets; each chunk is run by the harness once per model (models in
parallel) and graded. Only metadata is written to disk (tokens, calls, status, correctness): no prompts and no
answer text. Task copies and answers live in /tmp (tmpfs) and are deleted after the chunk.

The run has no limit by count or time. Stop it gracefully with `touch ~/harness-lab/STREAM_STOP` (the current chunk
is finished and graded first); `kill` also works, finished chunks are already saved.

One hybrid run gives both variants on the same tasks: `pure_*` fields are the scripted pipeline alone (a task that went
to the fallback agent counts as unanswered), `hybrid_*` fields are pipeline + fallback agent.

usage: stream_run.py [--models m1,m2,...] [--chunk 40] [--budget 1000000000] [--fallback on|off] [--seed 1]
                     [--workers 5] [--week-guard 600000] [--rpm 55] [--selftest]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
UNIFIED = HOME / "dataset-full" / "unified"
LAB = HOME / "harness-lab"
OUT = HOME / "datasets" / "token_juiwen" / "stream"
TMP = Path("/tmp/hl_stream")
STOP = LAB / "STREAM_STOP"
PROXY_STOP = HOME / "datasets" / "token_juiwen" / "PROXY_STOP"
PROXY_PORT = 18443
ALL_MODELS = ["qwen3.6-35b-a3b-noreason", "qwen3.6-35b-a3b", "qwen3.8-27b-noreason", "qwen3.8-27b"]
# pytest-checked sources are left out: the harness regex for the test command does not match `python3 -m pytest`
SOURCES = {
    "math": ["gsm8k", "svamp", "asdiv", "math500", "math_algebra", "math_prealgebra", "math_number_theory",
             "math_counting_and_probability", "math_geometry", "math_intermediate_algebra", "math_precalculus"],
    "search": ["hotpotqa", "2wikimultihopqa", "musique_full", "squad_v2"],
    "data_analysis": ["tabfact", "wikitablequestions", "dabench"],
    "coding": ["humanevalfix", "mbpp", "codegen_fix"],
}
ROTATION = ["math", "search", "data_analysis", "coding"]
SRC_PATH = {s: UNIFIED / t / f"{s}.jsonl" for t, ss in SOURCES.items() for s in ss}


TAG = ""          # "" for the gateway stream, "_local" for the local-model stream


def log(msg: str) -> None:
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"driver{TAG}.log", "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ sampling without repetition
def load_row(src: str, off: int) -> dict:
    with open(SRC_PATH[src], "rb") as f:
        f.seek(off)
        return json.loads(f.readline())


def write_task_dir(r: dict, d: Path) -> None:
    """task.json without `expected` plus workspace files."""
    d.mkdir(parents=True)
    task = {x: r[x] for x in ("task_id", "task_type", "instruction", "reward", "check") if x in r}
    (d / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=1), encoding="utf-8")
    for rel, content in (r.get("files") or {}).items():
        p = d / "workspace" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="")


def grading_entry(r: dict, src: str, order: int, keep_ref: bool = False) -> dict:
    g = {"task_type": r["task_type"], "check": r["check"], "reward": r["reward"], "expected": r.get("expected"),
         "source": src, "unanswerable": r["check"] == "unanswerable", "order": order, "off": r.get("_off")}
    if keep_ref:
        g["ref"] = (r.get("meta") or {}).get("reference_files")
    return g


class Sampler:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.seed = seed
        self.idx_dir = LAB / "index"
        self.idx_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = OUT / f"state_seed{seed}{TAG}.json"
        self.state = json.load(open(self.state_file)) if self.state_file.exists() else {"pos": {}, "chunks": 0}
        self.offsets: dict[str, list[int]] = {}
        self.perm: dict[str, list[int]] = {}

    def _offsets(self, src: str) -> list[int]:
        if src in self.offsets:
            return self.offsets[src]
        cache = self.idx_dir / f"{src}.json"
        if cache.exists():
            offs = json.load(open(cache))
        else:
            log(f"indexing {src} ...")
            offs, pos = [], 0
            with open(SRC_PATH[src], "rb") as f:
                for line in f:
                    offs.append(pos)
                    pos += len(line)
            json.dump(offs, open(cache, "w"))
        self.offsets[src] = offs
        return offs

    def next_row(self, src: str) -> dict | None:
        offs = self._offsets(src)
        if src not in self.perm:
            p = list(range(len(offs)))
            random.Random(f"{self.seed}:{src}").shuffle(p)
            self.perm[src] = p
        pos = self.state["pos"].get(src, 0)
        if pos >= len(offs):
            return None
        self.state["pos"][src] = pos + 1
        off = offs[self.perm[src][pos]]
        r = load_row(src, off)
        r["_off"] = off
        return r

    def save(self) -> None:
        json.dump(self.state, open(self.state_file, "w"))


def build_chunk(sm: Sampler, n: int, chunk_dir: Path, keep_ref: bool = False) -> dict:
    """Write task dirs (task.json without `expected`) and return the grading table."""
    shutil.rmtree(chunk_dir, ignore_errors=True)
    (chunk_dir / "tasks").mkdir(parents=True)
    grading: dict[str, dict] = {}
    k = 0
    while len(grading) < n:
        t = ROTATION[k % 4]
        k += 1
        src = sm.rng.choice(SOURCES[t])
        r = sm.next_row(src)
        if r is None or r["task_id"] in grading:
            continue
        write_task_dir(r, chunk_dir / "tasks" / r["task_id"])
        grading[r["task_id"]] = grading_entry(r, src, len(grading), keep_ref)
    sm.save()
    return grading


# ------------------------------------------------------------------ grading
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‘": "'", "’": "'", "“": '"', "”": '"'})


def _norm(s: str) -> str:
    s = s.translate(_DASHES).lower()
    s = "".join(c for c in s if c not in set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _num(s: str):
    try:
        return float(str(s).replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


def _dab(s: str) -> dict:
    return {k.strip().lower(): v.strip() for k, v in re.findall(r"@([\w\-]+)\[(.*?)\]", s, re.S)}


def is_correct(check: str, expected, ans: str, ws: Path) -> bool:
    if check == "exact_int":
        a, e = _num(ans), _num(expected)
        return a is not None and e is not None and a == e
    if check == "exact_num":
        a, e = _num(ans), _num(expected)
        return a is not None and e is not None and abs(a - e) <= 1e-6 * max(1.0, abs(e))
    if check == "exact_str":
        return _norm(ans) == _norm(str(expected))
    if check == "exact_set":
        sp = lambda t: sorted(_norm(x) for x in re.split(r"\|", t) if x.strip())
        return sp(ans) == sp(str(expected))
    if check == "dabench_format":
        a, e = _dab(ans), _dab(str(expected))
        if not e or set(a) != set(e):
            return False
        for k, ev in e.items():
            x, y = _num(a[k]), _num(ev)
            if x is not None and y is not None:
                if abs(x - y) > 0.011 * max(1.0, abs(y) * 0) + 1e-9:
                    return False
            elif _norm(a[k]) != _norm(ev):
                return False
        return True
    if check == "unittest":
        if ans.strip() != "DONE":
            return False
        try:
            r = subprocess.run([sys.executable, "test_mylib.py"], cwd=ws, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0
    return False


def journal_rows(log_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(log_dir / "run-*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def grade_chunk(model: str, chunk_idx: int, tasks_dir: Path, log_dir: Path, grading: dict) -> list[dict]:
    ev = collections.defaultdict(list)
    for r in journal_rows(log_dir):
        if "task" in r:
            ev[r["task"]].append(r)
    out = []
    for tid, g in grading.items():
        e = ev.get(tid, [])
        calls = [x for x in e if x["event"] == "llm_call"]
        fb = next((x for x in e if x["event"] == "pipeline_fallback"), None)
        start = next((x for x in e if x["event"] == "task_start"), None)
        end = next((x for x in e if x["event"] == "task_end"), None)
        pcalls = [c for c in calls if fb is None or c["ts"] <= fb["ts"]]
        acalls = [c for c in calls if fb is not None and c["ts"] > fb["ts"]]
        errs = [f'{x["event"]}:{x.get("error")}:{(x.get("detail") or "")[:160]}' for x in e
                if x["event"] in ("task_error", "pipeline_error")]
        a = tasks_dir / tid / "answer.txt"
        ans = a.read_text(encoding="utf-8", errors="replace").strip() if a.exists() else None
        answered = ans is not None and ans != ""
        if g["unanswerable"]:
            ok = not answered
        else:
            ok = answered and is_correct(g["check"], g["expected"], ans, tasks_dir / tid / "workspace")
        lenient = False
        if not g["unanswerable"] and answered and g["check"] == "exact_str":
            e_, a_ = _norm(str(g["expected"])), _norm(ans)
            lenient = bool(e_) and bool(a_) and (e_ in a_ or a_ in e_)     # long reference phrases vs the short value
        used_fb = fb is not None
        c_all = sum(c["cost"] for c in calls)
        out.append({
            "chunk": chunk_idx, "attempt": 1, "model": model, "task_id": tid, "type": g["task_type"], "source": g["source"],
            "check": g["check"], "reward": g["reward"], "unanswerable": g["unanswerable"], "order": g["order"],
            "status": end["status"] if end else "not_run", "errors": errs[:3], "lenient_correct": bool(ok or lenient), "answered": answered, "correct": bool(ok),
            "fallback": used_fb,
            # hybrid = pipeline + fallback agent; pure = pipeline alone (a fallback task is unanswered in pure)
            "hybrid_correct": bool(ok), "hybrid_cost": round(c_all, 1),
            "pure_correct": bool(ok) if not used_fb else bool(g["unanswerable"]),
            "pure_answered": answered and not used_fb,
            "pure_cost": round(sum(c["cost"] for c in pcalls), 1),
            "pipeline_calls": len(pcalls), "agent_calls": len(acalls),
            "in_tokens": sum(c["input_tokens"] for c in calls), "out_tokens": sum(c["output_tokens"] for c in calls),
            "seconds": round(end["ts"] - start["ts"], 1) if (start and end) else None,
        })
    return out



# ------------------------------------------------------------------ bucket of broken cells and their retries
TRANSIENT = re.compile(r"proxy halted|RateLimitError|InternalServerError|APITimeoutError|Request timed out|APIConnectionError|"
                       r"Connection error|Слишком много|Превышен лимит|Bad Gateway|Service Unavailable|Gateway Time-out", re.I)
MAX_ATTEMPTS = 3
RETRY_BATCH = 40          # cells per model in one retry pass


def is_transient(row: dict) -> bool:
    """A cell worth running again: not run at all, or an exception that looks like infrastructure (or unknown: old rows)."""
    if row["status"] == "not_run":
        return True
    if row["status"] != "error":
        return False
    errs = row.get("errors")
    return (not errs) or any(TRANSIENT.search(e) for e in errs)


def bucket_path() -> Path:
    return OUT / f"bucket{TAG}.jsonl"


def load_bucket() -> list[dict]:
    p = bucket_path()
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def save_bucket(entries: list[dict]) -> None:
    tmp = bucket_path().with_suffix(".tmp")
    with open(tmp, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, bucket_path())


def add_to_bucket(rows: list[dict], grading: dict) -> int:
    cur = load_bucket()
    have = {(e["model"], e["task_id"], e["chunk"]) for e in cur}
    n = 0
    for r in rows:
        key = (r["model"], r["task_id"], r["chunk"])
        if is_transient(r) and key not in have and grading.get(r["task_id"], {}).get("off") is not None:
            g = grading[r["task_id"]]
            cur.append({"model": r["model"], "task_id": r["task_id"], "source": g["source"], "off": g["off"],
                        "chunk": r["chunk"], "attempts": r.get("attempt", 1)})
            n += 1
    if n:
        save_bucket(cur)
    return n


def process_bucket(models: list[str], a, base_url: str, results: Path) -> int:
    """Run the broken cells again (only them), model by model in parallel; returns how many were retried."""
    bucket = load_bucket()
    batch: dict[str, list[dict]] = collections.defaultdict(list)
    for e in bucket:
        if e["model"] in models and e["attempts"] < MAX_ATTEMPTS and len(batch[e["model"]]) < RETRY_BATCH:
            batch[e["model"]].append(e)
    if not batch:
        return 0
    rid = int(time.time())
    base = TMP / f"retry_{rid}"
    shutil.rmtree(base, ignore_errors=True)
    grading_by_model, procs = {}, {}
    for m, ents in batch.items():
        mdir = base / m
        (mdir / "tasks").mkdir(parents=True)
        gr = {}
        for e in ents:
            r = load_row(e["source"], e["off"])
            r["_off"] = e["off"]
            write_task_dir(r, mdir / "tasks" / r["task_id"])
            gr[r["task_id"]] = grading_entry(r, e["source"], len(gr))
        grading_by_model[m] = gr
        env = dict(os.environ, HARNESS_FORCE_MODEL=m, LLM_BASE_URL=base_url, LLM_API_KEY="x", HARNESS_CWD=str(mdir / "cwd"),
                   BUDGET=str(a.budget), HARNESS_LOG_DIR=str(mdir / "logs"), HARNESS_WORKERS=str(a.workers),
                   HARNESS_LIMITS="off", HARNESS_FALLBACK=a.fallback)
        procs[m] = subprocess.Popen([str(LAB / ".venv" / "bin" / "python"), str(LAB / "harness" / "run_agent.py"),
                                     "--tasks-dir", str(mdir / "tasks")], cwd=str(LAB), env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    for p in procs.values():
        p.wait()
    new_rows, fixed = [], 0
    for m, ents in batch.items():
        rows = grade_chunk(m, -1, base / m / "tasks", base / m / "logs", grading_by_model[m])
        by_id = {r["task_id"]: r for r in rows}
        for e in ents:
            r = by_id.get(e["task_id"])
            if r is None:
                continue
            r["chunk"], r["attempt"], r["retry"] = e["chunk"], e["attempts"] + 1, True
            new_rows.append(r)
            if is_transient(r):
                e["attempts"] += 1
            else:
                e["done"] = True
                fixed += 1
    with open(results, "a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    save_bucket([e for e in bucket if not e.get("done")])
    shutil.rmtree(base, ignore_errors=True)
    log(f"retry pass: {len(new_rows)} cells run, {fixed} fixed, {len(new_rows) - fixed} still broken; "
        f"bucket left {len([e for e in load_bucket() if e['attempts'] < MAX_ATTEMPTS])} due")
    return len(new_rows)


def backfill_bucket() -> None:
    """Fill the bucket from the results file: transient cells that have no later good attempt. Offsets are found by task id."""
    results = OUT / f"results{TAG}.jsonl"
    rows = [json.loads(l) for l in open(results) if l.strip()]
    good = {(r["model"], r["task_id"], r["chunk"]) for r in rows if not is_transient(r)}
    if TAG == "":      # the debug rerun of chunk 0 replaced the broken cells of that chunk
        dbg = OUT / "results_dbg.jsonl"
        if dbg.exists():
            good |= {(r["model"], r["task_id"], r["chunk"]) for r in map(json.loads, open(dbg)) if not is_transient(r)}
    broken = {}
    for r in rows:
        k = (r["model"], r["task_id"], r["chunk"])
        if is_transient(r) and k not in good:
            broken[k] = r
    need = collections.defaultdict(set)
    for (m, tid, ch), r in broken.items():
        need[r["source"]].add(tid)
    offs: dict[tuple, int] = {}
    pat = re.compile(rb'^\{"task_id": "([^"]+)"')
    for src, ids in need.items():
        log(f"locating {len(ids)} tasks in {src} ...")
        pos = 0
        with open(SRC_PATH[src], "rb") as f:
            for line in f:
                m = pat.match(line)
                if m and m.group(1).decode() in ids:
                    offs[(src, m.group(1).decode())] = pos
                pos += len(line)
    cur = load_bucket()
    have = {(e["model"], e["task_id"], e["chunk"]) for e in cur}
    n = 0
    for (m, tid, ch), r in broken.items():
        if (m, tid, ch) in have or (r["source"], tid) not in offs:
            continue
        cur.append({"model": m, "task_id": tid, "source": r["source"], "off": offs[(r["source"], tid)], "chunk": ch,
                    "attempts": r.get("attempt", 1)})
        n += 1
    save_bucket(cur)
    log(f"backfill: {n} cells added, bucket size {len(cur)}; by status {dict(collections.Counter(r['status'] for r in broken.values()))}")

# ------------------------------------------------------------------ gateway guard
def _key() -> str:
    return open(HOME / "secrets" / "neuraldeep.key").read().strip()


def limits() -> dict | None:
    try:
        req = urllib.request.Request("https://api.neuraldeep.ru/v1/limits", headers={"Authorization": "Bearer " + _key()})
        return json.load(urllib.request.urlopen(req, timeout=15))
    except Exception:
        return None


def proxy_state() -> dict | None:
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/health", timeout=3))
    except Exception:
        return None


def start_proxy(rpm: float) -> None:
    py = str(HOME / "jiuwenswarm" / ".venv" / "bin" / "python3")
    PROXY_STOP.unlink(missing_ok=True)
    subprocess.Popen([py, str(HOME / "scripts" / "nd_proxy.py"), "--listen", f"127.0.0.1:{PROXY_PORT}", "--rpm", str(rpm),
                      "--max-concurrent", "3", "--daily-cap", "60000", "--capacity-halt", "90",
                      "--log", str(OUT / "gateway_calls_stream.jsonl"), "--stop-file", str(PROXY_STOP)],
                     stdout=open(OUT / "proxy_stream.out", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    time.sleep(3)


def gateway_ready(rpm: float, week_guard: int) -> bool:
    """True when it is safe to send the next chunk; restarts the proxy after it halted and the limits recovered."""
    lim = limits()
    if lim is not None:
        week = (lim.get("chat", {}).get("week") or {}).get("used", 0)
        if week >= week_guard:
            log(f"weekly usage {week} >= guard {week_guard}: stopping the stream to keep a reserve")
            STOP.touch()
            return False
    st = proxy_state()
    if st is None:
        start_proxy(rpm)
        return proxy_state() is not None
    if st.get("stopped"):
        d = (lim or {}).get("daily_capacity") or {}
        ok = lim is not None and not d.get("exhausted") and d.get("pct_used", 100) < 80 and \
            ((lim.get("chat", {}).get("session") or {}).get("remaining", 0) > 1500) and not (lim.get("abuse_cooldown_sec") or 0)
        if not ok:
            log("proxy halted and the gateway limits have not recovered; waiting 5 min")
            return False
        subprocess.run(["pkill", "-f", f"nd_proxy.py --listen 127.0.0.1:{PROXY_PORT}"])
        time.sleep(2)
        start_proxy(rpm)
        log("proxy restarted after halt")
    return True


# ------------------------------------------------------------------ run one chunk for all models
def run_chunk(models: list[str], chunk_idx: int, grading: dict, chunk_dir: Path, a, base_url: str) -> list[dict]:
    procs = {}
    for m in models:
        mdir = chunk_dir / m
        shutil.copytree(chunk_dir / "tasks", mdir / "tasks")
        env = dict(os.environ, HARNESS_FORCE_MODEL=m, LLM_BASE_URL=base_url, LLM_API_KEY="x", HARNESS_CWD=str(mdir / "cwd"),
                   BUDGET=str(a.budget), HARNESS_LOG_DIR=str(mdir / "logs"), HARNESS_WORKERS=str(a.workers),
                   HARNESS_LIMITS="off", HARNESS_FALLBACK=a.fallback)
        procs[m] = subprocess.Popen([str(LAB / ".venv" / "bin" / "python"), str(LAB / "harness" / "run_agent.py"),
                                     "--tasks-dir", str(mdir / "tasks")], cwd=str(LAB), env=env,
                                    stdout=subprocess.DEVNULL, stderr=open(mdir / "stderr.txt", "w"), stdin=subprocess.DEVNULL)
    for p in procs.values():
        p.wait()
    rows = []
    for m in models:
        rows += grade_chunk(m, chunk_idx, chunk_dir / m / "tasks", chunk_dir / m / "logs", grading)
        if a.keep_logs:
            dst = OUT / "debug" / f"chunk{chunk_idx}{TAG}_{m}"
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir(parents=True)
            for f in glob.glob(str(chunk_dir / m / "logs" / "run-*.jsonl")):
                shutil.copy(f, dst)
            keep = []
            try:
                for line in open(chunk_dir / m / "stderr.txt", encoding="utf-8", errors="replace"):
                    if len(line) <= 400 and re.search(r"Traceback|Error|Exception|error|timeout|Timeout", line):
                        keep.append(line)
            except OSError:
                pass
            (dst / "stderr_filtered.txt").write_text("".join(keep[:400]), encoding="utf-8")
    return rows


SHIM_PORT = 18444


def shim_ready() -> bool:
    """The translating shim (tools/local_shim.py) in front of llama-server; started here if it is not up."""
    def up() -> bool:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{SHIM_PORT}/health", timeout=3)
            urllib.request.urlopen("http://127.0.0.1:8011/health", timeout=3)
            return True
        except Exception:
            return False
    if up():
        return True
    try:
        urllib.request.urlopen("http://127.0.0.1:8011/health", timeout=3)
    except Exception:
        return False
    subprocess.Popen([sys.executable, str(LAB / "local_shim.py")], stdout=open(OUT / "shim.out", "a"),
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(2)
    return up()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--chunk", type=int, default=40, help="tasks per chunk, a multiple of 4 keeps the types balanced")
    ap.add_argument("--budget", type=int, default=1_000_000_000)
    ap.add_argument("--fallback", default="on", choices=["on", "off"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--week-guard", type=int, default=600_000)
    ap.add_argument("--rpm", type=float, default=55)
    ap.add_argument("--backend", default="gateway", choices=["gateway", "local"],
                    help="local: llama-server on :8011 through the translating shim on :18444 (models local-35b, local-35b-noreason)")
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = endless; N = stop after N chunks of this invocation")
    ap.add_argument("--out-suffix", default="", help="appended to the names of state, results and log files (debug runs)")
    ap.add_argument("--keep-logs", action="store_true", help="copy the metadata journals and filtered error lines to stream/debug/")
    ap.add_argument("--backfill", action="store_true", help="fill the bucket of broken cells from the results file and exit")
    ap.add_argument("--selftest", action="store_true", help="check the sampler and grader with reference answers, no model calls")
    a = ap.parse_args()
    global TAG, TMP
    local = a.backend == "local"
    if local:
        TAG = "_local"
        TMP = Path("/tmp/hl_stream_local")
        if a.models == ",".join(ALL_MODELS):
            a.models = "local-35b-noreason,local-35b"
    TAG += a.out_suffix
    models = [m for m in a.models.split(",") if m]
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    sm = Sampler(a.seed)

    if a.selftest:
        cd = TMP / "selftest"
        g = build_chunk(sm, a.chunk, cd, keep_ref=True)
        bad = collections.Counter()
        tot = collections.Counter()
        for tid, x in g.items():
            ws = cd / "tasks" / tid / "workspace"
            ans_path = cd / "tasks" / tid / "answer.txt"
            if x["unanswerable"]:
                ok = not ans_path.exists()
            else:
                if x["check"] == "unittest":
                    if not x.get("ref"):
                        continue
                    for rel, content in x["ref"].items():
                        (ws / rel).write_text(content, encoding="utf-8")
                    ans_path.write_text("DONE")
                else:
                    ans_path.write_text(str(x["expected"]))
                ok = is_correct(x["check"], x["expected"], ans_path.read_text(), ws)
            key = f'{x["source"]}/{x["check"]}'
            tot[key] += 1
            bad[key] += 0 if ok else 1
        print({k: f"{tot[k] - bad[k]}/{tot[k]}" for k in sorted(tot)})
        print("types in order:", [g[t]["task_type"][:4] for t in list(g)[:8]])
        shutil.rmtree(cd, ignore_errors=True)
        return

    if a.backfill:
        backfill_bucket()
        return
    log(f"stream start: models={models} chunk={a.chunk} budget={a.budget} fallback={a.fallback} seed={a.seed}")
    results = OUT / f"results{TAG}.jsonl"
    base_url = f"http://127.0.0.1:{SHIM_PORT}" if local else f"http://127.0.0.1:{PROXY_PORT}"
    stop = LAB / f"STREAM_STOP{TAG}"
    n_done = 0
    while not stop.exists() and not (a.max_chunks and n_done >= a.max_chunks):
        if local:
            if not shim_ready():
                log("local shim or llama-server not reachable; waiting 60 s")
                time.sleep(60)
                continue
        elif not gateway_ready(a.rpm, a.week_guard):
            if stop.exists():
                break
            time.sleep(300)
            continue
        if process_bucket(models, a, base_url, results):
            continue          # one retry pass per loop turn; the stop file and the gateway guard are checked again
        idx = sm.state["chunks"]
        cd = TMP / f"chunk_{idx}"
        grading = build_chunk(sm, a.chunk, cd)
        t0 = time.time()
        log(f"chunk {idx}: {len(grading)} tasks, models {len(models)}")
        rows = run_chunk(models, idx, grading, cd, a, base_url)
        with open(results, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        nb = add_to_bucket(rows, grading)
        if nb:
            log(f"chunk {idx}: {nb} broken cells put in the bucket")
        sm.state["chunks"] = idx + 1
        sm.save()
        n_done += 1
        shutil.rmtree(cd, ignore_errors=True)
        by = collections.defaultdict(list)
        for r in rows:
            by[r["model"]].append(r)
        log(f"chunk {idx} done in {time.time() - t0:.0f}s: " + "; ".join(
            f"{m} hybrid {sum(x['hybrid_correct'] for x in v)}/{len(v)} cost {sum(x['hybrid_cost'] for x in v):.0f}"
            for m, v in by.items()))
    log("stream finished (stop file or chunk limit)")


if __name__ == "__main__":
    main()
