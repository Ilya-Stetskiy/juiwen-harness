#!/usr/bin/env python3
"""Stage 2 collector for the answer verifier study (runs on the GPU host, spends gateway tokens).

For a sample of tasks it runs the scripted pipeline several times (main pass, repeat, an independent method,
another model) plus an LLM judge, and writes ONLY features to the output file: booleans, numbers, agreement
flags, token costs, the correctness label. Prompts, programs and answers live in memory for the duration of
one task and are never written (competition rule 2.4).

Imports the harness read-only from ~/harness-lab/harness and `stream_run` for sampling and grading.
usage: verify_collect.py --types math,search,data_analysis --n 40 --out ~/harness-lab/verify/features.jsonl
                         [--seed 4242] [--workers 3] [--base http://127.0.0.1:18443] [--smoke]
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import contextvars
import json
import math as _m
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

HOME = Path.home()
LAB = HOME / "harness-lab"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "harness"))

import stream_run as sr                                   # noqa: E402  (sampling + grading, read-only use)
from harness import pipelines as P                         # noqa: E402
from harness.config import Config                          # noqa: E402
from harness.journal import Journal                        # noqa: E402
from harness.ledger import TokenLedger                     # noqa: E402
from harness.llm import CallContext                        # noqa: E402
from harness.llm import ask as real_ask                    # noqa: E402
from harness.predictor import Predictor                    # noqa: E402
from harness.tasks import load_tasks                       # noqa: E402
from harness.answers import extract_final                  # noqa: E402

MAIN_MODEL = "qwen3.6-35b-a3b-noreason"
OTHER_MODEL = "qwen3.8-27b-noreason"
TRACE: contextvars.ContextVar = contextvars.ContextVar("trace", default=None)


async def _ask(ctx, system, user, max_tokens):
    reply = await real_ask(ctx, system, user, max_tokens)
    tr = TRACE.get()
    if tr is not None:
        tr["replies"].append(reply)
    return reply


async def _run_python(code, cwd, timeout=30.0):
    rc, out = await real_run_python(code, cwd, timeout)
    tr = TRACE.get()
    if tr is not None:
        tr["runs"].append((code, rc, out))
    return rc, out


real_run_python = P.run_python
P.ask = _ask
P.run_python = _run_python
FORCE_VARIANT: contextvars.ContextVar = contextvars.ContextVar("force_variant", default=None)
_real_pick = P.pick


def _pick(task, kind):
    name = FORCE_VARIANT.get()
    if name:
        from harness import prompts
        for v in prompts._load().get(kind, []):
            if v["name"] == name:
                return v
    return _real_pick(task, kind)


P.pick = _pick


# ------------------------------------------------------------------ one pass
async def run_pass(task, model: str, cfg: Config, tmp: Path, mode: str = "pipeline", variant: str | None = None) -> dict:
    """Returns {answer, status, cost, calls, replies, runs}; answer None = no answer."""
    ledger = TokenLedger()
    journal = Journal(str(tmp / "j"))
    ctx = CallContext(task, model, Predictor().predict(task, model), cfg, ledger, journal)
    tr = {"replies": [], "runs": []}
    TRACE.set(tr)
    FORCE_VARIANT.set(variant)
    status, ans = "failed", None
    try:
        if mode == "cot":                      # independent method for math: reasoning in text, no program
            q = P.clean_instruction(task.instruction)
            reply = await _ask(ctx, "Solve the problem. Reason briefly step by step (at most 150 words), then end "
                                    "with one line: FINAL: <integer answer>", q, 600)
            f = extract_final(reply, fallback=False)
            if f:
                v = sr._num(f)
                status, ans = "solved", (str(int(v)) if v is not None and float(v).is_integer() else f.strip())
        else:
            res = await P.PIPELINES[task.task_type](ctx)
            if res is not None:
                status, ans = res
    except Exception as exc:                  # noqa: BLE001
        status = "error:" + type(exc).__name__
    journal.close()
    return {"answer": ans, "status": status, "cost": ledger.task_cost(task.task_id), "calls": ledger.calls,
            "replies": tr["replies"], "runs": tr["runs"], "ctx": ctx}


# ------------------------------------------------------------------ free features
def _eq(check: str, a, b, ws) -> bool:
    """Agreement of two answers (None = no answer)."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return bool(sr.is_correct(check if check not in ("unanswerable", "unittest") else "exact_str", b, a, ws))
    except Exception:                          # noqa: BLE001
        return a.strip() == b.strip()


def code_features(runs) -> dict:
    f = {"n_runs": len(runs), "prog_ok": 0, "prog_nan": 0, "hardcode": 0, "code_len": 0, "n_calls_code": 0,
         "n_ops": 0, "retry_used": int(len(runs) > 1)}
    if not runs:
        return f
    code, rc, out = runs[-1]
    f["prog_ok"] = int(rc == 0)
    f["prog_nan"] = int(bool(re.search(r"\bnan\b", out, re.I)))
    f["code_len"] = len(code)
    f["out_lines"] = len([l for l in out.splitlines() if l.strip()])
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f
    prints = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print"]
    ops = [n for n in ast.walk(tree) if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp))]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") != "print"]
    f["n_ops"], f["n_calls_code"] = len(ops), len(calls)
    f["hardcode"] = int(bool(prints) and all(all(isinstance(a, (ast.Constant, ast.JoinedStr)) for a in p.args) for p in prints)
                        and not ops and not calls)
    f["print_literal_only"] = f["hardcode"]
    return f


def fmt_ok(check: str, ans, instruction: str) -> int:
    if ans is None:
        return 0
    a = ans.strip()
    if check == "exact_int":
        return int(bool(re.fullmatch(r"-?\d+", a)))
    if check == "dabench_format":
        want = set(k.lower() for k in re.findall(r"@([\w\-]+)\[", instruction))
        got = set(k.lower() for k in re.findall(r"@([\w\-]+)\[", a))
        return int(bool(want) and want <= got)
    return int(0 < len(a) <= 120)


def norm_text(s: str) -> str:
    return sr._norm(s)


def read_docs(ws) -> str:
    if ws is None:
        return ""
    out = []
    for p in sorted(Path(ws).rglob("*")):
        if p.is_file():
            try:
                out.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return "\n".join(out)


def extract_feats(task, ans, docs_norm: str) -> dict:
    f = {"ans_len": len(ans) if ans else 0}
    if ans is None:
        return f
    n = norm_text(ans)
    f["ans_in_docs"] = int(bool(n) and n in docs_norm)
    if task.task_type == "math":
        v = sr._num(ans)
        f["ans_int"] = int(v is not None and float(v).is_integer())
        f["ans_log10"] = round(_m.log10(abs(v) + 1), 2) if v is not None else None
        f["ans_neg"] = int(v is not None and v < 0)
    f["ans_numeric"] = int(sr._num(ans) is not None)
    return f


# ------------------------------------------------------------------ judge
JUDGE = {
    "search": ("You verify answers. Given a question, documents and a candidate answer, decide whether the documents "
               "support the candidate as the answer. Reply on line 1 with YES or NO. On line 2 give the shortest "
               "fragment copied verbatim from the documents that supports it (empty if NO)."),
    "data_analysis": ("You verify answers. Given a question about a table, the table overview, the Python program that "
                      "was run and the answer it printed, decide whether the program correctly answers the question "
                      "and the printed answer is in the requested format. Reply with one word: YES or NO."),
    "math": ("You verify answers. Given a math problem, the Python program that was run and the answer it printed, "
             "decide whether the program really computes the answer from the problem (not a hard-coded guess) and the "
             "answer is plausible. Reply with one word: YES or NO."),
}


async def judge(task, main: dict, cfg: Config, tmp: Path, docs: str) -> dict:
    ledger = TokenLedger()
    journal = Journal(str(tmp / "jj"))
    ctx = CallContext(task, MAIN_MODEL, Predictor().predict(task, MAIN_MODEL), cfg, ledger, journal)
    q = P.clean_instruction(task.instruction)
    ans = main["answer"] if main["answer"] is not None else "(no answer)"
    if task.task_type == "search":
        user = f"Question: {q}\n\nDocuments:\n{docs[:P.MAX_CONTEXT_CHARS]}\n\nCandidate answer: {ans}"
        mt = 120
    else:
        code = main["runs"][-1][0] if main["runs"] else "(none)"
        extra = f"\n\nTable overview:\n{P._data_overview(task.workspace)}" if task.task_type == "data_analysis" else ""
        user = f"Question: {q}{extra}\n\nProgram:\n{code}\n\nPrinted answer: {ans}"
        mt = 10
    try:
        reply = await real_ask(ctx, JUDGE[task.task_type], user, mt)
    except Exception:                          # noqa: BLE001
        journal.close()
        return {"judge_yes": None, "judge_cost": ledger.task_cost(task.task_id)}
    journal.close()
    lines = [l for l in reply.strip().splitlines() if l.strip()]
    yes = int(bool(lines) and lines[0].strip().upper().startswith("YES"))
    out = {"judge_yes": yes, "judge_cost": ledger.task_cost(task.task_id)}
    if task.task_type == "search":
        quote = norm_text(lines[1]) if len(lines) > 1 else ""
        out["quote_in_docs"] = int(len(quote) >= 3 and quote in norm_text(docs))
        out["quote_has_ans"] = int(main["answer"] is not None and norm_text(main["answer"]) in quote)
    return out


# ------------------------------------------------------------------ orchestration
def sample(types, n, seed):
    rng = random.Random(seed)
    cache = {}
    rows = []
    for t in types:
        srcs = sr.SOURCES[t]
        picked, tries = [], 0
        while len(picked) < n and tries < 20 * n:
            tries += 1
            src = rng.choice(srcs)
            if t == "math" and rng.random() < 0.6:           # oversample the harder MATH sources (more wrong answers)
                src = rng.choice([x for x in srcs if x.startswith("math")])
            offs = cache.setdefault(src, json.load(open(sr.LAB / "index" / f"{src}.json")))
            r = sr.load_row(src, rng.choice(offs))
            if r["task_id"] in {x[0]["task_id"] for x in picked}:
                continue
            picked.append((r, src))
        rows += picked
    return rows


async def judge2(task, cfg: Config, tmp: Path, docs: str) -> dict:
    """Search only: can the question be answered from the documents at all? (no candidate answer shown)"""
    ledger = TokenLedger()
    journal = Journal(str(tmp / "jj2"))
    ctx = CallContext(task, MAIN_MODEL, Predictor().predict(task, MAIN_MODEL), cfg, ledger, journal)
    q = P.clean_instruction(task.instruction)
    try:
        reply = await real_ask(ctx, "Decide whether the question is answered EXPLICITLY by the documents. Reply with one "
                                    "word: YES if the documents state the answer, NO if they do not (related facts only, "
                                    "or the information is missing).", f"Question: {q}\n\nDocuments:\n{docs[:P.MAX_CONTEXT_CHARS]}", 10)
    except Exception:                          # noqa: BLE001
        journal.close()
        return {"answerable_yes": None, "judge2_cost": ledger.task_cost(task.task_id)}
    journal.close()
    return {"answerable_yes": int(reply.strip().upper().startswith("YES")), "judge2_cost": ledger.task_cost(task.task_id)}


async def one_task(r, src, root: Path, cfg: Config, out_path: Path, sem, tag: str, passes: set):
    async with sem:
        tid = r["task_id"]
        parent = root / "t" / tid
        d = parent / tid
        shutil.rmtree(parent, ignore_errors=True)
        sr.write_task_dir(r, d)
        task = load_tasks(parent)[0]
        tmp = root / "tmp" / tid
        tmp.mkdir(parents=True, exist_ok=True)
        ws = task.workspace
        docs = read_docs(ws)
        docs_norm = norm_text(docs)
        t0 = time.time()
        try:
            main = await run_pass(task, MAIN_MODEL, cfg, tmp)
            rep = await run_pass(task, MAIN_MODEL, cfg, tmp) if "rep" in passes else None
            other = await run_pass(task, OTHER_MODEL, cfg, tmp) if "other" in passes else None
            indep = None
            if "indep" not in passes:
                pass
            elif task.task_type == "math":
                indep = await run_pass(task, MAIN_MODEL, cfg, tmp, mode="cot")
            elif task.task_type == "data_analysis":
                indep = await run_pass(task, MAIN_MODEL, cfg, tmp, variant="v1")   # different prompt, no retry
            jd = await judge(task, main, cfg, tmp, docs) if "judge" in passes else {}
            if "judge2" in passes and task.task_type == "search":
                jd.update(await judge2(task, cfg, tmp, docs))
        except Exception as exc:               # noqa: BLE001
            print("ERR", tid, type(exc).__name__, str(exc)[:120], flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(parent, ignore_errors=True)
            return
        exp = r.get("expected")
        unans = r["check"] == "unanswerable"
        ans = main["answer"]
        label = (ans is None) if unans else (ans is not None and sr.is_correct(r["check"], exp, ans, ws))
        row = {"task_id": tid, "type": task.task_type, "source": src, "check": r["check"], "unanswerable": unans,
               "reward": task.reward, "instr_len": len(task.instruction), "ws_bytes": task.workspace_bytes,
               "n_files": task.n_files, "main_status": main["status"], "answered": ans is not None,
               "label": bool(label), "cost_main": round(main["cost"], 1), "calls_main": main["calls"],
               "cost_rep": round(rep["cost"], 1) if rep else None, "cost_other": round(other["cost"], 1) if other else None,
               "cost_indep": round(indep["cost"], 1) if indep else None, "seconds": round(time.time() - t0, 1)}
        row.update(code_features(main["runs"]))
        row["fmt_ok"] = fmt_ok(r["check"], ans, task.instruction)
        row.update(extract_feats(task, ans, docs_norm))
        lab = lambda x: bool((x["answer"] is None) if unans else (x["answer"] is not None and sr.is_correct(r["check"], exp, x["answer"], ws)))
        if rep:
            row["agree_rep"] = int(_eq(r["check"], rep["answer"], ans, ws))
            row["rep_answered"] = int(rep["answer"] is not None)
            row["label_rep"] = lab(rep)
        if other:
            row["agree_other"] = int(_eq(r["check"], other["answer"], ans, ws))
            row["other_answered"] = int(other["answer"] is not None)
            row["label_other"] = lab(other)
        if indep is not None:
            row["agree_indep"] = int(_eq(r["check"], indep["answer"], ans, ws))
        answers = [x["answer"] for x in (main, rep, other, indep) if x is not None]
        row["n_distinct"] = len({norm_text(x) if x is not None else None for x in answers})
        row["n_agree_main"] = sum(_eq(r["check"], x, ans, ws) for x in answers[1:])
        row.update(jd)
        row["cost_total"] = round(sum(x["cost"] for x in (main, rep, other, indep) if x) + jd.get("judge_cost", 0) + jd.get("judge2_cost", 0), 1)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(parent, ignore_errors=True)      # task copy with its workspace (answers never written anyway)
        print(f"ok {tid} label={row['label']} cost={row['cost_total']}", flush=True)


async def amain(a):
    cfg = Config.from_env()
    out = Path(os.path.expanduser(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["task_id"] for l in open(out) if l.strip()}
    root = Path("/tmp/hl_verify")
    (root / "t").mkdir(parents=True, exist_ok=True)
    nbt = dict(kv.split("=") for kv in a.n_by_type.split(",") if kv)
    rows = []
    for t in a.types.split(","):
        rows += sample([t], int(nbt.get(t, a.n)), a.seed)
    rows = [(r, s) for r, s in rows if r["task_id"] not in done]
    print(f"{len(rows)} tasks to run", flush=True)
    sem = asyncio.Semaphore(a.workers)
    await asyncio.gather(*(one_task(r, s, root, cfg, out, sem, "", set(a.passes.split(","))) for r, s in rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="math,search,data_analysis")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-by-type", default="")      # e.g. math=50,search=50,data_analysis=40
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--passes", default="rep,other,indep,judge")   # subset of rep,other,indep,judge,judge2
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="~/harness-lab/verify/features.jsonl")
    a = ap.parse_args()
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:18443")
    os.environ.setdefault("LLM_API_KEY", "x")
    os.environ.setdefault("HARNESS_LOG_DIR", "/tmp/hl_verify/logs")
    os.environ["BUDGET"] = "1000000000"
    os.makedirs("/tmp/hl_verify/cwd", exist_ok=True)
    os.chdir("/tmp/hl_verify/cwd")          # openjiuwen drops memory/log dirs into the cwd
    asyncio.run(amain(a))
