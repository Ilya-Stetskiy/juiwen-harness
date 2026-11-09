#!/usr/bin/env python3
"""Offline collector for the search verification level (no gateway tokens: the LOCAL model through local_shim).

Per search task: main pass (the scripted search pipeline), free source features (harness/search_check.py), and the second
independent path (shortlisted files by question overlap + the `short` prompt). Only FEATURES and labels are written
(competition rule 2.4: no prompts, programs or answers on disk); answers live in memory for one task.

usage (host): LLM_BASE_URL=http://127.0.0.1:18444 python3 verify_search_collect.py --n 300 --seed 9300 --workers 3 \
        --exclude ~/harness-lab/verify/exclude_ids.txt --out ~/harness-lab/verify/search_a.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_collect as VC                      # noqa: E402
import harness.pipelines as P                    # noqa: E402
from harness import search_check as SC           # noqa: E402
from harness import prompts                      # noqa: E402
from harness.answers import extract_final        # noqa: E402
from harness.llm import CallContext              # noqa: E402

LOCAL = "local-35b-noreason"


async def second_path(task, cfg, tmp: Path, files):
    """Independent path: shortlisted files only, another prompt (`short`), the question restated. Returns (answer|None, tokens)."""
    short = shortlist = SC.shortlist(P.clean_instruction(task.instruction), files, 4)
    kb = "\n\n".join(f"## {n}\n{t.strip()}" for n, t in short)
    variant = next((v for v in prompts._load().get("search", []) if v["name"] == "short"), None) or prompts._load()["search"][-1]
    q = P.clean_instruction(task.instruction)
    ledger = VC.TokenLedger()
    journal = VC.Journal(str(tmp / "j2"))
    ctx = CallContext(task, LOCAL, VC.Predictor().predict(task, LOCAL), cfg, ledger, journal)
    VC.TRACE.set({"replies": [], "runs": []})
    try:
        reply = await VC._ask(ctx, variant["system"], f"Question: {q}\n\nDocuments:\n{kb}", variant.get("max_tokens", 400))
    except Exception:  # noqa: BLE001
        journal.close()
        return "error", 0.0
    journal.close()
    final = extract_final(reply, fallback=False)
    if final is None or final.upper().startswith("UNSOLVABLE"):
        return None, ledger.task_cost(task.task_id)
    return P.normalize(task.check, final), ledger.task_cost(task.task_id)


async def one_task(r, src, root: Path, cfg, out_path: Path, sem):
    async with sem:
        tid = r["task_id"]
        parent = root / "t" / tid
        shutil.rmtree(parent, ignore_errors=True)
        VC.sr.write_task_dir(r, parent / tid)
        task = VC.load_tasks(parent)[0]
        tmp = root / "tmp" / tid
        tmp.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            exp, unans = r.get("expected"), r["check"] == "unanswerable"
            res = await VC.run_pass(task, LOCAL, cfg, tmp)
            ans = res["answer"]
            files = SC.read_files(task.workspace)
            q = P.clean_instruction(task.instruction)
            feats = SC.source_features(q, ans, files)
            d, why = SC.doubt(feats)
            a2, cost2 = await second_path(task, cfg, tmp, files)
            ws = task.workspace
            label = (ans is None) if unans else (ans is not None and VC.sr.is_correct(r["check"], exp, ans, ws))
            if a2 == "error":
                label2, agree = None, None
            else:
                label2 = (a2 is None) if unans else (a2 is not None and VC.sr.is_correct(r["check"], exp, a2, ws))
                agree = int(VC._eq(r["check"], ans, a2, ws))
            row = {"task_id": tid, "source": src, "check": r["check"], "unanswerable": unans, "reward": task.reward,
                   "label": bool(label), "answered": ans is not None, "status": res["status"],
                   "cost_main": round(res["cost"], 1), "n_files": len(files), "ws_bytes": task.workspace_bytes,
                   "doubt": int(d), "doubt_why": why, "agree2": agree, "label2": None if label2 is None else bool(label2),
                   "answered2": a2 is not None and a2 != "error", "cost2": round(cost2, 1), "seconds": round(time.time() - t0, 1)}
            row.update(feats)
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"ok {tid} label={int(label)} doubt={int(d)} agree2={agree} sec={time.time() - t0:.0f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("ERR", tid, type(exc).__name__, str(exc)[:100], flush=True)
        finally:
            shutil.rmtree(parent, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)


async def amain(a):
    cfg = VC.Config.from_env()
    out = Path(os.path.expanduser(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["task_id"] for x in open(out) if x.strip()}
    root = Path("/tmp/hl_verify_search")
    (root / "t").mkdir(parents=True, exist_ok=True)
    exclude = set()
    if a.exclude:
        exclude = {x.strip() for x in open(os.path.expanduser(a.exclude)) if x.strip()}
    rows = [(r, s) for r, s in VC.sample(["search"], a.n, a.seed) if r["task_id"] not in done and r["task_id"] not in exclude]
    print(f"{len(rows)} search tasks", flush=True)
    sem = asyncio.Semaphore(a.workers)
    await asyncio.gather(*(one_task(r, s, root, cfg, out, sem) for r, s in rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=9300)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", default="~/harness-lab/verify/search_a.jsonl")
    a = ap.parse_args()
    os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:18444")
    os.environ.setdefault("LLM_API_KEY", "x")
    os.environ.setdefault("HARNESS_LOG_DIR", "/tmp/hl_verify_search/logs")
    os.environ["BUDGET"] = "1000000000"
    os.makedirs("/tmp/hl_verify_search/cwd", exist_ok=True)
    os.chdir("/tmp/hl_verify_search/cwd")
    asyncio.run(amain(a))
