#!/usr/bin/env python3
"""Dataset for the cost predictor study (runs on the host, harness-lab venv: needs harness imports, tokenizers, numpy, safetensors).

One row per unique task of the stream results (11k result rows = ~5k tasks x models): features that exist BEFORE the task is
solved -- type, source, reward, workspace size, exact Qwen token counts of the instruction and of the prompt material, a static
sentence embedding (potion-base, numpy only). Outputs a feature file only (no task text on disk).

usage: predictor_dataset.py --out ~/harness-lab/verify/pred_data.npz
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
LAB = HOME / "harness-lab"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "harness"))
import stream_run as sr                                    # noqa: E402
import harness.pipelines as P                              # noqa: E402
from harness.tasks import load_tasks                       # noqa: E402
from tokenizers import Tokenizer                           # noqa: E402

QTOK = Tokenizer.from_file(os.environ.get("QWEN_TOKENIZER", "/tmp/qtok/Qwen3.6-35B-A3B/tokenizer.json"))
POTION = Path(os.environ.get("POTION_DIR", "/tmp/qtok/potion-base-8M"))


def potion_loader():
    from safetensors.numpy import load_file
    tk = Tokenizer.from_file(str(POTION / "tokenizer.json"))
    tensors = load_file(str(POTION / "model.safetensors"))
    key = "embeddings" if "embeddings" in tensors else list(tensors)[0]
    emb = tensors[key].astype(np.float32)
    weights = tensors.get("weights")

    def encode(text: str) -> np.ndarray:
        ids = tk.encode(text[:20000], add_special_tokens=False).ids[:1000]
        if not ids:
            return np.zeros(emb.shape[1], np.float32)
        v = emb[ids]
        if weights is not None:
            v = v * weights[ids][:, None]
        return v.mean(0)
    return encode, emb.shape[1]


def ntok(text: str) -> int:
    return len(QTOK.encode(text).ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LAB / "verify" / "pred_data.npz"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(sr.OUT / "results.jsonl") if l.strip()]
    ids = {}
    for r in rows:
        ids.setdefault((r["source"], r["task_id"]), r)
    by_src: dict[str, set] = {}
    for (src, tid) in ids:
        by_src.setdefault(src, set()).add(tid)
    pat = re.compile(rb'^\{"task_id": "([^"]+)"')
    offs: dict[tuple, int] = {}
    for src, want in by_src.items():
        pos = 0
        with open(sr.SRC_PATH[src], "rb") as f:
            for line in f:
                m = pat.match(line)
                if m and m.group(1).decode() in want:
                    offs[(src, m.group(1).decode())] = pos
                pos += len(line)
        print(src, len(want), "found", sum(1 for k in offs if k[0] == src), flush=True)
    enc, dim = potion_loader()
    root = Path("/tmp/preddata")
    shutil.rmtree(root, ignore_errors=True)
    out = {"task_id": [], "source": [], "type": [], "reward": [], "ws_bytes": [], "n_files": [], "tok_instr": [], "tok_prompt": [], "emb": []}
    for i, ((src, tid), off) in enumerate(sorted(offs.items())):
        r = sr.load_row(src, off)
        d = root / "t"
        shutil.rmtree(d, ignore_errors=True)
        sr.write_task_dir(r, d / tid)
        task = load_tasks(d)[0]
        q = P.clean_instruction(task.instruction)
        material = q
        if task.task_type == "search":
            material = q + "\n\nDocuments:\n" + (P._read_kb(task.workspace, q) or "")
        elif task.task_type == "data_analysis" and task.workspace is not None:
            material = q + "\n\n" + P._data_overview(task.workspace)
        out["task_id"].append(tid)
        out["source"].append(src)
        out["type"].append(task.task_type)
        out["reward"].append(float(task.reward))
        out["ws_bytes"].append(int(task.workspace_bytes))
        out["n_files"].append(sum(1 for _ in Path(task.workspace).rglob("*") if _.is_file()) if task.workspace else 0)
        out["tok_instr"].append(ntok(q))
        out["tok_prompt"].append(ntok(material))
        out["emb"].append(enc(q))
        if i % 500 == 0:
            print(i, flush=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()})
    print("saved", a.out, len(out["task_id"]))


if __name__ == "__main__":
    main()
