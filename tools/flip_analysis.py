"""Compare what the model wrote for the same task in different runs (lab tool, reads raw openjiuwen logs).

usage: python flip_analysis.py <llm.log> <tasks_dir> <task_id> [<task_id> ...]
Prints for every logged call whose user message contains the task question: the system prompt (first
80 chars) and the model reply (code), oldest first.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

CLEAN = re.compile(r"(answer\.txt|\bbash\b|current directory|python3|file tools|Work only)", re.IGNORECASE)


def question_of(instr: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", instr.strip())
    return " ".join(p for p in parts if not CLEAN.search(p)).strip()


_CONTENT = re.compile(r"ChatCompletionMessage\(content=(['\"])(.*?)\1, refusal=", re.DOTALL)


def reply_of(event: dict) -> str:
    """The reply text is inside the repr of the ChatCompletion stored in metadata.response."""
    resp = str((event.get("metadata") or {}).get("response", ""))
    m = _CONTENT.search(resp)
    if not m:
        return ""
    try:
        return ast.literal_eval(m.group(1) + m.group(2) + m.group(1))
    except (ValueError, SyntaxError):
        return m.group(2)


def main(log: str, tasks_dir: str, ids: list[str]) -> None:
    events = []
    for line in open(log, encoding="utf-8", errors="replace"):
        if '"llm_call_end"' not in line or '"messages"' not in line:
            continue
        i = line.find("| INFO | ")
        if i < 0:
            continue
        try:
            events.append(json.loads(line[i + 9:]))
        except ValueError:
            continue
    for tid in ids:
        instr = json.load(open(Path(tasks_dir) / tid / "task.json", encoding="utf-8"))["instruction"]
        q = question_of(instr)
        hits = [e for e in events if any(q[:80] in str(m.get("content", "")) for m in e.get("messages", []))]
        print(f"=== {tid}: {len(hits)} logged calls | question: {q[:140]!r}")
        for e in hits:
            msgs = e["messages"]
            system = next((m["content"] for m in msgs if m.get("role") == "system"), "")
            print(f"--- {e.get('timestamp', '')[:19]} system={system[:70]!r} n_msgs={len(msgs)}")
            print(reply_of(e)[:900])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
