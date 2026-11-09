"""Guard (S12): prompt-injection and secret-leak checks. Skeleton with one trivial variant; OFF by default.

    HARNESS_GUARD=off|log|strip      off: nothing (default), log: scan and write `guard` events, strip: also cut the
                                     flagged sentences out of the text the model is shown (instruction, documents,
                                     data overview); the files on disk are never changed.

Inputs: free text (instruction, a document). Output: list of (kind, snippet). The patterns are a first guess, a
placeholder for a real detector; the lab measures what they catch (adversarial set) and what they break (false hits on
ordinary tasks). `scrub_answer` removes values of the secret environment variables from an answer before it is written.
"""
from __future__ import annotations

import os
import re

PATTERNS: dict[str, re.Pattern] = {
    "override": re.compile(
        r"ignore (?:all |any |the |your )?(?:previous|above|prior|earlier|other) (?:instructions|rules|messages)"
        r"|disregard (?:all |any |the )?(?:previous|above|prior|earlier)?\s*(?:instructions|rules)"
        r"|new instructions?\s*:|you are now\b|system prompt|developer message", re.I),
    "answer_file": re.compile(r"(?:write|save|put|output|store)[^.\n]{0,60}answer\.txt|answer\.txt[^.\n]{0,40}(?:instead|overwrite)", re.I),
    "secret": re.compile(r"LLM_API_KEY|LLM_BASE_URL|printenv|os\.environ|getenv|/proc/self|api[ _-]?key|\benvironment variables?\b|\bBUDGET\b", re.I),
    "escape": re.compile(r"\.\./|/etc/|/proc/|/root\b|/home/|~/\.|\bssh\b.{0,20}key", re.I),
    "exfil": re.compile(r"\b(?:curl|wget|netcat)\b|requests\.(?:get|post)|urllib\.request|socket\.connect|https?://\S+\?\S*=", re.I),
}
# the task's own boilerplate legitimately mentions answer.txt, so the instruction is not checked for that kind
INSTRUCTION_KINDS = ("override", "secret", "escape", "exfil")
SECRET_ENV = ("LLM_API_KEY", "LLM_BASE_URL", "OPENAI_API_KEY")


def mode() -> str:
    m = os.environ.get("HARNESS_GUARD", "off").strip().lower()
    return m if m in ("off", "log", "strip") else "off"


def scan(text: str, kinds=None) -> list[tuple[str, str]]:
    hits = []
    for kind, rx in PATTERNS.items():
        if kinds is not None and kind not in kinds:
            continue
        for m in rx.finditer(text or ""):
            hits.append((kind, m.group(0)[:80]))
    return hits


def sanitize(text: str, kinds=None) -> str:
    """strip mode: drop the sentences (or lines) that contain a hit; identity in the other modes."""
    if mode() != "strip" or not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    kept = [p for p in parts if not scan(p, kinds)]
    return "\n".join(kept) if len(kept) != len(parts) else text


def inspect_task(task, journal) -> None:
    """log/strip modes: write one `guard` event per place that has hits (never the text itself)."""
    if mode() == "off":
        return
    places: dict[str, list[str]] = {}
    h = scan(task.instruction, INSTRUCTION_KINDS)
    if h:
        places["instruction"] = sorted({k for k, _ in h})
    ws = getattr(task, "workspace", None)
    if ws is not None:
        for p in sorted(ws.rglob("*")):
            if not p.is_file() or p.stat().st_size > 2_000_000:
                continue
            try:
                h = scan(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            if h:
                places[f"file:{p.name}"] = sorted({k for k, _ in h})
    if places:
        journal.event("guard", task=task.task_id, places=places, action=mode())


def scrub_answer(answer: str | None, task_id: str, journal) -> str | None:
    """Remove secret values from an answer (any mode except off)."""
    if mode() == "off" or not answer:
        return answer
    out = answer
    for name in SECRET_ENV:
        val = os.environ.get(name, "")
        if len(val) >= 6 and val in out:
            out = out.replace(val, "[removed]")
            journal.event("guard", task=task_id, places={"answer": ["secret_value"]}, action="scrub")
    return out
