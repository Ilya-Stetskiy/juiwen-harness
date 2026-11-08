"""Scripted pipelines: the harness gathers the data, the model only answers.

Instead of letting an agent explore with tools (every tool schema costs tokens on every
step), the harness prepares the context itself, makes one or two plain model calls and
executes any code the model writes. A pipeline returns (status, answer) or None when it
cannot handle the task (the executor then falls back to the tool agent if enabled).
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .answers import extract_final, normalize
from .contradiction import find_contradictions
from .llm import CallContext, ask
from .prompts import pick

_DROP = re.compile(r"(answer\.txt|\bbash\b|current directory|python3|file tools|Work only)", re.IGNORECASE)
_CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_FAILED = re.compile(r"FAILED \(([^)]*)\)")
MAX_CONTEXT_CHARS = 40_000


# Boilerplate often follows the question without a full stop ("...statement text Use bash with python3 to check it."),
# which glued the question to the boilerplate and got it dropped. Start a new sentence before each boilerplate phrase.
_BOUNDARY = re.compile(
    r"(?<![.!?])\s+(?=(?:Use bash|Use the file tools|Compute it with the bash|Work only|When you are done|Write ONLY)\b)")
_KEEP = re.compile(r"@\w+\[|\bformat\b", re.IGNORECASE)   # answer-format sentences carry required information


def clean_instruction(text: str) -> str:
    """Drop sentences about tools, the working directory and answer.txt: the harness handles those."""
    text = _BOUNDARY.sub(". ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [p for p in parts if not _DROP.search(p) or _KEEP.search(p)]
    return " ".join(kept).strip() or text.strip()


def extract_code(reply: str) -> str:
    m = _CODE.findall(reply)
    return (m[-1] if m else reply).strip()


async def run_python(code: str, cwd: str, timeout: float = 30.0) -> tuple[int, str]:
    def _run() -> tuple[int, str]:
        try:
            r = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True, timeout=timeout)
            return r.returncode, (r.stdout or "") + (("\n" + r.stderr) if r.returncode else "")
        except subprocess.TimeoutExpired:
            return 124, "timeout"
    return await asyncio.to_thread(_run)


def _last_line(text: str) -> str | None:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


async def _program_answer(ctx: CallContext, system: str, user: str, cwd: str, max_tokens: int,
                          retry: bool = True, multiline: bool = False) -> str | None:
    """Model writes a program, the harness runs it. Optional retry that shows the model its own error."""
    prev_code = prev_out = None
    for attempt in range(2 if retry else 1):
        prompt = user
        if prev_code is not None:
            prompt = (f"{user}\n\nYour previous program:\n```python\n{prev_code}\n```\n"
                      f"It did not produce a usable answer. Its output:\n{(prev_out or '')[-600:]}\nFix the program.")
        reply = await ask(ctx, system, prompt, max_tokens)
        code = extract_code(reply)
        rc, out = await run_python(code, cwd)
        if rc == 0 and multiline:
            # an answer made of several @name[value] lines: every printed line belongs to it
            lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
            ans = "\n".join(lines) if lines else None
        else:
            ans = _last_line(out) if rc == 0 else None
        if ans is not None and ans.lower() not in ("nan", "none", ""):
            return ans
        prev_code, prev_out = code, out
    return None


async def math(ctx: CallContext):
    q = clean_instruction(ctx.task.instruction)
    v = pick(ctx.task, "math")
    ctx.journal.event("variant", task=ctx.task.task_id, pipeline="math", name=v["name"])
    tmp = tempfile.mkdtemp(prefix="m_")
    try:
        ans = await _program_answer(ctx, v["system"], q, tmp, v.get("max_tokens", 700), v.get("retry", True))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return None if ans is None else ("solved", normalize(ctx.task.check, ans))


def _data_overview(ws: Path) -> str:
    """Name, size, columns with example values and the first rows of every file (a few hundred tokens)."""
    import csv

    chunks = []
    for p in sorted(ws.iterdir()):
        if not p.is_file():
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        head = f"File {p.name}: {len(lines)} lines."
        if p.suffix.lower() == ".csv" and lines:
            try:
                rows = list(csv.reader(lines[:300]))
                cols, body = rows[0], rows[1:]
                ex = []
                for i, c in enumerate(cols):
                    seen: list[str] = []
                    for r in body:
                        v = r[i].strip() if i < len(r) else ""
                        if v and v[:24] not in seen:
                            seen.append(v[:24])
                        if len(seen) == 3:
                            break
                    ex.append(f"{c} (e.g. {', '.join(seen)})")
                head += " Columns: " + "; ".join(ex) + "\nFirst rows:\n"
                head += "\n".join(",".join(x[:40] for x in r) for r in rows[:7])
            except (csv.Error, IndexError):
                head += " First lines:\n" + "\n".join(lines[:6])
        else:
            head += " First lines:\n" + "\n".join(lines[:6])
        chunks.append(head[:3000])
    return "\n\n".join(chunks)


async def data_analysis(ctx: CallContext):
    ws = ctx.task.workspace
    if ws is None:
        return None
    q = clean_instruction(ctx.task.instruction)
    v = pick(ctx.task, "data_analysis")
    ctx.journal.event("variant", task=ctx.task.task_id, pipeline="data_analysis", name=v["name"])
    ans = await _program_answer(ctx, v["system"], f"{q}\n\n{_data_overview(ws)}", str(ws),
                                v.get("max_tokens", 800), v.get("retry", True), v.get("multiline", False))
    return None if ans is None else ("solved", normalize(ctx.task.check, ans))


def _read_kb(ws: Path, question: str) -> str | None:
    files = sorted(p for p in ws.rglob("*") if p.is_file())
    docs = []
    for p in files:
        try:
            docs.append((p, p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    total = sum(len(t) for _, t in docs)
    if total > MAX_CONTEXT_CHARS:
        # large base: keep files that mention a capitalised word of the question
        words = {w for w in re.findall(r"[A-Z][a-z]+", question)}
        docs = [(p, t) for p, t in docs if any(w in t or w.lower() in p.name.lower() for w in words)]
        if sum(len(t) for _, t in docs) > MAX_CONTEXT_CHARS:
            return None
    if not docs:
        return None
    return "\n\n".join(f"## {p.name}\n{t.strip()}" for p, t in docs)


async def search(ctx: CallContext):
    ws = ctx.task.workspace
    if ws is None:
        return None
    q = clean_instruction(ctx.task.instruction)
    kb = _read_kb(ws, q)
    if kb is None:
        return None
    v = pick(ctx.task, "search")
    ctx.journal.event("variant", task=ctx.task.task_id, pipeline="search", name=v["name"])
    reply = await ask(ctx, v["system"], f"{q}\n\nDocuments:\n{kb}", v.get("max_tokens", 400))
    final = extract_final(reply, fallback=bool(v.get("final_fallback", False)))
    if final is None:
        return None
    if final.upper().startswith("UNSOLVABLE"):
        return "unsolvable", None
    return "solved", normalize(ctx.task.check, final)


async def _run_tests(cmd_file: str, ws: Path) -> tuple[bool, str]:
    def _run() -> tuple[bool, str]:
        try:
            r = subprocess.run([sys.executable, cmd_file], cwd=str(ws), capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, "tests timed out"
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0, out[-3000:]
    return await asyncio.to_thread(_run)


def fail_count(output: str) -> int | None:
    """Number of failing tests from a unittest summary line, None if it cannot be read."""
    m = _FAILED.search(output)
    if not m:
        return None
    return sum(int(n) for n in re.findall(r"=(\d+)", m.group(1))) or None


async def coding(ctx: CallContext, rounds: int = 2):
    ws = ctx.task.workspace
    if ws is None:
        return None
    instr = ctx.task.instruction
    m_test = re.search(r"python3?\s+(\S+\.py)", instr)
    m_target = re.search(r"bugs in\s+`?(\S+?\.py)`?", instr)
    if not (m_test and m_target):
        return None
    test_file, target = m_test.group(1), m_target.group(1)
    tpath, spath = ws / test_file, ws / target
    if not (tpath.is_file() and spath.is_file()):
        return None
    if os.environ.get("HARNESS_CONTRADICTION", "on").lower() != "off":
        found = find_contradictions(tpath.read_text(encoding="utf-8", errors="replace"))
        if found:  # no implementation can pass these tests: no model call, no answer, no fallback
            ctx.journal.event("contradiction", task=ctx.task.task_id, detail=found[0][:200])
            return "unsolvable", None
    original = spath.read_text(encoding="utf-8")
    passed, out = await _run_tests(test_file, ws)
    if passed:
        return "solved", "DONE"
    tests_src = tpath.read_text(encoding="utf-8")
    prev_fails = fail_count(out)
    for _ in range(rounds):
        src = spath.read_text(encoding="utf-8")
        user = (f"Fix the bugs in {target}. Do not change the tests.\n\n{target}:\n```python\n{src}\n```\n\n"
                f"{test_file}:\n```python\n{tests_src}\n```\n\nTest output:\n{out}")
        reply = await ask(ctx, f"Return the complete corrected content of {target} in one ```python block and "
                               "nothing else. If a failing test contradicts the documented behaviour or another "
                               "test, so that no correct implementation can pass it, reply only: UNSOLVABLE",
                          user, 2500)
        if reply.strip().upper().startswith("UNSOLVABLE"):
            spath.write_text(original, encoding="utf-8")
            return "unsolvable", None
        fixed = extract_code(reply)
        if not fixed:
            break
        spath.write_text(fixed + "\n", encoding="utf-8")
        passed, out = await _run_tests(test_file, ws)
        if passed:
            return "solved", "DONE"
        cur = fail_count(out)
        # early exit: a second round that starts from no progress rarely helps and costs ~2x
        if prev_fails is not None and cur is not None and cur >= prev_fails:
            break
        prev_fails = cur
    spath.write_text(original, encoding="utf-8")  # leave the source as we found it
    return None


PIPELINES = {"math": math, "data_analysis": data_analysis, "search": search, "coding": coding}
