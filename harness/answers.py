"""Answer writing (atomic) and normalisation."""
from __future__ import annotations

import os
import re
from pathlib import Path

_FINAL = re.compile(r"FINAL:[ \t]*(.+)", re.IGNORECASE)
_INT = re.compile(r"-?\d+")


def extract_final(output: str, fallback: bool = True) -> str | None:
    """Last `FINAL: <value>` line of the model reply.

    With fallback=True a short last line is used when there is no FINAL line (tool agent). The
    scripted pipelines pass fallback=False: a reply cut off by max_tokens must count as "no
    answer", not as a garbage last line.
    """
    if not output:
        return None
    matches = _FINAL.findall(output)
    if matches:
        return matches[-1].strip()
    if not fallback:
        return None
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if lines and len(lines[-1]) <= 100:
        return lines[-1]
    return None


def normalize(check: str, text: str) -> str:
    text = text.strip().rstrip(".").strip()  # trailing period first, so closing quotes become visible
    text = text.strip("`*\"'").strip()
    if check == "exact_int":
        m = _INT.search(text.replace(",", "").replace(" ", ""))
        return m.group(0) if m else text
    return text


def write_answer(task_dir: Path, text: str) -> None:
    """answer.txt next to task.json; temp file + rename so a kill never leaves a torn file."""
    tmp = task_dir / ".answer.txt.tmp"
    final = task_dir / "answer.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
