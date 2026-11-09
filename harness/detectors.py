"""Loop detector (S13) for model replies: repeated text, no progress while tokens grow. Skeleton; OFF by default.

    HARNESS_LOOP=off|log|retry        off (default): nothing; log: a `loop_check` event per call and `loop` when flagged;
                                      retry: if flagged and the reply has no FINAL line, ask once more with another
                                      temperature (HARNESS_LOOP_TEMP, default 0.6) and a request to be brief.
    HARNESS_LOOP_THRESHOLD            share of repeated 6-word windows that flags a loop (default 0.5)

Reasoning text is not part of the reply on this gateway, so a loop that lives inside the thinking shows only as
tokens: a reply that is cut at max_tokens without a FINAL line is flagged as `truncated` (cost >= 90% of the limit).
The detector sees finished replies (post hoc); stopping a stream in the middle needs a streaming call and is a later step.
"""
from __future__ import annotations

import os
import re


def mode() -> str:
    m = os.environ.get("HARNESS_LOOP", "off").strip().lower()
    return m if m in ("off", "log", "retry") else "off"


def threshold() -> float:
    try:
        return float(os.environ.get("HARNESS_LOOP_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


def loop_score(text: str, window: int = 6) -> dict:
    words = re.findall(r"\S+", text or "")
    n = len(words)
    if n < 3 * window:
        return {"words": n, "repeat": 0.0, "dup_lines": 0.0}
    grams = [tuple(words[i:i + window]) for i in range(n - window + 1)]
    seen: set = set()
    rep = 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    dup = (len(lines) - len(set(lines))) / len(lines) if lines else 0.0
    return {"words": n, "repeat": round(rep / len(grams), 3), "dup_lines": round(dup, 3)}


def is_loop(score: dict) -> bool:
    return score["repeat"] >= threshold() or (score["dup_lines"] >= threshold() and score["words"] >= 40)


def truncated(reply: str, cost: float, max_tokens: int) -> bool:
    return "FINAL" not in (reply or "").upper() and cost >= 0.9 * max_tokens
