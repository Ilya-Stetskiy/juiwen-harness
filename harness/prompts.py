"""Prompt variants per task type, chosen by cheap task features (no LLM).

`profiles/prompts.json` holds, for every pipeline kind, an ordered list of variants. A variant may
carry a `when` condition (`check`: list of check types, `instruction_regex`); the first variant whose
condition holds is used, a variant without `when` is the default. `HARNESS_VARIANT_<KIND>=name` forces
a variant by name (A/B experiments of the lab).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .tasks import TaskInfo

_PATH = Path(__file__).resolve().parent.parent / "profiles" / "prompts.json"
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(_PATH.read_text(encoding="utf-8"))
    return _cache


def _matches(task: TaskInfo, when: dict) -> bool:
    if "check" in when and task.check not in when["check"]:
        return False
    if "instruction_regex" in when and not re.search(when["instruction_regex"], task.instruction, re.IGNORECASE):
        return False
    return True


def pick(task: TaskInfo, kind: str) -> dict:
    variants = _load().get(kind, [])
    forced = os.environ.get(f"HARNESS_VARIANT_{kind.upper()}")
    if forced:
        for v in variants:
            if v["name"] == forced:
                return v
    for v in variants:
        if _matches(task, v.get("when", {})):
            return v
    return variants[-1] if variants else {}
