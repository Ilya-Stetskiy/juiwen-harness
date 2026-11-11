"""Exact input-token counts with the real Qwen tokenizer (HARNESS_TOKENIZER=off|on, default off).

The tokenizer file profiles/qwen_tokenizer.json is the Qwen3.6-35B-A3B tokenizer.json (12.8 MB; the Qwen3.8-27B one differs only in
special tokens). Checked on 36 real task prompts: token ids are identical to llama.cpp /tokenize of the served model; and for the
prompt of a chat call prompt_tokens = tokens(system) + tokens(user) + overhead, with overhead 16-17 for the -noreason models and
40-41 for the reasoning ones (chat template; measured on the gateway). chars/4 is off by a median 14 % and 40 % on code or
tables; the exact count is off by 0-1 token. Without the file or the `tokenizers` package the count falls back to chars / 4.

    count(text) -> int
    chat_prompt_tokens(system, user, model) -> int
"""
from __future__ import annotations

import os
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "profiles" / "qwen_tokenizer.json"
_tok = None
_failed = False
OVERHEAD_NOREASON = 17
OVERHEAD_REASON = 41


def enabled() -> bool:
    return os.environ.get("HARNESS_TOKENIZER", "off").strip().lower() == "on"


def _load():
    global _tok, _failed
    if _tok is None and not _failed:
        try:
            from tokenizers import Tokenizer
            _tok = Tokenizer.from_file(os.environ.get("HARNESS_TOKENIZER_FILE") or str(_PATH))
        except Exception:  # noqa: BLE001 - no file / no package: the estimate falls back to chars / 4
            _failed = True
    return _tok


def exact() -> bool:
    return _load() is not None


def count(text: str) -> int:
    tk = _load()
    if tk is None:
        return max(1, len(text) // 4)
    return len(tk.encode(text).ids)


def chat_prompt_tokens(system: str, user: str, model: str) -> int:
    over = OVERHEAD_NOREASON if model.endswith("-noreason") else OVERHEAD_REASON
    return count(system) + count(user) + over
