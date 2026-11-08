"""Metadata-only JSONL journal. Never writes prompts or model answers."""
from __future__ import annotations

import json
import os
import threading
import time


class Journal:
    def __init__(self, log_dir: str) -> None:
        self._lock = threading.Lock()
        self._f = None
        try:
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"run-{int(time.time())}.jsonl")
            self._f = open(path, "a", encoding="utf-8")
            self.path = path
        except OSError:
            self.path = ""

    def event(self, kind: str, **fields) -> None:
        if self._f is None:
            return
        row = {"ts": round(time.time(), 3), "event": kind, **fields}
        try:
            with self._lock:
                self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._f.flush()
        except (OSError, ValueError):
            pass  # a logging failure must never break the run

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
