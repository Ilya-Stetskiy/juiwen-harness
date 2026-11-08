"""Runtime configuration: environment variables and defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_BUDGET = 100_000

MODELS = (
    "qwen3.6-35b-a3b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b-noreason",
    "qwen3.8-27b",
)


def normalize_base_url(url: str) -> str:
    """The proxy address may come without a path; OpenAI-compatible clients need /v1."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return url
    if urlparse(url).path in ("", "/"):
        url += "/v1"
    return url


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    budget: int
    api_base: str
    api_key: str
    workers: int
    force_model: str | None  # for tests and lab experiments only
    limits_on: bool  # hard time limits (off in tests, on for the final run)
    log_dir: str
    max_task_seconds: int
    run_seconds: int

    @staticmethod
    def from_env() -> "Config":
        return Config(
            budget=_env_int("BUDGET", DEFAULT_BUDGET),
            api_base=normalize_base_url(os.environ.get("LLM_BASE_URL", "")),
            api_key=os.environ.get("LLM_API_KEY", ""),
            workers=max(1, min(5, _env_int("HARNESS_WORKERS", 5))),
            force_model=os.environ.get("HARNESS_FORCE_MODEL") or None,
            limits_on=os.environ.get("HARNESS_LIMITS", "off").lower() == "on",
            log_dir=os.environ.get("HARNESS_LOG_DIR", "/tmp/harness_logs"),
            max_task_seconds=_env_int("HARNESS_MAX_TASK_SECONDS", 480),
            run_seconds=_env_int("HARNESS_RUN_SECONDS", 3600),
        )
