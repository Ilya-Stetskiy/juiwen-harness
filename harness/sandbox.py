"""Sandbox for the code the model writes (S12). Interface stub with one trivial variant; OFF by default.

    HARNESS_SANDBOX=off|scrub|limits   off (default): the old behaviour, the child inherits the whole environment
                                       (including LLM_API_KEY); scrub: the child gets a minimal environment, no secrets;
                                       limits: scrub + CPU / memory / file-size limits (Linux, resource module).

`run(argv, cwd, timeout) -> (returncode, stdout, stderr)`; returncode 124 means timeout. Not done yet (so NOT a
security boundary): network isolation, filesystem confinement (the child can still read the whole container), pids.
"""
from __future__ import annotations

import os
import subprocess


def mode() -> str:
    m = os.environ.get("HARNESS_SANDBOX", "off").strip().lower()
    return m if m in ("off", "scrub", "limits") else "off"


def enabled() -> bool:
    return mode() != "off"


def _limits(timeout: float):
    def apply() -> None:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 5, int(timeout) + 10))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 ** 2, 50 * 1024 ** 2))
    return apply


def clean_env() -> dict:
    keep = ("PATH", "LANG", "LC_ALL", "PYTHONIOENCODING")
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["HOME"] = "/nonexistent"
    return env


def run(argv: list[str], cwd: str, timeout: float) -> tuple[int, str, str]:
    kw = {"env": clean_env()}
    if mode() == "limits" and os.name == "posix":
        kw["preexec_fn"] = _limits(timeout)
    try:
        r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    return r.returncode, r.stdout or "", r.stderr or ""
