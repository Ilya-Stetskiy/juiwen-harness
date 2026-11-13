"""No API: hide_secrets removes the key from os.environ (children and openjiuwen tools inherit nothing), scrub_answer
still removes the remembered value from answers, whatever HARNESS_GUARD is."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEY = "sk-test-secret-123456"
os.environ["LLM_API_KEY"] = KEY
os.environ["LLM_BASE_URL"] = "http://proxy.example:8080"
os.environ.pop("HARNESS_GUARD", None)

from harness import guard  # noqa: E402
from harness.config import Config  # noqa: E402


class J:
    def __init__(self):
        self.ev = []

    def event(self, name, **kw):
        self.ev.append((name, kw))


cfg = Config.from_env()
removed = guard.hide_secrets()
assert removed == ["LLM_API_KEY"], removed
assert cfg.api_key == KEY, "Config keeps the key"
assert "LLM_API_KEY" not in os.environ
out = subprocess.run([sys.executable, "-c", "import os; print(os.environ.get('LLM_API_KEY', 'none'))"],
                     capture_output=True, text=True).stdout.strip()
assert out == "none", out
try:
    from openjiuwen.core.sys_operation.local.utils import OperationUtils
    assert "LLM_API_KEY" not in OperationUtils.prepare_environment(None), "agent bash would see the key"
    print("openjiuwen tool env: clean")
except ImportError:
    print("openjiuwen not installed here: tool env not checked")
j = J()
assert guard.scrub_answer(f"the key is {KEY}", "t1", j) == "the key is [removed]"
assert j.ev and j.ev[0][1]["action"] == "scrub"
assert guard.scrub_answer("42", "t1", J()) == "42"
os.environ["HARNESS_SCRUB_ANSWER"] = "off"
assert KEY in guard.scrub_answer(KEY, "t1", J())
print("test_secrets: ok")
