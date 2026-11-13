"""No API: the late queue is ranked by reward x P(agent solves | type) / mean agent cost (type), not by reward alone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.executor import AGENT_STATS, agent_rating  # noqa: E402


class T:
    def __init__(self, tid, typ, reward):
        self.task_id, self.task_type, self.reward = tid, typ, reward


coding5, da3, math5, search3 = T("c5", "coding", 5), T("d3", "data_analysis", 3), T("m5", "math", 5), T("s3", "search", 3)
# the examples of FINDINGS / the discussion: DA reward 3 beats coding reward 5, search reward 3 beats math reward 5
assert agent_rating(da3) > agent_rating(coding5), (agent_rating(da3), agent_rating(coding5))
assert agent_rating(search3) > agent_rating(math5)
q = sorted([coding5, da3, math5, search3], key=lambda t: -agent_rating(t))
assert [t.task_id for t in q] == ["s3", "d3", "c5", "m5"], [t.task_id for t in q]
# stable on a tie: same type and reward keep their order
a, b = T("a", "coding", 4), T("b", "coding", 4)
assert [t.task_id for t in sorted([a, b], key=lambda t: -agent_rating(t))] == ["a", "b"]
assert agent_rating(T("x", "unknown", 2)) > 0
assert set(AGENT_STATS) == {"coding", "math", "data_analysis", "search"}
print("test_late_order: ok", [(t.task_id, round(agent_rating(t), 2)) for t in q])
