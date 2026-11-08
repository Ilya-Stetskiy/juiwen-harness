"""Answer verifier: P(answer is correct) from cheap-to-expensive signals, independent of who produced the answer.

NOT wired into executor.py / run_agent.py (decision of the user). Interface:

    verify(task, answer, context=None) -> float                 # probability that `answer` is correct
    verify_detailed(task, answer, context=None) -> Verdict      # + which signals were used, features, level

`task` is a TaskInfo (or anything with task_type, check, instruction, workspace). `answer` is the normalised answer
text, or None when the producer gave no answer (for an unanswerable task that is the correct outcome).
`context` is an optional dict with whatever the caller has; every key is optional:

    runs           list of (code, returncode, stdout) of the program(s) the solver executed (math, data_analysis)
    docs           concatenated text of the task documents (search); read from task.workspace when absent
    retry_used     True if the solver needed a second attempt
    agree_other    answer of the same pipeline run with another model equals `answer` (True/False)
    other_answered the other model produced an answer
    judge_yes      LLM judge said the answer is supported / the program is right (see tools/verify_collect.py JUDGE)
    quote_in_docs  the judge's evidence quote occurs verbatim in the documents
    quote_has_ans  the quote contains the answer

Levels (a coefficient set is fitted per task type and level by tools/verify_fit.py, see profiles/verifier_model.json):
    free         no tokens: format, program health, retry used, hard-coded output, extractability, answer shape
    judge        free + LLM judge (judge_prompt / parse_judge)
    other        free + agreement with the answer of another model (`agree_other`, `other_answered`)
    other_judge  free + both
Missing models fall back to a per-type prior and `used == []`.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "profiles" / "verifier_model.json"
_model_cache: dict | None = None

_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‘": "'", "’": "'", "“": '"', "”": '"'})
_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def norm_text(s: str) -> str:
    """Same normalisation as the lab grader (tools/stream_run.py `_norm`)."""
    s = s.translate(_DASHES).lower()
    s = "".join(c for c in s if c not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def to_num(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


# ------------------------------------------------------------------ free signals
def code_features(runs) -> dict:
    f = {"n_runs": len(runs), "prog_ok": 0, "prog_nan": 0, "hardcode": 0, "code_len": 0, "n_calls_code": 0,
         "n_ops": 0, "retry_used": int(len(runs) > 1)}
    if not runs:
        return f
    code, rc, out = runs[-1]
    f["prog_ok"] = int(rc == 0)
    f["prog_nan"] = int(bool(re.search(r"\bnan\b", out, re.I)))
    f["code_len"] = len(code)
    f["out_lines"] = len([ln for ln in out.splitlines() if ln.strip()])
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f
    prints = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print"]
    ops = [n for n in ast.walk(tree) if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp))]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") != "print"]
    f["n_ops"], f["n_calls_code"] = len(ops), len(calls)
    f["hardcode"] = int(bool(prints) and all(all(isinstance(a, (ast.Constant, ast.JoinedStr)) for a in p.args)
                                             for p in prints) and not ops and not calls)
    return f


def format_ok(check: str, answer, instruction: str) -> int:
    if answer is None:
        return 0
    a = answer.strip()
    if check == "exact_int":
        return int(bool(re.fullmatch(r"-?\d+", a)))
    if check == "dabench_format":
        want = {k.lower() for k in re.findall(r"@([\w\-]+)\[", instruction)}
        got = {k.lower() for k in re.findall(r"@([\w\-]+)\[", a)}
        return int(bool(want) and want <= got)
    return int(0 < len(a) <= 120)


def read_docs(ws) -> str:
    if ws is None:
        return ""
    out = []
    for p in sorted(Path(ws).rglob("*")):
        if p.is_file():
            try:
                out.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return "\n".join(out)


def answer_features(task_type: str, answer, docs_norm: str) -> dict:
    f = {"ans_len": len(answer) if answer else 0}
    if answer is None:
        return f
    n = norm_text(answer)
    f["ans_in_docs"] = int(bool(n) and n in docs_norm)
    if task_type == "math":
        v = to_num(answer)
        f["ans_int"] = int(v is not None and float(v).is_integer())
        f["ans_log10"] = round(math.log10(abs(v) + 1), 2) if v is not None else None
        f["ans_neg"] = int(v is not None and v < 0)
    f["ans_numeric"] = int(to_num(answer) is not None)
    return f


def free_features(task, answer, context: dict | None = None) -> dict:
    ctx = context or {}
    f = code_features(ctx.get("runs") or [])
    if "retry_used" in ctx:
        f["retry_used"] = int(bool(ctx["retry_used"]))
    f["answered"] = int(answer is not None)
    f["fmt_ok"] = format_ok(task.check, answer, task.instruction)
    docs = ctx.get("docs")
    if docs is None and task.task_type == "search":
        docs = read_docs(getattr(task, "workspace", None))
    f.update(answer_features(task.task_type, answer, norm_text(docs or "")))
    return f


# ------------------------------------------------------------------ model
@dataclass
class Verdict:
    p: float
    level: str
    used: list = field(default_factory=list)
    features: dict = field(default_factory=dict)


def load_model(path: Path | None = None) -> dict:
    global _model_cache
    if path is not None or _model_cache is None:
        p = path or MODEL_PATH
        try:
            _model_cache = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _model_cache = {}
    return _model_cache


def _levels(ctx: dict) -> list[str]:
    has_j, has_o = "judge_yes" in ctx, "agree_other" in ctx
    lv = []
    if has_j and has_o:
        lv.append("other_judge")
    if has_o:
        lv.append("other")
    if has_j:
        lv.append("judge")
    lv.append("free")
    return lv


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def verify_detailed(task, answer, context: dict | None = None, model: dict | None = None) -> Verdict:
    ctx = dict(context or {})
    feats = free_features(task, answer, ctx)
    for k in ("agree_other", "other_answered", "judge_yes", "quote_in_docs",
              "quote_has_ans", "agree_indep"):
        if k in ctx and ctx[k] is not None:
            feats[k] = int(bool(ctx[k]))
    models = (model if model is not None else load_model()).get(task.task_type, {})
    for level in _levels(ctx):
        m = models.get(level)
        if not m:
            continue
        z = m["intercept"]
        for name, mean, scale, coef in zip(m["features"], m["mean"], m["scale"], m["coef"]):
            v = feats.get(name)
            x = mean if v is None else float(v)
            z += coef * (x - mean) / (scale or 1.0)
        used = [k for k in ("agree_other", "agree_indep", "judge_yes", "quote_in_docs") if k in feats]
        return Verdict(_sigmoid(z), level, used, feats)
    prior = (model if model is not None else load_model()).get("_prior", {}).get(task.task_type, 0.7)
    return Verdict(float(prior), "prior", [], feats)


def verify(task, answer, context: dict | None = None) -> float:
    """Probability that `answer` is the right outcome for `task`."""
    return verify_detailed(task, answer, context).p


# ------------------------------------------------------------------ LLM judge (same prompts as tools/verify_collect.py)
JUDGE_SYSTEM = {
    "search": ("You verify answers. Given a question, documents and a candidate answer, decide whether the documents "
               "support the candidate as the answer. Reply on line 1 with YES or NO. On line 2 give the shortest "
               "fragment copied verbatim from the documents that supports it (empty if NO)."),
    "data_analysis": ("You verify answers. Given a question about a table, the table overview, the Python program that "
                      "was run and the answer it printed, decide whether the program correctly answers the question "
                      "and the printed answer is in the requested format. Reply with one word: YES or NO."),
    "math": ("You verify answers. Given a math problem, the Python program that was run and the answer it printed, "
             "decide whether the program really computes the answer from the problem (not a hard-coded guess) and the "
             "answer is plausible. Reply with one word: YES or NO."),
}
JUDGE_MAX_TOKENS = {"search": 120, "data_analysis": 10, "math": 10}
MAX_DOC_CHARS = 40_000


def judge_prompt(task, answer, context: dict | None = None):
    """(system, user, max_tokens) of the judge call, or None for types without a judge (coding: run the tests)."""
    t = task.task_type
    if t not in JUDGE_SYSTEM:
        return None
    from .pipelines import clean_instruction
    ctx = context or {}
    q = clean_instruction(task.instruction)
    ans = answer if answer is not None else "(no answer)"
    if t == "search":
        docs = ctx.get("docs")
        if docs is None:
            docs = read_docs(getattr(task, "workspace", None))
        user = f"Question: {q}\n\nDocuments:\n{docs[:MAX_DOC_CHARS]}\n\nCandidate answer: {ans}"
    else:
        runs = ctx.get("runs") or []
        code = runs[-1][0] if runs else "(none)"
        extra = ""
        if t == "data_analysis" and getattr(task, "workspace", None) is not None:
            from .pipelines import _data_overview
            extra = f"\n\nTable overview:\n{_data_overview(task.workspace)}"
        user = f"Question: {q}{extra}\n\nProgram:\n{code}\n\nPrinted answer: {ans}"
    return JUDGE_SYSTEM[t], user, JUDGE_MAX_TOKENS[t]


def parse_judge(reply: str, task=None, answer=None, context: dict | None = None) -> dict:
    """Context keys from the judge reply: judge_yes and, for search, quote_in_docs / quote_has_ans."""
    lines = [ln for ln in (reply or "").strip().splitlines() if ln.strip()]
    out = {"judge_yes": int(bool(lines) and lines[0].strip().upper().startswith("YES"))}
    if task is not None and task.task_type == "search":
        docs = (context or {}).get("docs")
        if docs is None:
            docs = read_docs(getattr(task, "workspace", None))
        quote = norm_text(lines[1]) if len(lines) > 1 else ""
        out["quote_in_docs"] = int(len(quote) >= 3 and quote in norm_text(docs))
        out["quote_has_ans"] = int(answer is not None and norm_text(answer) in quote)
    return out
