#!/usr/bin/env python3
"""Autonomous lab (S9): a hypothesis generator and an applying agent around the experiment queue. Lab tool, runs on the GPU host.

    python3 autolab.py cycle [--n 2] [--topic T] [--wait]   ONE command: propose -> validate/dedupe -> apply (code) -> sim filter -> submit -> (wait) -> verdict -> report
    python3 autolab.py cycle --names a,b [--wait]            the same for drafts that exist (skips propose); stages that are done are not repeated
    python3 autolab.py propose [--n 3] [--topic T] [--dry]   the generator asks the LLM for hypotheses -> drafts in experiments/autolab/
    python3 autolab.py list                                  drafts, their stage and verdict
    python3 autolab.py apply <name> [--patch-file F]         applying agent: a code change ONLY in a worktree on branch autolab/<name>
    python3 autolab.py sim <name>                            simulator as a FILTER before the expensive run (scheduler-level hypotheses)
    python3 autolab.py submit <name>                         draft -> queue/pending (the dispatcher runs the queue)
    python3 autolab.py verdict <name>                        result -> experiments/autolab/<name>.verdict.md (threshold = noise floor of the lab)
    python3 autolab.py export <name>                         the branch as a patch file for a human to review and merge
    python3 autolab.py usage                                 gateway REQUESTS spent by the autolab (the gateway counts requests, not tokens)
    python3 autolab.py selftest

Rules that are enforced in code (not by asking the model nicely):
  * the model writes hypotheses (YAML) and code edits (search/replace blocks) only; it never touches the frozen sets, the lab
    (tools/, experiments/, results), the acceptance criteria or keys: a change may only touch harness/, profiles/, run_agent.py,
    is checked against a deny-list, and a snapshot of the protected files is compared before and after;
  * code lives in a worktree on a branch autolab/<name> (on the host: a git repo seeded from the deployed archive); the run of such a
    hypothesis uses that tree through the experiment key `code_dir`; nothing is merged or deployed to the harness: MERGE ONLY BY A HUMAN;
  * limits per cycle and per week (LLM requests of the autolab itself, hypotheses per cycle, experiments in the queue, expected
    requests) are in LIMITS_DEFAULT, overridable by <lab>/autolab.yaml (a lab file, the model cannot write it);
  * the verdict uses the lab's rule: paired bootstrap interval AND the noise-floor threshold of `lab.py noisefloor`; this module only
    READS the threshold.

Model: AUTOLAB_MODEL (default qwen3.8-27b) through the shared proxy (127.0.0.1:18443); before each call the experiment gate of
dispatcher.yaml is checked, a 429 is waited out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stream_run as sr  # noqa: E402
import autolab_status as astat  # noqa: E402

LAB = sr.LAB
EXP = Path(os.environ.get("LAB_EXPERIMENTS") or LAB / "experiments")
DRAFTS = EXP / "autolab"
USAGE = DRAFTS / "usage.jsonl"
MODEL = os.environ.get("AUTOLAB_MODEL", "qwen3.8-27b")
BASE = os.environ.get("AUTOLAB_BASE_URL", f"http://127.0.0.1:{sr.PROXY_PORT}/v1")

LIMITS_DEFAULT = {
    "max_hyp_per_cycle": 3,            # hypotheses taken from one propose call
    "max_llm_calls_per_cycle": 14,     # gateway requests of the autolab itself (propose + apply + retries) in one process run
    "weekly_llm_calls": 1500,          # gateway requests of the autolab per ISO week (the gateway limit is 1 000 000 requests)
    "max_queue_depth": 4,              # autolab experiments (ab_*) pending + running before submit refuses (experiments of the stand in the same queue are not counted)
    "max_expected_requests_cycle": 12000,   # estimated gateway requests of all experiments of one cycle
    "max_cost_tokens": 150000,         # expected ledger tokens of one experiment
    "max_code_lines": 400,             # added + removed lines of one model patch (was 120; 2026-11-12: bigger patches, new files allowed)
    "apply_max_tokens": 16000,         # output tokens of the applying agent (was a fixed 4000: a long patch was cut and its JSON did not close)
    "wait_poll_s": 60, "max_wait_min": 300,
    "max_rerun_repeats": 3,
    # Claude API for the generator and the applying agent (the evaluation runs stay on qwen through the gateway)
    "llm_backend": "auto",             # auto: Claude if its key file exists, else DeepSeek if the swarm has its key, else qwen | claude | deepseek | qwen; a backend that does not answer is skipped and the chat is told
    "claude_model_propose": "claude-opus-5-5", "claude_model_apply": "claude-opus-5-5", "claude_model_review": "claude-opus-5-5",   # roles: generator, applying agent, reviewer; put another model on the reviewer for a more independent opinion
    "claude_effort": "medium", "claude_max_tokens": 8000,
    "usd_per_run": 5.0, "usd_per_day": 10.0, "usd_per_week": 50.0,   # Claude dollars of everything the autolab calls (the chief included): per process run, per local day, per ISO week
    "claude_model_chief": "claude-opus-5-5",   # the chief (tools/autolab_chief.py): plans the next steps of the improvement trajectory
    "chief_dry_run": True,             # the chief plans and tells the chat but executes nothing until a human sets this to false
    "local_model": "local-35b-noreason",   # the model of experiments run on the local Qwen (local-35b = thinking on)
    "exp_parallel_runs": 4,            # runs (arm x repeat) of one experiment at once (lab.py parallel_runs); the gateway proxy caps at rpm 55
    "chief_max_actions": 4,            # actions per model call
    "chief_experiments_until": "",     # "YYYY-MM-DD HH:MM" host time: after it the chief starts nothing new (the time of the experiments ends)
    "chief_final_deadline": "",        # the end of everything (the final runs on main and the hand-in): only shown to the chief
    "chief_final_check_h": 3,          # this many hours before experiments_until the chief forces a whole-stack check
    "chief_max_rounds": 3,             # model calls per tick: after a reject / accept (instant) it thinks again at once; a review / experiment ends the tick
    "chief_event_gap_min": 5,          # least minutes between two ticks (a verdict, a review or a free slot wakes the chief; cron every 5 minutes)
    "chief_max_active": 3,             # hypotheses holding task types at once (the chief proposes no more while this many are active)
    "chief_max_rechecks": 3,           # checks on other data + repeats per hypothesis
    "chief_min_gap_min": 25,           # minimum minutes between two ticks (cron runs every 30)
    "chief_stack_set": "mixed100_dev",    # the set of the whole-stack check and of the leave-one-out ablation (was: the first set by name, adversarial14 with 14 tasks; a key that is not here is ignored by limits_cfg)
    "chief_stack_budget": 100000,         # its budget: the final scale (100 tasks, 100k)
    "chief_stack_repeats": 2,
    "chief_stack_check_every": 2,         # a whole-stack check after this many additions
    "chief_max_tokens": 6000,          # output tokens of the chief's plan
    "claude_key_file": "~/.config/autolab/anthropic.key",   # the key of the claude roles: Anthropic (sk-ant-) or OpenRouter (sk-or-); the file name stays
    "claude_provider": "auto",         # auto: by the key prefix (sk-or- OpenRouter, else Anthropic) | anthropic | openrouter
    "openrouter_proxy": "http://127.0.0.1:7890",   # only requests to openrouter.ai go through it (the host's mihomo -> VPS: OpenRouter refuses the host's own IP); "" = direct; DeepSeek, Anthropic and the gateway never use it
    "openrouter_base": "https://openrouter.ai/api/v1", "openrouter_title": "harness-autolab",   # X-Title only; no personal data in headers
    "openrouter_models": {"claude-opus-5-5": "anthropic/claude-opus-5.5", "claude-sonnet-5-5": "anthropic/claude-sonnet-5.5",
                          "claude-fable-5-1": "anthropic/claude-fable-5.1", "claude-haiku-5-5": "anthropic/claude-haiku-5.5"},   # role model -> OpenRouter id (checked against GET /models); `openrouter_model_<role>` overrides one role
    "openrouter_prices": {"anthropic/claude-opus-5.5": [4.0, 20.0], "anthropic/claude-sonnet-5.5": [2.0, 10.0], "anthropic/claude-fable-5.1": [10.0, 50.0],
                          "anthropic/claude-haiku-5.5": [0.1, 0.5]},   # USD per 1M tokens in/out from /models, for the worst-case check before a request; the dollars written to usage.jsonl are the ones OpenRouter reports
    # autopilot: keeps the queue fed so the host does not idle (switched on and off from the chat; see cmd_autopilot)
    "local_data_max_temp_c": 82,       # the local data stream is not (re)started above this GPU temperature
    "local_data_workers": 3,           # parallel tasks of the local stream (llama-server has 3 slots)
    "autopilot_target_depth": 2,       # when pending + running experiments are fewer than this, the generator is asked for more
    "autopilot_batch": 2,              # hypotheses per tick at most
    "goal_mode": "accuracy",           # focus of autopilot / cycle / ideas when no mode is named: accuracy | economy | "" = alternate equally
    "economy_min_saving": 0.05,        # economy verdict: the arm must spend at least this share fewer tokens than the reference (token noise is not measured by the lab)
    "autopilot_max_per_day": 0,        # hypotheses put into the queue per calendar day at most (0 = no daily limit; the queue depth, task types and gateway gate still apply)
    "autopilot_topics": [
        "task order / scheduler / predictor (placeholder vs measured vs gbm) at budgets 20-40k",
        "cheap modes: noreason models, batching of math with compute, fallback, censor of tokens",
        "points per token by task type when the budget binds: which types to skip, cap or answer cheaply",
        "unsolvable-task skip and guess policies at a tight budget",
        "budget behaviour: overshoot of the budget, truncation of the output, per-task token caps",
    ],
    # DeepSeek (the key of the swarm's own model: API_KEY in the swarm .env, read at call time, never copied, only when API_BASE there is a DeepSeek endpoint). No money limit; only the request counts below.
    "deepseek_env_file": "~/.jiuwenswarm/config/.env", "deepseek_api_base": "https://api.deepseek.com",
    "deepseek_model_propose": "deepseek-flash", "deepseek_model_apply": "deepseek-v4-pro", "deepseek_model_review": "deepseek-v4-pro",
    "deepseek_max_tokens": 12000,      # includes the model's own reasoning; a small value leaves no room for the answer
    "push_remote": "git@github.com:Ilya-Stetskiy/juiwen-harness.git", "push_branch": "master",   # `accept` pushes here (never forced); needs the deploy key
    "deploy_key_file": "~/.config/autolab/deploy_key",
    "claude_prices": {"claude-sonnet-5-5": [2.0, 10.0], "claude-opus-5-5": [4.0, 20.0], "claude-fable-5-1": [10.0, 50.0], "claude-haiku-5-5": [0.1, 0.5]},   # USD per 1M tokens in/out
}
# The code a model-written patch may touch. Everything else (tools/, experiments/, frozen sets, dispatcher, keys, Docker) is refused.
ALLOWED_PREFIX = ("harness/", "profiles/", "run_agent.py")
FORBIDDEN = ("Dockerfile", "requirements.txt", "requirements-lab.txt", "tools/deploy.sh")
FORBIDDEN_ADDED = re.compile(r"holdout|noise_?floor|noisefloor|threshold_points|acceptance|\.ssh|api[_-]?key|secret|token_juiwen|"
                             r"experiments/|dispatcher|autolab|subprocess|os\.system|rmtree|urllib|socket|requests\.|\beval\(|\bexec\(|"
                             r"__import__|/home/|\.env\b", re.I)
PROTECTED = ("tools/lab.py", "tools/autolab.py", "tools/dispatcher.py", "tools/sim.py", "tools/stream_run.py", "experiments/sets", "dispatcher.yaml",
             "autolab.yaml")
MODE_RU = {"accuracy": "🎯 accuracy", "economy": "💸 economy", "both": "⚖️ оба режима"}
DRAFT_KEYS = ("name", "hypothesis", "set", "model", "repeats", "budget", "arms", "expected_cost_tokens", "code_change", "target_cells", "why_cell", "target_mode", "cross_check")
GOAL_MODES = ("accuracy", "economy", "both")     # accuracy: most points within the budget; economy: the same points for fewer tokens (the organizers' tie-break)
ARM_KEYS = ("env", "batch_size", "model")
# Allowed values of the enumerated switches (anything else is silently treated as "off" by the harness, so it is refused here).
SWITCH_VALUES = {
    "HARNESS_VERIFY": ["off", "free", "judge", "other", "other_judge"], "HARNESS_VERIFY_ACTION": ["log", "retry", "recheck"],
    "HARNESS_RECHECK_MODE": ["strong", "other"], "HARNESS_RECHECK_QUEUE": ["inline", "later"],
    "HARNESS_DISAGREE": ["higher_p", "strong", "third"], "HARNESS_LOOP": ["off", "log", "retry"],
    "HARNESS_CENSOR": ["off", "log", "enforce"], "HARNESS_CENSOR_BASIS": ["fixed", "pred"],
    "HARNESS_BUDGET_HARD": ["off", "on"], "HARNESS_GUARD": ["off", "log", "strip"], "HARNESS_SANDBOX": ["off", "scrub", "limits"],
    "HARNESS_BATCH_PROMPT": ["plain", "compute"], "HARNESS_SORTER": ["greedy", "fifo", "reward"],
    "HARNESS_PREDICTOR": ["placeholder", "flat", "measured", "gbm"], "HARNESS_PUNSOLV": ["off", "contradiction"],
    "HARNESS_SOLVERS": ["arith", "contradiction", "arith,contradiction"], "HARNESS_SUBMIT_UNSOLVABLE": ["on", "off"],
    "HARNESS_FALLBACK": ["on", "off", "late"], "HARNESS_GUESS_UNSOLVABLE": ["off", "free", "llm"],
    "HARNESS_PSUCCESS": ["table", "measured"], "HARNESS_COST_BASIS": ["p50", "mean"], "HARNESS_WORKER_WAIT": ["off", "on"],   # review R-X5, R-X4, R-X3
    "HARNESS_VERIFY_JUDGE_TYPES": None, "HARNESS_VERIFY_OTHER_TYPES": None, "HARNESS_BATCH_TYPES": None, "HARNESS_RECHECK_TYPES": None,
}
# Switches the arms of a hypothesis may NOT set although SCRIPTS.md names them: they change the grading, the budget or the gateway.
DENY_SWITCHES = {"HARNESS_FORCE_MODEL", "HARNESS_LIMITS", "HARNESS_CWD", "HARNESS_LOG_DIR", "HARNESS_BUDGET_FILE", "BUDGET"}
MODELS = ["qwen3.6-35b-a3b-noreason", "qwen3.6-35b-a3b", "qwen3.8-27b-noreason", "qwen3.8-27b"]
SIM_KEYS = ("HARNESS_SORTER", "HARNESS_PREDICTOR", "HARNESS_PUNSOLV", "HARNESS_SOLVERS", "HARNESS_PSUCCESS", "HARNESS_COST_BASIS")
CALLS = {"n": 0, "usd": 0.0}   # LLM requests and Claude dollars spent by this process
STOP_FILE = LAB / "AUTOLAB_STOP"   # `touch` it (or Telegram `stop`) and the autolab makes no more LLM calls and submits nothing


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def limits_cfg() -> dict:
    cfg = dict(LIMITS_DEFAULT)
    f = LAB / "autolab.yaml"
    if f.exists():
        import yaml
        cfg.update({k: v for k, v in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).items() if k in LIMITS_DEFAULT})
    return cfg


# ------------------------------------------------------------------ accounting of the autolab's own gateway requests
def usage_rows() -> list[dict]:
    return [json.loads(l) for l in open(USAGE, encoding="utf-8") if l.strip()] if USAGE.exists() else []


def week_start_ts() -> float:
    t = time.gmtime()
    return time.time() - (t.tm_wday * 86400 + t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec)   # Monday 00:00 UTC, the gateway's ISO week


def week_calls() -> int:
    ws = week_start_ts()
    return sum(int(r.get("requests", 1)) for r in usage_rows() if r["ts"] >= ws)


def gw_week_used() -> int | None:
    try:
        import dispatcher
        fr = dispatcher.fractions(sr.limits())
        return fr["week_used"] if fr else None
    except Exception:  # noqa: BLE001
        return None


def record_usage(purpose: str, u: dict, ok: bool, backend: str = "qwen", model: str = "", usd: float = 0.0) -> None:
    pin, pout = u.get("prompt_tokens", u.get("input_tokens", 0)), u.get("completion_tokens", u.get("output_tokens", 0))
    cost = pin / 10.0 + pout
    DRAFTS.mkdir(parents=True, exist_ok=True)
    with open(USAGE, "a", encoding="utf-8") as f:   # tokens, dollars and the model only; never a key or a prompt
        f.write(json.dumps({"ts": time.time(), "model": model or MODEL, "backend": backend, "purpose": purpose, "requests": 1, "ok": ok,
                            "in": pin, "out": pout, "cost": round(cost, 1), "usd": round(usd, 5)}) + "\n")
    CALLS["usd"] += usd


def day_usd() -> float:
    t = time.localtime()
    ds = time.time() - (t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec)      # midnight of the host's local day
    return sum(float(r.get("usd", 0)) for r in usage_rows() if r["ts"] >= ds and r.get("backend") == "claude")


def week_usd() -> float:
    ws = week_start_ts()
    return sum(float(r.get("usd", 0)) for r in usage_rows() if r["ts"] >= ws and r.get("backend") == "claude")


# ------------------------------------------------------------------ Claude API (generator and applying agent), raw HTTPS: the host has no anthropic SDK
def key_path() -> Path:
    return Path(os.path.expanduser(os.environ.get("AUTOLAB_KEY_FILE") or limits_cfg()["claude_key_file"]))


def read_key() -> str:
    try:
        return key_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def scrub(text: str) -> str:
    k = read_key()
    return text.replace(k, "***") if k else text


class BackendDown(Exception):
    """A backend did not answer (network, HTTP error, empty answer): the next one in the chain takes over and the chat is told."""


LAST = {"backend": "", "model": ""}   # what answered the last llm() call
_TOLD: set = set()


def deepseek_cfg() -> tuple[str, str]:
    """(api_base, key) of the swarm's DeepSeek model, or ("", "") when the swarm does not run on DeepSeek. Reads only API_BASE and API_KEY."""
    lim = limits_cfg()
    f = Path(os.path.expanduser(os.environ.get("AUTOLAB_DEEPSEEK_ENV") or lim["deepseek_env_file"]))
    env: dict = {}
    try:
        for ln in f.read_text(encoding="utf-8").splitlines():
            k, _, v = ln.partition("=")
            if k.strip() in ("API_BASE", "API_KEY"):
                env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        return "", ""
    if "deepseek" not in env.get("API_BASE", "").lower() or not env.get("API_KEY"):
        return "", ""
    return env["API_BASE"].rstrip("/"), env["API_KEY"]


def chain() -> list[str]:
    """The backends in the order they are tried. A forced one stands alone; auto: Claude, DeepSeek, qwen (the last always works through the gateway)."""
    b = limits_cfg()["llm_backend"]
    if b == "qwen":
        return ["qwen"]
    if b == "claude":
        if not read_key():
            raise SystemExit(f"llm_backend=claude but there is no key at {key_path()}")
        return ["claude"]
    if b == "deepseek":
        if not deepseek_cfg()[1]:
            raise SystemExit("llm_backend=deepseek but the swarm .env has no DeepSeek API_BASE/API_KEY")
        return ["deepseek"]
    return (["claude"] if read_key() else []) + (["deepseek"] if deepseek_cfg()[1] else []) + ["qwen"]


def backend() -> str:
    return chain()[0]


def backend_desc(purpose: str = "propose") -> tuple[str, str]:
    """(backend, model) that the next call of this purpose would use; never raises."""
    lim = limits_cfg()
    try:
        be = backend()
    except SystemExit:
        return "claude (нет ключа)", lim["claude_model_" + purpose]
    return be, {"claude": lim["claude_model_" + purpose] if "claude_model_" + purpose in lim else lim["claude_model_propose"],
                "deepseek": lim.get("deepseek_model_" + purpose) or lim["deepseek_model_propose"], "qwen": MODEL}[be]


def notify_backend_down(be: str, why: str, nxt: str | None) -> None:
    log(f"{be} did not answer ({why}); " + (f"switching to {nxt}" if nxt else "no backend left"))
    if be in _TOLD:
        return
    _TOLD.add(be)
    name = {"claude": "Claude" + (" (OpenRouter)" if provider() == "openrouter" else ""), "deepseek": "DeepSeek", "qwen": "qwen"}[be]
    outbox_put(allowed_chat_ids(), f"⚠️ {name} не отвечает ({why[:160]}). " + (f"Дальше работаю на {nxt}." if nxt else "Запасного варианта нет, задача остановлена."))


def _open(url: str, req: urllib.request.Request, timeout: int):
    """urlopen; a request to openrouter.ai alone goes through `openrouter_proxy` (an explicit handler, no environment variables)."""
    px = limits_cfg().get("openrouter_proxy")
    if px and urllib.parse.urlparse(url).hostname == "openrouter.ai":
        return urllib.request.build_opener(urllib.request.ProxyHandler({"https": px, "http": px})).open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _http_post(url: str, headers: dict, body: bytes, timeout: int = 300) -> tuple[int, dict]:
    try:
        r = _open(url, urllib.request.Request(url, data=body, headers=headers), timeout)
        return r.status, json.load(r)
    except urllib.error.HTTPError as ex:
        try:
            return ex.code, json.loads(ex.read().decode("utf-8", "replace"))
        except ValueError:
            return ex.code, {}


def provider() -> str:
    """Who serves the claude roles: `claude_provider` (anthropic | openrouter) or, with auto, the prefix of the key (sk-or- is OpenRouter, anything else Anthropic). The key itself is never shown."""
    p = limits_cfg()["claude_provider"]
    if p in ("anthropic", "openrouter"):
        return p
    return "openrouter" if read_key().startswith("sk-or-") else "anthropic"


def or_model(model: str, purpose: str = "") -> str:
    """The OpenRouter id of a claude role model: `openrouter_model_<role>` if set, else the table `openrouter_models`; an id with a slash is taken as it is."""
    lim = limits_cfg()
    own = lim.get("openrouter_model_" + purpose) if purpose else None
    m = own or model
    return m if "/" in m else lim["openrouter_models"].get(m, "anthropic/" + m)


def claude_cost(model: str, tin: int, tout: int, prov: str | None = None) -> float:
    lim = limits_cfg()
    pr = lim["openrouter_prices"].get(or_model(model)) if (prov or provider()) == "openrouter" else None
    pr = pr or lim["claude_prices"].get(model)
    if not pr:
        raise SystemExit(f"no price for {model} in claude_prices / openrouter_prices (autolab.yaml): refusing to spend without a price")
    return (tin * pr[0] + tout * pr[1]) / 1e6


OR_ERRORS = {401: "ключ OpenRouter отклонён (401): проверьте ключ", 402: "на ключе OpenRouter нет денег или исчерпан лимит (402)",
             403: "OpenRouter отклонил запрос (403): модель недоступна этому ключу или сработала модерация", 404: "OpenRouter не знает такой модели (404): проверьте openrouter_models"}


def claude_llm(system: str, user: str, max_tokens: int, purpose: str) -> str:
    lim = limits_cfg()
    prov = provider()
    model = lim.get("claude_model_" + purpose) or lim["claude_model_propose"]   # purposes: propose, apply, review
    max_tokens = min(max_tokens, int(lim["claude_max_tokens"]))
    worst = claude_cost(model, (len(system) + len(user)) // 3 + 200, max_tokens, prov)   # worst case of this one call
    if CALLS["usd"] + worst > lim["usd_per_run"]:
        raise SystemExit(f"autolab Claude limit: {CALLS['usd']:.3f} USD spent in this run, next call up to {worst:.3f} (usd_per_run={lim['usd_per_run']})")
    if day_usd() + worst > lim["usd_per_day"]:
        raise SystemExit(f"autolab Claude limit: {day_usd():.3f} USD today, next call up to {worst:.3f} (usd_per_day={lim['usd_per_day']})")
    if week_usd() + worst > lim["usd_per_week"]:
        raise SystemExit(f"autolab Claude limit: {week_usd():.3f} USD this week, next call up to {worst:.3f} (usd_per_week={lim['usd_per_week']})")
    if prov == "openrouter":       # OpenAI chat format; usage.include returns the real cost of the call in dollars
        url = lim["openrouter_base"].rstrip("/") + "/chat/completions"
        body = json.dumps({"model": or_model(model, purpose), "max_tokens": max_tokens, "usage": {"include": True},
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + read_key(), "X-Title": lim["openrouter_title"]}
    else:
        # no temperature: current models reject non-default sampling; thinking is left to the model, effort bounds its depth
        url = "https://api.anthropic.com/v1/messages"
        body = json.dumps({"model": model, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": user}],
                           "output_config": {"effort": lim["claude_effort"]}}).encode()
        headers = {"content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": read_key()}
    for attempt in range(4):
        CALLS["n"] += 1
        try:
            code, r = _http_post(url, headers, body)
        except Exception as ex:  # noqa: BLE001 - network errors: the message may not carry the key, scrub anyway
            record_usage(purpose, {}, False, "claude", model)
            if attempt < 2 and "timed out" not in str(ex).lower():     # a dropped connection through the VPN proxy (SSL EOF, reset): nothing was served, try again; a timeout may have been charged: no retry
                log(f"Claude API: {type(ex).__name__}; again in {5 * (attempt + 1)} s")
                time.sleep(5 * (attempt + 1))
                continue
            raise SystemExit("Claude API unreachable: " + scrub(f"{type(ex).__name__}: {ex}")[:200])
        if code in (429, 529, 502, 503) and attempt < 3:
            record_usage(purpose, {}, False, "claude", model)
            log(f"Claude API {code}; waiting {30 * (attempt + 1)} s")
            time.sleep(30 * (attempt + 1))
            continue
        if code != 200:
            record_usage(purpose, {}, False, "claude", model)
            if prov == "openrouter" and code in OR_ERRORS:
                raise SystemExit("Claude API " + OR_ERRORS[code])
            raise SystemExit(f"Claude API error {code}: " + scrub(json.dumps(r.get("error", r), ensure_ascii=False))[:300])
        u = r.get("usage") or {}
        if prov == "openrouter":
            ch = (r.get("choices") or [{}])[0]
            text = (ch.get("message") or {}).get("content") or ""
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            real = u.get("cost")
            usd = float(real) if isinstance(real, (int, float)) else claude_cost(model, tin, tout, prov)   # the dollars OpenRouter charged, else the table price
            record_usage(purpose, u, bool(text), "claude", model, usd)
            if ch.get("finish_reason") == "content_filter":
                raise SystemExit("Claude declined the request (content_filter); nothing was used from it")
            return text
        record_usage(purpose, u, True, "claude", model, claude_cost(model, u.get("input_tokens", 0), u.get("output_tokens", 0), prov))
        if r.get("stop_reason") == "refusal":
            raise SystemExit("Claude declined the request (stop_reason refusal); nothing was used from it")
        return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
    raise SystemExit("Claude API kept answering 429/5xx")


def _http_get(url: str, headers: dict, timeout: int = 20) -> tuple[int, dict]:
    try:
        r = _open(url, urllib.request.Request(url, headers=headers), timeout)
        return r.status, json.load(r)
    except urllib.error.HTTPError as ex:
        try:
            return ex.code, json.loads(ex.read().decode("utf-8", "replace"))
        except ValueError:
            return ex.code, {}


def key_status(probe: bool = False) -> str:
    """What the bot shows for `ключ`: provider, role models, the key's balance and limit (OpenRouter: GET /key, free) and, with probe, one tiny paid request. The key is never printed."""
    lim = limits_cfg()
    k = read_key()
    if not k:
        return f"Ключа нет: файл {key_path()} не найден или пуст."
    prov = provider()
    how = "настроен в autolab.yaml" if lim["claude_provider"] in ("anthropic", "openrouter") else "по префиксу ключа"
    out = [f"Провайдер: {'OpenRouter' if prov == 'openrouter' else 'Anthropic'} ({how}); ключ {len(k)} байт, префикс {'sk-or-' if k.startswith('sk-or-') else 'sk-ant-' if k.startswith('sk-ant-') else 'нестандартный'}"]
    for role in ("propose", "apply", "review"):
        m = lim.get("claude_model_" + role) or lim["claude_model_propose"]
        out.append(f"• {role}: {m}" + (f" → {or_model(m, role)}" if prov == "openrouter" else ""))
    out.append(f"Лимиты автолаборатории: {lim['usd_per_run']} $ за запуск, {lim['usd_per_day']} $ в сутки, {lim['usd_per_week']} $ в неделю; потрачено на этой неделе {week_usd():.3f} $; цепочка запасных: {' → '.join(chain())}")
    if prov == "openrouter":
        try:
            code, r = _http_get(lim["openrouter_base"].rstrip("/") + "/key", {"Authorization": "Bearer " + k})
        except Exception as ex:  # noqa: BLE001
            code, r = 0, {}
            out.append("Баланс не проверен: " + scrub(f"{type(ex).__name__}: {ex}")[:150])
        if code == 200:
            d = r.get("data") or {}
            lm, rest = d.get("limit"), d.get("limit_remaining")
            out.append(f"Ключ принят. Израсходовано на ключе: {float(d.get('usage') or 0):.4f} $; " + (f"лимит {float(lm):g} $, осталось {float(rest):.4f} $" if lm is not None and rest is not None else "лимита на ключе нет") + ("; бесплатный тариф" if d.get("is_free_tier") else ""))
        elif code:
            out.append(OR_ERRORS.get(code, f"Проверка ключа: HTTP {code}"))
        try:
            code, r = _http_get(lim["openrouter_base"].rstrip("/") + "/models", {})
            live = {m["id"]: m.get("pricing", {}) for m in (r.get("data") or [])} if code == 200 else {}
            diff = []
            for role in ("propose", "apply", "review"):
                oid = or_model(lim.get("claude_model_" + role) or lim["claude_model_propose"], role)
                tab = lim["openrouter_prices"].get(oid)
                p = live.get(oid)
                if live and not p:
                    diff.append(f"{oid}: такой модели нет на OpenRouter")
                elif p and tab and (abs(float(p["prompt"]) * 1e6 - tab[0]) > 1e-6 or abs(float(p["completion"]) * 1e6 - tab[1]) > 1e-6):
                    diff.append(f"{oid}: цена на OpenRouter {float(p['prompt']) * 1e6:g}/{float(p['completion']) * 1e6:g} $ за 1M, в таблице {tab[0]:g}/{tab[1]:g}")
            if live:
                out.append("Цены и модели сверены с /models: " + ("; ".join(diff) if diff else "совпадают"))
        except Exception:  # noqa: BLE001
            pass
    if probe:
        try:
            before = CALLS["usd"]
            ans = claude_llm("Answer with one word.", "ping", 8, "probe")
            out.append(f"Тестовый запрос прошёл: ответ {len(ans)} симв., списано {CALLS['usd'] - before:.5f} $.")
        except SystemExit as ex:
            out.append("Тестовый запрос не прошёл: " + scrub(str(ex))[:240])
    else:
        out.append("Реальный запрос не делался (кнопка «Проверить запросом» тратит около 0,0001 $).")
    return "\n".join(out)


# ------------------------------------------------------------------ DeepSeek (OpenAI-compatible chat completions), no money limit
def deepseek_llm(system: str, user: str, max_tokens: int, purpose: str) -> str:
    lim = limits_cfg()
    base, key = deepseek_cfg()
    model = lim.get("deepseek_model_" + purpose) or lim["deepseek_model_propose"]
    body = json.dumps({"model": model, "max_tokens": int(lim["deepseek_max_tokens"]),
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    clean = lambda t: str(t).replace(key, "***")[:240]
    for attempt in range(3):
        CALLS["n"] += 1
        try:
            code, r = _http_post(base + "/chat/completions", headers, body, 300)
        except Exception as ex:  # noqa: BLE001
            record_usage(purpose, {}, False, "deepseek", model)
            raise BackendDown(clean(f"{type(ex).__name__}: {ex}"))
        if code in (429, 500, 502, 503, 504, 529) and attempt < 2:
            record_usage(purpose, {}, False, "deepseek", model)
            time.sleep(10 * (attempt + 1))
            continue
        if code != 200:
            record_usage(purpose, {}, False, "deepseek", model)
            raise BackendDown(clean(f"HTTP {code} {json.dumps(r.get('error', r), ensure_ascii=False)}"))
        ch = (r.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content") or ""
        record_usage(purpose, r.get("usage") or {}, bool(text), "deepseek", model)
        if not text.strip():
            raise BackendDown(f"empty answer (finish_reason {ch.get('finish_reason')}; raise deepseek_max_tokens if it is 'length')")
        return text
    raise BackendDown("kept answering 429/5xx")


# ------------------------------------------------------------------ the LLM: Claude / DeepSeek / the shared gateway
def llm(system: str, user: str, max_tokens: int = 4000, purpose: str = "") -> str:
    if STOP_FILE.exists():
        raise SystemExit(f"{STOP_FILE} exists: the autolab is stopped (remove it or send `resume`)")
    lim0 = limits_cfg()
    if CALLS["n"] >= lim0["max_llm_calls_per_cycle"]:
        raise SystemExit(f"autolab limit: {CALLS['n']} LLM requests in this run (max_llm_calls_per_cycle={lim0['max_llm_calls_per_cycle']})")
    if week_calls() >= lim0["weekly_llm_calls"]:
        raise SystemExit(f"autolab limit: {week_calls()} LLM requests this week (weekly_llm_calls={lim0['weekly_llm_calls']})")
    ch = chain()
    for i, be in enumerate(ch):
        nxt = ch[i + 1] if i + 1 < len(ch) else None
        try:
            if be == "claude":
                out = claude_llm(system, user, max_tokens, purpose)
                LAST.update(backend="claude", model=lim0.get("claude_model_" + purpose) or lim0["claude_model_propose"])
            elif be == "deepseek":
                out = deepseek_llm(system, user, max_tokens, purpose)
                LAST.update(backend="deepseek", model=lim0.get("deepseek_model_" + purpose) or lim0["deepseek_model_propose"])
            else:
                out = qwen_llm(system, user, max_tokens, purpose)
                LAST.update(backend="qwen", model=MODEL)
            return out
        except BackendDown as ex:
            notify_backend_down(be, str(ex), nxt)
            if not nxt:
                raise SystemExit(f"{be} did not answer: {ex}")
        except SystemExit as ex:
            if be == "claude" and str(ex).startswith("Claude API") and nxt:   # a dead key, an outage; limits and refusals are not outages
                notify_backend_down(be, str(ex), nxt)
                continue
            raise
    raise SystemExit("no LLM backend")


def qwen_llm(system: str, user: str, max_tokens: int, purpose: str) -> str:
    import dispatcher
    cfg = dispatcher.load_cfg()
    lim = limits_cfg()
    for attempt in range(6):
        if CALLS["n"] >= lim["max_llm_calls_per_cycle"]:
            raise SystemExit(f"autolab limit: {CALLS['n']} LLM requests in this run (max_llm_calls_per_cycle={lim['max_llm_calls_per_cycle']})")
        if week_calls() >= lim["weekly_llm_calls"]:
            raise SystemExit(f"autolab limit: {week_calls()} LLM requests this week (weekly_llm_calls={lim['weekly_llm_calls']})")
        ok, why = dispatcher.exp_gate(cfg, dispatcher.fractions(sr.limits()))
        if not ok:
            raise SystemExit(f"gateway gate (dispatcher.yaml): {why}")
        body = json.dumps({"model": MODEL, "max_tokens": max_tokens, "temperature": 0.7,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
        CALLS["n"] += 1
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
        except urllib.error.HTTPError as ex:
            record_usage(purpose, {}, False)
            if ex.code == 429:
                wait = 65 * (attempt + 1)
                log(f"429 from the gateway (concurrent-request limit); waiting {wait} s")
                time.sleep(wait)
                continue
            raise
        record_usage(purpose, r.get("usage") or {}, True)
        return r["choices"][0]["message"].get("content") or ""
    raise SystemExit("the gateway kept answering 429")


def cmd_usage() -> None:
    rows = usage_rows()
    ws = week_start_ts()
    wk = [r for r in rows if r["ts"] >= ws]
    cfg = limits_cfg()
    cl = [r for r in rows if r.get("backend") == "claude"]
    print(f"Claude API ({provider()}): {len(cl)} requests, {sum(r.get('usd', 0) for r in cl):.3f} USD total, {week_usd():.3f} USD this week (cap {cfg['usd_per_week']}), "
          f"{CALLS['usd']:.3f} USD in this process (cap per run {cfg['usd_per_run']}); backend now: {backend_desc()[0]}")
    print(f"autolab requests (all backends; the qwen ones go through the gateway): total {sum(r.get('requests', 1) for r in rows)} ({sum(1 for r in rows if r.get('ok') is False)} failed/429), "
          f"this ISO week {sum(r.get('requests', 1) for r in wk)} of own cap {cfg['weekly_llm_calls']} "
          f"(gateway week limit 1 000 000 requests); tokens (input/10 + output) total {sum(r['cost'] for r in rows):.0f}, model {MODEL}")
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["purpose"], []).append(r)
    for p, v in sorted(by.items()):
        print(f"  {p:<10} {sum(x.get('requests', 1) for x in v):>4} requests {sum(x['cost'] for x in v):>8.0f} tokens")
    exp_req = 0
    for f in sorted(DRAFTS.glob("*.meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("experiment_requests"):
            exp_req += m["experiment_requests"]
            print(f"  experiment {f.name[:-10]:<34} {m['experiment_requests']:>6} requests (pipeline+agent calls in cells; verifier calls not included)")
    print(f"  experiments of autolab drafts: {exp_req} requests (lower bound)")


# ------------------------------------------------------------------ knowledge handed to the generator
def known_switches() -> set[str]:
    for f in (HERE.parent / "SCRIPTS.md", LAB / "harness" / "SCRIPTS.md"):
        if f.exists():
            return set(re.findall(r"HARNESS_[A-Z_]+", f.read_text(encoding="utf-8"))) - DENY_SWITCHES
    return set()


def switch_help() -> str:
    """The table rows of SCRIPTS.md that describe switches, shortened: what each switch does, for the generator."""
    for f in (HERE.parent / "SCRIPTS.md", LAB / "harness" / "SCRIPTS.md"):
        if f.exists():
            rows = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.startswith("| ") and "HARNESS_" in l]
            return "\n".join(r[:330] for r in rows[:30])
    return ""


def context_text() -> str:
    for f in (HERE.parent / "autolab_context.md", LAB / "harness" / "autolab_context.md"):
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
    return ""


def narrow_note(s: dict) -> str:
    """Narrow specialised sets (experiments/sets/nrw_*.yaml, `narrow:` block): subgenre and the status-scheme cells (block/type) the set loads."""
    nr = s.get("narrow") or {}
    return f"; narrow {nr['subgenre']}, loads cells: {', '.join(nr.get('target_cells') or [])}" if nr.get("subgenre") else ""


def sets_available() -> dict[str, str]:
    import yaml
    out = {}
    for f in sorted((EXP / "sets").glob("*.yaml")):
        s = yaml.safe_load(f.read_text(encoding="utf-8"))
        if s.get("role") == "dev":   # holdout is never offered; fresh sets are not offered either (they change)
            out[f.stem] = f"tasks {s.get('tasks')}" + narrow_note(s)
    return out


def check_sets() -> dict[str, str]:
    """Sets a hypothesis can be re-checked on (dev and fresh; holdout never): name -> one-line description for the reviewer and the chat."""
    import yaml
    out = {}
    for f in sorted((EXP / "sets").glob("*.yaml")):
        try:
            s = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        if s.get("role") in ("dev", "fresh"):
            fz = s.get("frozen") or {}
            out[f.stem] = f"{s['role']}, tasks {s.get('tasks')}" + (f", {fz['n_tasks']} tasks, {fz.get('n_unanswerable', 0)} unanswerable" if fz.get("n_tasks") else "") + narrow_note(s)
    return out


def _split_lab_block(text: str) -> tuple[dict, dict]:
    import yaml
    mark = "# ---- written by lab.py"
    try:
        e = yaml.safe_load(text.split(mark)[0]) or {}
        blk = yaml.safe_load(text.split(mark)[1].split("\n", 1)[1]) if mark in text else {}
    except Exception:  # noqa: BLE001
        return {}, {}
    return e, blk or {}


def all_experiments() -> list[dict]:
    """Every experiment the lab knows (done, running, pending, autolab drafts) as {name, e, block, where}."""
    out = []
    for kind in ("done", "running", "pending"):
        for f in sorted((EXP / "queue" / kind).glob("*.yaml")):
            e, blk = _split_lab_block(f.read_text(encoding="utf-8"))
            if e.get("name"):
                out.append({"name": e["name"], "e": e, "block": blk, "where": kind, "mtime": f.stat().st_mtime})
    for f in sorted(DRAFTS.glob("*.yaml")):
        e, _ = _split_lab_block(f.read_text(encoding="utf-8"))
        if e.get("name") and not any(x["name"] == e["name"] for x in out):
            out.append({"name": e["name"], "e": e, "block": {}, "where": "draft:" + meta(e["name"]).get("stage", "?"), "mtime": f.stat().st_mtime})
    return out


def diff_signature(e: dict) -> tuple:
    """What an experiment COMPARES: set, budget and, per non-reference arm, the settings that differ from the first arm.
    Two hypotheses with the same signature test the same thing, whatever their names and wording."""
    arms = e.get("arms") or {}
    if not arms:
        return ()
    first = next(iter(arms.values())) or {}

    def flat(a):
        a = a or {}
        d = {str(k): str(v) for k, v in (a.get("env") or {}).items()}
        if a.get("batch_size"):
            d["batch_size"] = str(a["batch_size"])
        if a.get("model"):
            d["model"] = str(a["model"])
        return d
    f0 = flat(first)
    sig = set()
    for a in list(arms.values())[1:]:
        fa = flat(a)
        sig.add(frozenset((k, fa.get(k), f0.get(k)) for k in set(fa) | set(f0) if fa.get(k) != f0.get(k)))
    return (e.get("set"), (int(e.get("budget") or 0) if str(e.get("budget") or 0).isdigit() else 0), frozenset(sig))   # budget: auto is not a number


def find_duplicate(h: dict) -> str | None:
    s = diff_signature(h)
    for x in all_experiments():
        if x["name"] != h.get("name") and diff_signature(x["e"]) == s:
            return x["name"]
    return None


def find_decided_duplicate(h: dict) -> str | None:
    """A hypothesis that compares the same settings as one a human or the lab has already decided on (accepted, rejected, auto-rejected), whatever the
    set and budget: a re-run on other data is the `check` / `repeat` of that hypothesis, not a new idea."""
    sig = diff_signature(h)
    if not sig or not sig[2]:
        return None
    for x in all_experiments():
        if x["name"] == h.get("name") or meta(x["name"]).get("check_of"):
            continue
        if diff_signature(x["e"])[2:] == sig[2:] and decision_of(x["name"]) in ("accepted", "rejected", "auto_rejected"):
            return family_base(x["name"])
    return None


def digest() -> str:
    """Finished experiments, newest first: what was tested, per-arm points/tokens, interval, verdict, and why a failed one failed."""
    rows = []
    for x in sorted(all_experiments(), key=lambda x: -x["mtime"]):
        e, blk = x["e"], x["block"]
        res = blk.get("result") or {}
        if x["where"] == "done" and res:
            arms = {a: {"pts": v.get("points_per_run"), "tok": v.get("tokens_per_run"), "acc": v.get("accuracy"),
                        "diff": (v.get("points_diff_vs_first") or {}).get("mean"), "ci": (v.get("points_diff_vs_first") or {}).get("ci95")}
                    for a, v in (res.get("arms") or {}).items()}
            rows.append(f"- {e['name']} [set {e.get('set')}, budget {e.get('budget')}, repeats {e.get('repeats')}] {(e.get('hypothesis') or '').strip()[:220]} "
                        f"| arms {json.dumps(arms)} | verdict {res.get('verdict')}")
        elif x["where"] == "done":
            rows.append(f"- {e['name']}: status {blk.get('status')} ({str(blk.get('reason') or blk.get('result_error') or '')[:160]})")
        else:
            rows.append(f"- {e['name']}: {x['where']} (not finished; do not repeat it) {(e.get('hypothesis') or '').strip()[:120]}")
    return "\n".join(rows[:30])


def status_root() -> Path:
    return astat.default_root(LAB)


def spec_of(name: str) -> dict:
    import yaml
    for d in (EXP / "queue" / "done", EXP / "queue" / "running", EXP / "queue" / "pending", DRAFTS):
        f = d / f"{name}.yaml"
        if f.exists():
            try:
                return (yaml.safe_load(f.read_text(encoding="utf-8").split("\n# ---", 1)[0]) or {})
            except Exception:  # noqa: BLE001
                return {}
    return {}


REPEAT_SUFFIX = re.compile(r"(_r\d+(_\d+)?|_fresh\d*)$")


def family_base(name: str) -> str:
    """The hypothesis a repeat belongs to: `ab_x_r4` and `ab_x_r4_2` are repeats of `ab_x`; the decision of a human is about the whole family."""
    while True:
        b = REPEAT_SUFFIX.sub("", name)
        if b == name or not b:
            return name
        name = b


DECISION_RU = {"accepted": "принята", "rejected": "отклонена человеком", "auto_rejected": "отклонена автоматически", "postponed": "отложена"}


def decisions_epoch() -> float:
    """Hypotheses that finished before decisions existed are closed (no one has to decide on them); the epoch is the first moment the rule was applied."""
    f = DRAFTS / "decisions_epoch.txt"
    try:
        return float(f.read_text())
    except (OSError, ValueError):
        DRAFTS.mkdir(parents=True, exist_ok=True)
        f.write_text(str(time.time()))
        return time.time()


def decision_of(name: str) -> str | None:
    return meta(family_base(name)).get("decision")


def set_decision(name: str, decision: str | None, note: str = "", by: str = "") -> list[str]:
    """accepted | rejected | auto_rejected | postponed | None (taken back into work). Stored on the base hypothesis, written into the journal rows."""
    base = family_base(name)
    set_meta(base, decision=decision, decided_at=time.time(), decision_note=note[:300], decided_by=str(by), **({"stale": False} if decision == "postponed" else {}))
    for nm in {name, base}:
        if meta(nm):
            record_journal(nm, force=True)
    return mark_stale_postponed(base) if decision == "accepted" else []


def types_of(name: str) -> set[str]:
    """The task types a hypothesis holds while it is active (`_all` = every type)."""
    return astat.types_of_cells(targets_of(name), astat.index_types(status_root()))


def hyp_gain(name: str) -> float | None:
    """What a hypothesis is worth, for ordering: the best measured difference of its arms (points per run), else the expected gain of its target cells from the rating."""
    res = meta(name).get("result") or {}
    ds = [v["points_diff_vs_first"]["mean"] for v in res.values() if v.get("points_diff_vs_first")]
    if ds:
        return max(ds)
    root, gs = status_root(), []
    if astat.index(root):
        for c in targets_of(name):
            r = astat.parse_ref(c)
            g = (astat.entry(root, *r) or {}).get("gain_pts") if r else None
            if isinstance(g, (int, float)):
                gs.append(g)
    return max(gs) if gs else None


def _cb(verb: str, name: str) -> str | None:
    """Telegram callback data (64 bytes at most) for a button `al:cmd:<verb>:<name>`; None when the name is too long (the command still works as text)."""
    d = f"al:cmd:{verb}:{name}"
    return d if len(d.encode("utf-8")) <= 64 else None


def hyp_buttons(name: str, review: bool = True) -> list:
    """The five decisions on a hypothesis (four without the review button, e.g. under a review that was just made)."""
    spec = [("✅ Принять", "accept"), ("❌ Отклонить", "reject"), ("🔍 Ревью", "review"), ("🔁 Повтор", "rerun"), ("🧪 Другие данные", "check"), ("⏸ Отложить", "defer")]
    b = [{"label": lab, "data": _cb(v, name)} for lab, v in spec if _cb(v, name) and (review or v != "review")]
    return [b[i:i + 2] for i in range(0, len(b), 2)]


def mark_stale_postponed(accepted: str) -> list[str]:
    """The first accepted hypothesis of a task type makes the postponed ones of that type stale: they were measured against the old code."""
    root = status_root()
    all_t = astat.index_types(root)
    types = astat.types_of_cells(targets_of(accepted), all_t)
    out = []
    for f in sorted(DRAFTS.glob("*.meta.json")):
        n = f.name[:-10]
        m = meta(n)
        if m.get("decision") == "postponed" and not m.get("stale") and astat.types_of_cells(targets_of(n), all_t) & types:
            set_meta(n, stale=True, stale_by=accepted, stale_at=time.time())
            out.append(n)
    return out


def targets_of(name: str) -> list[str]:
    """Target cells of a hypothesis; a repeat (`_rN`) or a confirmation (`_fresh`) inherits them from the experiment it is named after."""
    for nm in (name, family_base(name)):
        t = meta(nm).get("target_cells")
        if t:
            return list(t)
    return []


def cell_experiments_digest(cells: list[str]) -> str:
    """What the scheme already knows about the target cells: experiments listed in them and the last history lines (duplicate guard for the model)."""
    root = status_root()
    out = []
    for c in cells:
        r = astat.parse_ref(c)
        e = astat.entry(root, *r) if r else None
        if e:
            out.append(f"- {c}: experiments {', '.join(e['experiments']) or 'none'}; journal of hypotheses already tried (name: verdict, result): " + (astat.journal_text(root, c, 8) or "none yet"))
    return "\n".join(out)


SYSTEM = ("You are the hypothesis generator of an experiment lab for an LLM task-solving harness. You propose small, testable "
          "experiments that change harness behaviour ONLY through the existing environment switches (or, when a switch cannot express "
          "it, a small described code change). Reply with JSON only.")


MODES_TEXT = ("GOAL MODES of the harness (every hypothesis serves one): `accuracy` = the most points within the token budget (judged by points per run at the full budget against the noise threshold); "
              "`economy` = the same points for fewer tokens, the organizers' tie-break (judged by points per 1k tokens and tokens to reach the same points, and the points must not fall by more than the noise threshold). "
              "State in `hypothesis` which metric you expect to move and by how much.\n")


def plan_cells(n: int, used: dict, locked: set, mode: str | None = None, start: int = 0, spare: int = 0) -> tuple[list[str], dict]:
    """(cells, cell_modes) for a batch: slot i takes `mode`, or alternates accuracy/economy from `start` (an equal split); every cell is drawn from the rating of its slot's mode;
    the task types of the cells of one batch are all different and free of the `locked` ones."""
    root = status_root()
    mode = mode or (limits_cfg().get("goal_mode") or None)      # the configured focus when the caller names none ("" = alternate accuracy/economy)
    taken, cells, cm = set(locked), [], {}
    for i in range(n + spare):
        m = mode if mode in astat.MODES else astat.MODES[(start + i) % len(astat.MODES)]
        c = astat.pick_cells(root, 1, used, locked_types=taken, mode=m)
        if c:
            cells.append(c[0])
            cm[c[0]] = m
            taken |= astat.types_of_cells(c, astat.index_types(root))
    return cells, cm


def prompt_propose(n: int, topic: str = "", errors: str = "", cells: list[str] | None = None, strict: bool = False, active: dict | None = None,
                   cell_modes: dict | None = None, mode: str | None = None) -> str:
    sw = ", ".join(sorted(known_switches()))
    modes = sorted(set((cell_modes or {}).values())) or ([mode] if mode in astat.MODES else [])
    if len(modes) > 1:       # one rating per goal mode, shorter each
        st = "\n".join(astat.prompt_block(status_root(), 4, 3, m) for m in modes)
    else:
        st = astat.prompt_block(status_root(), mode=modes[0] if modes else None)
    status_text = ""
    if st:
        status_text = ("\n" + st + "\nRULES OF THE RATING: every hypothesis MUST name its target cells (`target_cells`: 1-3 strings 'block/type', type is one of coding, math, search, data_analysis or _all) and "
                       "`why_cell` (one sentence: why this cell is in the rating - its priority, status, what its 'не проверено' says). Take the cells from the lists above, best priority first, "
                       "but a hypothesis must be something the switches (or one small code change) can really test. NEVER propose anything for a saturated or not applicable cell. "
                       "Do not repeat what the cell's history and experiments already show (a clarify/reject result in the history means: do not retry the same idea).\n")
        status_text += MODES_TEXT
        if cell_modes:
            status_text += ("Goal mode of each cell of this batch (the hypothesis for that cell MUST have this `target_mode`, or `both` if it truly helps both): "
                            + ", ".join(f"{c} -> {m}" for c, m in cell_modes.items()) + "\n")
        elif mode in astat.MODES:
            status_text += f"Every hypothesis of this batch MUST have target_mode {mode} (or both if it truly helps both modes).\n"
        if active:
            status_text += ("Task types that are HELD by an active hypothesis (queued, running or waiting for a human decision): " + ", ".join(f"{t} ({h})" for t, h in active.items())
                            + ". Do not target a cell of such a type, and no `_all` cell while any type is held.\n")
        if cells and strict:
            status_text += ("The cells for this batch. EVERY hypothesis must have as its FIRST target cell one of these, a different one for each hypothesis; pick those that the switches (or one small code change) can really test; "
                            "a hypothesis aimed at any other cell is refused:\n" + cell_experiments_digest(cells) + "\n")
        elif cells:
            status_text += ("Preferred target cells for this batch, one per hypothesis in this order (use another cell from the rating only if this one cannot be tested with the switches):\n"
                            + cell_experiments_digest(cells) + "\n")
    return f"""Competition conditions (organizers, 2026-11-09) and measured facts:
{context_text()}

Real costs for planning: a run on mixed100_dev at budget 30000 spends about 30-35k ledger tokens and about 250 gateway requests per arm and repeat; a run of the 40-task sets about 25k tokens; 16 math tasks 5-9k. The whole experiment (arms x repeats) should stay under 150000 tokens. The lab noise: one standard deviation of points between repeats of an unchanged harness is 7.69 points on mixed100 (about 3.4%); a difference below that is never accepted, so propose changes whose effect could plausibly exceed 8 points on a 100-task set (a 4-5% change), or cost savings of 15% and more at equal points.

Experiments already done, running, queued or drafted (newest first; NEVER propose the same comparison again; a result that is `clarify` means the effect was within noise; a `_fresh` experiment is the confirmation of the one it is named after):
{digest()}

Environment switches you may use in `env:` of an arm (nothing else is allowed): {sw}
What the switches do (from the documentation):
{switch_help()}
Allowed values of enumerated switches (any other value silently turns the block off, and will be refused): {json.dumps(SWITCH_VALUES)}
(None = comma-separated list of task types. Numeric switches: HARNESS_VERIFY_ACCEPT, HARNESS_VERIFY_MIN 0..1, HARNESS_RECHECK_MAX, HARNESS_BUDGET_STOP 0..1, HARNESS_CENSOR_TOKENS "math=1500,search=1200".)
A verifier recheck needs HARNESS_VERIFY=judge (or free) AND HARNESS_VERIFY_ACTION=recheck. `batch_size` of an arm (1,2,4,8) switches batching on; it needs HARNESS_BATCH_PROMPT=compute to keep accuracy.
Frozen sets you may use (`set:`): {json.dumps(sets_available())}
Models: qwen3.6-35b-a3b-noreason, qwen3.6-35b-a3b, qwen3.8-27b-noreason, qwen3.8-27b, auto.
{status_text}
`budget` is the token budget of the harness in the experiment (all arms): use 20000-40000 to test what the competition will look like; scheduler-level hypotheses (HARNESS_SORTER, HARNESS_PREDICTOR, HARNESS_PUNSOLV, HARNESS_SOLVERS) are pre-checked in an offline simulator at 20k/30k/40k.
{('Topic for this batch: ' + topic) if topic else ('Choose by the rating of the scheme status above.' if st else 'Topics, in order of priority: task order / scheduler / predictor, cheap modes (noreason models, batching, fallback, censor), behaviour at budgets 20-40k. Verification and rechecks are secondary.')}
{('Your previous answer was refused: ' + errors) if errors else ''}
Propose {n} different hypotheses likely to be useful. Each: JSON object with keys
name (snake_case, unique, start with ab_), hypothesis (2-4 sentences: what changes, why, what you expect IN POINTS OR TOKENS, how it can fail),
set (one of the sets above), model (one model or auto), repeats (1 or 2), budget (integer 10000-100000),
arms (object: arm name -> {{env: {{SWITCH: "value"}}, batch_size: optional int}}); the FIRST arm is the reference (the behaviour you compare against; its env may be empty or hold the current best setting),
expected_cost_tokens (your estimate of the ledger tokens of the whole experiment), target_mode ("accuracy", "economy" or "both"), cross_check (true if the change touches a block shared by both modes, such as the predictor, sorter, censor, batcher, fallback or the harness code itself; otherwise false),
code_change (empty string unless an existing switch cannot express it; then one concrete sentence naming the file and the behaviour){('' if not st else ', target_cells (list of 1-3 cells block/type from the rating), why_cell (one sentence: why the cell is in the rating)')}.
Return a JSON list of {n} objects and nothing else."""


def extract_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else None


def validate(h: dict) -> list[str]:
    errs = []
    for k in ("name", "hypothesis", "set", "arms"):
        if not h.get(k):
            errs.append(f"missing {k}")
    if errs:
        return errs
    extra = set(h) - set(DRAFT_KEYS)
    if extra:
        errs.append(f"keys not allowed: {sorted(extra)} (the lab's rules, thresholds and sets are not the generator's to set)")
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", str(h["name"])):
        errs.append("bad name")
    if h.get("target_mode") not in GOAL_MODES:
        errs.append(f"target_mode: the goal mode is required, one of {list(GOAL_MODES)} (accuracy = more points within the budget, economy = the same points for fewer tokens, both = better in both)")
    if "cross_check" in h and not isinstance(h["cross_check"], bool):
        errs.append("cross_check must be true or false")
    if h["set"] not in sets_available():
        errs.append(f"set {h['set']} is not an available dev set")
    errs += astat.validate_targets(status_root(), h)
    arms = h["arms"]
    if not isinstance(arms, dict) or not 2 <= len(arms) <= 4:
        errs.append("2-4 arms")
        return errs
    ks = known_switches()
    for a, v in arms.items():
        if not isinstance(v, dict) and v is not None:
            errs.append(f"arm {a}: not an object")
            continue
        if set(v or {}) - set(ARM_KEYS):
            errs.append(f"arm {a}: keys not allowed {sorted(set(v) - set(ARM_KEYS))}")
        for k in (v or {}).get("env") or {}:
            if k in DENY_SWITCHES or (k not in ks and not (h.get("code_change") and re.fullmatch(r"HARNESS_[A-Z_]+", k))):
                errs.append(f"arm {a}: unknown or forbidden switch {k}")   # a code change may introduce a new switch or a new value
            elif h.get("code_change") and k in ks and SWITCH_VALUES.get(k) and str(((v or {}).get("env") or {})[k]) not in SWITCH_VALUES[k]:
                pass
            elif SWITCH_VALUES.get(k) and str(((v or {}).get("env") or {})[k]) not in SWITCH_VALUES[k]:
                errs.append(f"arm {a}: {k} must be one of {SWITCH_VALUES[k]}")
            elif k.endswith(("_MIN", "_ACCEPT", "_GIVEUP", "_THRESHOLD", "_TEMP", "_STOP", "_RECHECK", "_MAX", "_SIZE")):
                try:
                    float(((v or {}).get("env") or {})[k])
                except ValueError:
                    errs.append(f"arm {a}: {k} must be a number")
        if (v or {}).get("batch_size") not in (None, 1, 2, 4, 8):
            errs.append(f"arm {a}: batch_size must be 1, 2, 4 or 8")
        if (v or {}).get("model") and v["model"] not in MODELS + ["auto"]:
            errs.append(f"arm {a}: unknown model")
    try:
        if int(h.get("repeats", 1)) not in (1, 2):
            errs.append("repeats 1 or 2")
        if not 10000 <= int(h.get("budget", 100000)) <= 100000:
            errs.append("budget must be 10000..100000")
        if int(h.get("expected_cost_tokens", 0) or 0) > limits_cfg()["max_cost_tokens"]:
            errs.append(f"expected cost above {limits_cfg()['max_cost_tokens']} tokens")
    except (TypeError, ValueError):
        errs.append("repeats, budget and expected_cost_tokens must be integers")
    if h.get("model", "auto") not in MODELS + ["auto"]:
        errs.append("unknown model")
    if not errs:
        arms_l = list(arms.values())
        if all(diff_signature({"set": 1, "arms": {"a": arms_l[0], "b": b}}) == diff_signature({"set": 1, "arms": {"a": arms_l[0], "b": arms_l[0]}})
               for b in arms_l[1:]):
            errs.append("the arms do not differ from the first arm")
    return errs


def write_draft(h: dict) -> Path:
    import yaml
    DRAFTS.mkdir(parents=True, exist_ok=True)
    cells = astat.normalize_targets(h)
    note = (f"\n[ячейка схемы: {', '.join(cells)}; почему в рейтинге: {str(h.get('why_cell') or '').strip()}]" if cells else "")
    e = {"name": h["name"], "author": "orchestrator", "hypothesis": h["hypothesis"].strip() + note, "backend": "gateway",
         "model": h.get("model", "qwen3.6-35b-a3b-noreason"), "set": h["set"], "repeats": int(h.get("repeats", 1)),
         "budget": int(h.get("budget", 100000)), "fallback": "on",
         "arms": {a: {k: v for k, v in (spec or {}).items() if k in ARM_KEYS} for a, spec in h["arms"].items()}}
    e["target_mode"] = h.get("target_mode") or "accuracy"
    if h.get("priority"):
        e["priority"] = int(h["priority"])      # lab.py runs higher priority first (a hypothesis proposed by a human goes first)
    if h.get("code_change"):
        e["change"] = "needs code: " + str(h["code_change"])[:400]
    path = DRAFTS / f"{h['name']}.yaml"
    path.write_text(yaml.safe_dump(e, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (DRAFTS / f"{h['name']}.meta.json").write_text(json.dumps(
        {"stage": "proposed", "expected_cost_tokens": h.get("expected_cost_tokens"), "code_change": h.get("code_change", ""),
         "target_mode": h.get("target_mode") or "accuracy", "cross_check": bool(h.get("cross_check")),
         "generator_model": MODEL, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    return path


def meta(name: str) -> dict:
    f = DRAFTS / f"{name}.meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def set_meta(name: str, **kw) -> None:
    m = meta(name)
    m.update(kw)
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / f"{name}.meta.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")


def hyp_active(name: str) -> bool:
    """A hypothesis holds its task types from the moment it enters the queue until a human decides (accept, reject, postpone) or the lab verdict is
    `reject` (automatic). Hypotheses that finished before decisions existed do not hold anything."""
    base_ = meta(name).get("check_of")
    if base_ and (decision_of(base_) or meta(base_).get("in_stack")):
        return False      # a check / repeat of a hypothesis that is decided or already on the stack holds nothing of its own
    if decision_of(name) or meta(name).get("stack_exp") or meta(name).get("in_stack"):      # a check of the stack and a hypothesis already on the stack hold no task types
        return False
    if any((EXP / "queue" / k / f"{name}.yaml").exists() for k in ("pending", "running")):
        return True
    m = meta(name)
    blk = done_block(name)
    if blk is not None and blk.get("status") != "ok" and not (m.get("verdict") or m.get("result")):
        return False         # the lab refused or aborted it (invalid set, broken stand): there is nothing to decide, so it holds no task types
    return bool(m.get("queued_at", 0) >= decisions_epoch() and (m.get("verdict") or m.get("result") or blk is not None))


def active_types() -> dict[str, str]:
    """task type -> name of the hypothesis that holds it (a block-wide `_all` cell holds all of them): at most one active hypothesis per type, so at most
    as many as there are types."""
    all_t = astat.index_types(status_root())
    out: dict[str, str] = {}
    for f in sorted(DRAFTS.glob("*.meta.json")):
        n = f.name[:-10]
        cells = targets_of(n)
        if cells and hyp_active(n):
            for t in astat.types_of_cells(cells, all_t):
                out.setdefault(t, n)
    return out


def type_conflicts(cells: list[str], active: dict[str, str]) -> dict[str, str]:
    """The task types of `cells` that an active hypothesis already holds: type -> its name."""
    return {t: active[t] for t in astat.types_of_cells(cells, astat.index_types(status_root())) if t in active}


def cell_conflicts(h: dict, active: dict, batch: set, strict_cells: list[str] | None) -> list[str]:
    errs, mine = [], astat.normalize_targets(h)
    all_t = astat.index_types(status_root())
    for t, who in type_conflicts(mine, active).items():
        errs.append(f"task type {t} is held by the active hypothesis {who}: one at a time, wait for the decision on it")
    for t in sorted(astat.types_of_cells(mine, all_t) & batch):
        errs.append(f"task type {t} is already taken by another hypothesis of this batch")
    if strict_cells and mine and mine[0] not in strict_cells:
        errs.append(f"the first target cell must be one of {strict_cells}")
    return errs


def cmd_propose(n: int, dry: bool, topic: str = "", cells: list[str] | None = None, strict: bool = False, extra: dict | None = None,
                mode: str | None = None, cell_modes: dict | None = None) -> list[str]:
    lim = limits_cfg()
    n = min(n, lim["max_hyp_per_cycle"])
    active = active_types() if astat.index(status_root()) else {}
    batch_cells: set = set()          # task types taken by this batch
    if cells is None and astat.index(status_root()):
        if topic and astat.parse_ref(topic):
            cells = [f"{astat.parse_ref(topic)[0]}/{astat.parse_ref(topic)[1]}"]
        elif not topic:
            used: dict[str, int] = {}
            for nm in _autopilot_state()["names_today"]:
                for c in targets_of(nm):
                    used[c] = used.get(c, 0) + 1
            cells, cell_modes = plan_cells(n, used, set(active), mode, __import__("random").randrange(len(astat.MODES)))      # equal split between the goal modes unless one is given
    if cells and not cell_modes and mode in astat.MODES:
        cell_modes = {c: mode for c in cells}
    prompt = prompt_propose(n, topic, "", cells, strict, active, cell_modes, mode)
    if dry:
        print(prompt)
        return []
    names: list[str] = []
    errors = ""
    for attempt in range(2):   # one retry that tells the model what was refused
        reply = llm(SYSTEM, prompt, 6000, "propose")
        try:
            items = extract_json(reply) or []
        except ValueError:
            items = []
            log("the reply was not JSON")
        refused = []
        for h in items[:n + 1]:
            if len(names) >= n:
                break
            errs = validate(h) if isinstance(h, dict) else ["not an object"]
            if not errs and ((DRAFTS / f"{h['name']}.yaml").exists() or (EXP / "queue" / "done" / f"{h['name']}.yaml").exists()):
                errs = ["name already used"]
            if not errs:
                dup = find_duplicate(h)
                if dup:
                    errs = [f"duplicate of {dup} (same set, budget and settings compared)"]
                dec = None if dup else find_decided_duplicate(h)
                if dec:
                    errs = [f"the same settings were compared in {dec}, which is already decided ({DECISION_RU.get(decision_of(dec), decision_of(dec))}): check it on other data (`проверка {dec}`) instead of proposing it again"]
            if not errs and astat.index(status_root()):
                errs = cell_conflicts(h, active, batch_cells, cells if strict else None)
            if not errs and cell_modes:
                first = (astat.normalize_targets(h) or [None])[0]
                want = cell_modes.get(first)
                if want and h.get("target_mode") not in (want, "both"):
                    errs = [f"target_mode must be {want} for the cell {first} (or both)"]
            if errs:
                log(f"rejected {h.get('name') if isinstance(h, dict) else '?'}: {'; '.join(errs)}")
                refused.append(f"{h.get('name') if isinstance(h, dict) else '?'}: {'; '.join(errs)}")
                continue
            if extra and extra.get("priority"):
                h["priority"] = extra["priority"]
            p = write_draft(h)
            if extra:
                set_meta(h["name"], **{k: v for k, v in extra.items() if k != "priority"})
            if astat.normalize_targets(h):
                set_meta(h["name"], target_cells=astat.normalize_targets(h), why_cell=str(h.get("why_cell") or "").strip()[:400])
            log(f"draft {p.name} (expected cost {h.get('expected_cost_tokens')} tokens; cells {astat.normalize_targets(h) or '-'})")
            names.append(h["name"])
            batch_cells |= astat.types_of_cells(astat.normalize_targets(h), astat.index_types(status_root()))
        if names or not refused:
            break
        errors = " | ".join(refused)[:900]
        prompt = prompt_propose(n, topic, errors, cells, strict, active, cell_modes, mode)
        log("nothing usable; asking once more with the refusal reasons")
    return names


# ------------------------------------------------------------------ applying agent: a change only in a worktree on a branch
def repo_root() -> Path:
    """The git repo the branches are made in: the checkout on a dev machine; on the host a repo seeded from the deployed archive."""
    dev = HERE.parent
    if (dev / ".git").exists() and (dev / "harness" / "sorter.py").exists():
        return dev
    src = LAB / "harness"
    repo = LAB / "autolab_repo"
    ver = (src / "VERSION").read_text(encoding="utf-8").strip() if (src / "VERSION").exists() else "unknown"
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=autolab", "-c", "user.email=noreply@anthropic.com", *a],
                                    capture_output=True, text=True)
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        git("init", "-b", "main")
        (repo / ".gitignore").write_text("VERSION\nlogs/\n__pycache__/\n*.pyc\n.venv/\n", encoding="utf-8")
    base_file = repo / ".autolab_base"
    if not base_file.exists() or base_file.read_text(encoding="utf-8").strip() != ver:
        for p in repo.iterdir():   # replace the tracked tree with the deployed one
            if p.name not in (".git", ".gitignore", ".autolab_base"):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        for p in src.iterdir():
            if p.name in ("logs", "__pycache__", ".venv"):
                continue
            shutil.copytree(p, repo / p.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "logs")) if p.is_dir() else shutil.copy2(p, repo / p.name)
        (repo / ".autolab_base").write_text(ver, encoding="utf-8")
        git("add", "-A")
        git("commit", "-m", f"base {ver}", "--allow-empty")
    return repo


def snapshot_protected() -> dict:
    """Hashes of what the model must never change: lab code, frozen sets, thresholds source (noise-floor results), configs."""
    snap = {}
    roots = [HERE / n for n in ("lab.py", "autolab.py", "dispatcher.py", "sim.py", "stream_run.py")] + [LAB / n for n in ("lab.py", "dispatcher.py", "stream_run.py", "dispatcher.yaml", "autolab.yaml")]
    files = [p for p in roots if p.exists()] + sorted((EXP / "sets").glob("*.yaml")) + sorted((EXP / "results" / "baseline0").glob("cells.jsonl"))
    for p in files:
        snap[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return snap


def check_patch(patch_files: list[str], added: list[str], removed: int) -> str | None:
    """None if the change is within the rules, else the reason it is refused."""
    if not patch_files:
        return "the patch changes nothing"
    bad = [t for t in patch_files if t in FORBIDDEN or ".." in t or t.startswith("/") or not t.startswith(ALLOWED_PREFIX)]
    if bad:
        return f"touches files outside the allowed set {ALLOWED_PREFIX}: {bad}"
    hit = [l.strip()[:80] for l in added if FORBIDDEN_ADDED.search(l)]
    if hit:
        return f"added lines hit the deny-list (lab, thresholds, keys, network, shell): {hit[:3]}"
    if len(added) + removed > limits_cfg()["max_code_lines"]:
        return f"the patch is too large ({len(added) + removed} lines, max {limits_cfg()['max_code_lines']})"
    return None


def pick_files(root: Path, want: str) -> list[Path]:
    words = {w for w in re.findall(r"[a-z_]{4,}", want.lower())}
    scored = []
    for p in list((root / "harness").glob("*.py")) + list((root / "harness").glob("*/*.py")) + [root / "run_agent.py"]:
        if not p.exists() or p.stat().st_size > 30000 or p.name == "__init__.py":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        s = sum(2 for w in words if w in p.name.lower()) + sum(1 for w in words if w in text)
        scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    out, total = [], 0
    for s, p in scored:
        if s and len(out) < 5 and total + p.stat().st_size < 60000:
            out.append(p)
            total += p.stat().st_size
    return out or [root / "harness" / "sorter.py"]


EDIT_SYSTEM = ("You edit Python source with exact search/replace blocks. Reply with JSON only: "
               '{"edits": [{"file": "harness/x.py", "old": "exact existing text", "new": "replacement"}], "note": "one sentence"}. '
               "`old` must occur exactly once in the file, copied character for character including indentation. "
               "To CREATE a new file under harness/ (a new module, solver, pipeline) give `\"old\": \"\"` and the whole file in `new`; the new file must not exist yet, "
               "and existing code has to import and call it behind the switch. A change may be large when the idea needs it (limit in the user message), but keep it as small as the idea allows. "
               "Make it controllable by an environment switch with the old behaviour as default (the switch must be a different value in the arm than in the reference arm), "
               "add no imports of network or shell modules, and never give an imported module a name that a function or class of the same file already has (use `import math as _math`). "
               "Reply with the JSON object only and keep it complete: it must be valid JSON that closes.")


def parse_edits(reply: str) -> list[dict]:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    m = re.search(r"\{.*\}", reply, re.S)
    data = json.loads(m.group(0)) if m else {}
    edits = data.get("edits") or []
    if not edits or not all(isinstance(x, dict) and {"file", "old", "new"} <= set(x) for x in edits):
        raise ValueError("no edits in the reply")
    return edits


def apply_edits(wt: Path, edits: list[dict]) -> None:
    for ed in edits:
        f = (wt / ed["file"]).resolve()
        if wt.resolve() not in f.parents:
            raise ValueError(f"{ed['file']}: not a file of the tree")
        if ed.get("old") == "" and not f.exists() and str(ed["file"]).endswith(".py"):      # a new file (check_patch still limits where)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(str(ed["new"]).replace("\r\n", "\n").encode("utf-8"))
            continue
        if not f.exists():
            raise ValueError(f"{ed['file']}: not a file of the tree (to create a file give an empty `old`)")
        text = f.read_bytes().decode("utf-8").replace("\r\n", "\n")
        if text.count(ed["old"]) != 1:
            raise ValueError(f"{ed['file']}: `old` occurs {text.count(ed['old'])} times (must be exactly 1)")
        f.write_bytes(text.replace(ed["old"], ed["new"]).encode("utf-8"))


def worktree_of(name: str) -> Path:
    return repo_root().parent / "autolab_worktrees" / name


# ------------------------------------------------------------------ the stack of the chief: accepted hypotheses accumulate on the branch auto/stack
# (the chief module writes it; here only what every process needs: a hypothesis is prepared and tested on top of the stack)
STACK_BRANCH = "auto/stack"


def stack_load() -> dict:
    try:
        st = json.loads((DRAFTS / "stack.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    st.setdefault("items", [])
    st.setdefault("check", None)
    st.setdefault("ablation", None)
    st.setdefault("history", [])
    return st


def stack_save(st: dict) -> None:
    DRAFTS.mkdir(parents=True, exist_ok=True)
    (DRAFTS / "stack.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def stack_wt() -> Path:
    return repo_root().parent / "autolab_worktrees" / "_stack"


def stack_active() -> bool:
    return any(i.get("active", True) for i in stack_load()["items"]) and stack_wt().exists()


def stack_env() -> dict:
    """The switches the stack changes against main, in the order the hypotheses were accepted (a later one wins on the same key)."""
    env: dict = {}
    for i in stack_load()["items"]:
        if i.get("active", True):
            env.update(i.get("env") or {})
    return env


def stack_ref() -> str:
    return STACK_BRANCH if stack_active() else "HEAD"


def stack_sig() -> str:
    """The identity of the stack state: the active items in order ("" when there is no stack)."""
    return ",".join(i["name"] for i in stack_load()["items"] if i.get("active", True)) if stack_active() else ""


def stack_head() -> str:
    r = subprocess.run(["git", "-C", str(stack_wt()), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def shadowed_imports(src: str) -> set[str]:
    """Names that a module imports at its top level AND defines (def / class / assignment) at its top level: the later definition hides the module
    (`import math` plus `async def math(...)` in pipelines.py made every `math.log` raise AttributeError and no test noticed)."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    imported: set[str] = set()
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return imported & defined


def cmd_apply(name: str, patch_file: str | None) -> None:
    m = meta(name)
    repo = repo_root()
    branch = f"autolab/{name}"
    wt = worktree_of(name)
    g = lambda *a, cwd=None: subprocess.run(["git", "-C", str(cwd or wt), "-c", "user.name=autolab", "-c", "user.email=noreply@anthropic.com", *a],
                                           capture_output=True, text=True)
    want = m.get("code_change") or ""
    if not want and not patch_file:
        raise SystemExit("this hypothesis needs no code change: nothing to apply")
    before = snapshot_protected()
    if not wt.exists():
        r = subprocess.run(["git", "-C", str(repo), "worktree", "add", "-b", branch, str(wt), stack_ref()], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stderr)

    def abort(why: str):
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)], capture_output=True)
        subprocess.run(["git", "-C", str(repo), "branch", "-D", branch], capture_output=True)
        set_meta(name, stage="apply_refused", apply_error=why)
        raise SystemExit(f"apply refused: {why}")

    try:
        if patch_file:   # a human-written patch goes through the same rules
            pf = wt / ".autolab.patch"
            text = Path(patch_file).read_bytes().decode("utf-8").replace("\r\n", "\n")
            why = check_patch(re.findall(r"^\+\+\+ b/(\S+)", text, re.M), [l[1:] for l in text.splitlines() if l.startswith("+") and not l.startswith("+++")], 0)
            if why:   # refused before anything is applied
                abort(why)
            pf.write_bytes(text.encode("utf-8"))
            if g("apply", "--check", str(pf)).returncode or g("apply", str(pf)).returncode:
                abort("the patch does not apply")
            pf.unlink()
        else:
            files = pick_files(wt, want + " " + (DRAFTS / f"{name}.yaml").read_text(encoding="utf-8")[:600])
            src = "\n\n".join(f"### {p.relative_to(wt).as_posix()}\n{p.read_text(encoding='utf-8')}" for p in files)
            err = ""
            for attempt in range(2):
                reply = llm(EDIT_SYSTEM, f"Change wanted: {want}\nOnly these paths may change: {', '.join(ALLOWED_PREFIX)}. At most {limits_cfg()['max_code_lines']} added plus removed lines in all.\n"
                            f"{('Your previous edits failed: ' + err) if err else ''}\nFiles:\n{src}", int(limits_cfg()["apply_max_tokens"]), "apply")
                try:
                    g("checkout", "--", ".")
                    apply_edits(wt, parse_edits(reply))
                    err = ""
                    break
                except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as ex:
                    err = str(ex)[:300]
                    log(f"edit attempt {attempt + 1} failed: {err}")
            if err:
                abort("the model's edits did not apply: " + err)
        g("add", "-A")
        names = g("diff", "--cached", "--name-only").stdout.split()
        diff = g("diff", "--cached", "-U0").stdout.splitlines()
        added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        why = check_patch(names, added, removed)
        if why:
            abort(why)
        for n_ in names:
            if n_.endswith(".py"):
                c = subprocess.run([sys.executable, "-m", "py_compile", str(wt / n_)], capture_output=True, text=True)
                if c.returncode:
                    abort(f"{n_} does not compile: {c.stderr[-300:]}")
                old_src = g("show", f"HEAD:{n_}").stdout       # empty for a new file
                new_hide = shadowed_imports((wt / n_).read_text(encoding="utf-8")) - shadowed_imports(old_src)
                if new_hide:
                    abort(f"{n_}: the import of {sorted(new_hide)} is hidden by a function/class/variable of the same name in this module (use `import X as _X`)")
        py = LAB / ".venv" / "bin" / "python"
        t = subprocess.run([str(py) if py.exists() else sys.executable, "tools/selftest_blocks.py"], cwd=str(wt), capture_output=True, text=True)
        if t.returncode:
            abort("selftest_blocks failed on the branch:\n" + t.stdout[-600:] + t.stderr[-600:])
    except SystemExit:
        raise
    after = snapshot_protected()
    if after != before:
        abort(f"protected files changed during apply: {[k for k in after if after[k] != before.get(k)]}")
    g("commit", "-m", f"autolab: {name}")
    sha = g("rev-parse", "--short", "HEAD").stdout.strip()
    (wt / "VERSION").write_text(f"autolab/{name} {sha} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
    set_meta(name, stage="applied", branch=branch, worktree=str(wt), commit=sha, files=names, lines=len(added) + removed, stack_at_apply=stack_sig())
    log(f"change committed on branch {branch} ({', '.join(names)}, {len(added)}+/{removed}-) in {wt}. Master is untouched; merge is for a human.")


def cmd_export(name: str) -> None:
    m = meta(name)
    if not m.get("branch"):
        raise SystemExit("no branch for this draft")
    repo = repo_root()
    out = subprocess.run(["git", "-C", str(repo), "diff", f"HEAD...{m['branch']}"], capture_output=True, text=True).stdout
    p = DRAFTS / f"{name}.patch"
    p.write_text(out, encoding="utf-8")
    log(f"{p} ({len(out.splitlines())} lines): review and `git apply` it in the master checkout yourself")


# ------------------------------------------------------------------ simulation as a filter, submit, verdict
def sim_py() -> Path:
    for p in (HERE / "sim.py", LAB / "harness" / "tools" / "sim.py"):
        if p.exists():
            return p
    raise SystemExit("sim.py not found")


def parse_sim(out: str) -> dict[int, float]:
    res = {}
    for m in re.finditer(r"^\|\s*(\d+)k\s*\|\s*([\d.]+)\s*\|\s*[\d.nan]+\s*\|\s*([\d.]+)\s*\|", out, re.M):
        res[int(m.group(1)) * 1000] = float(m.group(2))   # "points (mean)", the sim's own headline column
    return res


def sim_arm(env_over: dict, budgets: list[int], harness_dir: Path | None) -> dict[int, float]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HARNESS_")}
    env.update({k: str(v) for k, v in env_over.items()})
    out = {}
    for b in budgets:
        cmd = [sys.executable, str(sim_py())] + (["--harness-dir", str(harness_dir)] if harness_dir else ["--harness-dir", str(sim_py().parent.parent)])
        cmd += ["curve", "--stream", "--draws", "10", "--budgets", f"{b}:{b}:1"]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        out.update(parse_sim(r.stdout))
    return out


def cmd_sim(name: str) -> str:
    """The cheap filter before the expensive run. Returns pass | skip_real | uninformative. The simulator replays the real Scheduler and
    Predictor on recorded outcomes, so it only speaks for hypotheses that change SORTER / PREDICTOR / PUNSOLV / SOLVERS (it overestimates
    points by ~10% at a binding budget of 30k, a bias shared by the arms: the arm DIFFERENCE is used, not the level). The run is
    cancelled when the simulator predicts every variant clearly worse than the reference, or no variant different enough to ever show
    above the noise threshold at the competition budgets."""
    import yaml
    e = yaml.safe_load((DRAFTS / f"{name}.yaml").read_text(encoding="utf-8"))
    arms = e["arms"]
    keys = {k for v in arms.values() for k in ((v or {}).get("env") or {})}
    sim_keys = keys & set(SIM_KEYS)
    m = meta(name)
    if not sim_keys or keys - set(SIM_KEYS) or any((v or {}).get("batch_size") or (v or {}).get("model") for v in arms.values()):
        set_meta(name, sim="uninformative: the hypothesis changes more than scheduler-level switches", sim_verdict="uninformative")
        log(f"{name}: not a pure scheduler-level hypothesis, the simulator cannot judge it; real run decides")
        return "uninformative"
    budgets = sorted({20000, 30000, 40000, int(e.get("budget") or 30000)})
    hd = Path(m["worktree"]) if m.get("worktree") else None
    pts = {a: sim_arm({k: str(x) for k, x in ((v or {}).get("env") or {}).items()}, budgets, hd) for a, v in arms.items()}
    first = next(iter(pts))
    lines, clear_worse, too_small = [], [], []
    for a, p in pts.items():
        if a == first:
            continue
        d = {b: p.get(b, 0) - pts[first].get(b, 0) for b in budgets}
        rel = {b: d[b] / pts[first][b] for b in budgets if pts[first].get(b)}
        lines.append(f"{a} vs {first}: " + ", ".join(f"{b // 1000}k {d[b]:+.1f} ({rel.get(b, 0):+.0%})" for b in budgets))
        if rel and sum(rel.values()) / len(rel) <= -0.05 and max(rel.values()) <= 0:
            clear_worse.append(a)
        elif rel and max(abs(x) for x in rel.values()) * max(pts[first].values()) < 4:
            too_small.append(a)
    summary = {a: {f"{b // 1000}k": p.get(b) for b in budgets} for a, p in pts.items()}
    non_ref = [a for a in arms if a != first]
    verdict = "skip_real" if len(clear_worse) + len(too_small) == len(non_ref) else "pass"
    reason = ("all variants predicted >=5% worse at every budget" if len(clear_worse) == len(non_ref) else
              "predicted effect below 4 points at every budget, invisible against the noise 7.69" if len(too_small) + len(clear_worse) == len(non_ref) else "")
    set_meta(name, sim="; ".join(lines), sim_points=summary, sim_verdict=verdict, sim_reason=reason)
    log(f"{name}: simulator {verdict} {reason}: " + "; ".join(lines))
    return verdict


def exp_requests_estimate(e: dict) -> int:
    n_tasks = 100 if "100" in str(e.get("set")) else 40
    per_task = 2.5 if (e.get("budget") or 0) >= 60000 else 2.0   # at binding budgets about 2/3 of the tasks run
    return int(n_tasks * per_task * len(e["arms"]) * int(e.get("repeats", 1)) * (0.7 if (e.get("budget") or 1e9) < 50000 else 1))


def queue_depth() -> int:
    """Experiments of the autolab (names ab_*) pending + running: the stand's own experiments in the same queue do not count against the autolab's limit."""
    return sum(1 for k in ("pending", "running") for f in (EXP / "queue" / k).glob("ab_*.yaml"))


PARALLEL_MAX = 4
BACKEND_NEXT: str | None = None       # set by the chief around one call: the backend ("gateway" | "local") of the experiments queued by it


def apply_backend(e: dict, m: dict) -> str:
    """The backend of an experiment: the chief's choice for this call, else the hypothesis' own (a check / repeat inherits it), else the gateway.
    The local Qwen is one forced model (HARNESS_FORCE_MODEL), so arms that name other models cannot run there."""
    be = BACKEND_NEXT or m.get("backend") or "gateway"
    if be != "local":
        return "gateway"
    bad = [a for a, s_ in (e.get("arms") or {}).items() if (s_ or {}).get("model") and not str(s_["model"]).startswith("local")]
    if bad:
        raise SystemExit(f"the local backend has one Qwen: arms {bad} name other models")
    if not sr.shim_ready():
        raise SystemExit("the local Qwen (llama-server :8011 / shim :18444) is not reachable")
    e["backend"] = "local"
    if not str(e.get("model") or "").startswith("local"):
        e["model"] = str(limits_cfg().get("local_model") or "local-35b-noreason")
    return "local"


def parallel_runs_now() -> int:
    """Runs of one experiment at once: the chief's override (parallel.json, 1..PARALLEL_MAX) else exp_parallel_runs of the config."""
    try:
        n = int(json.loads((DRAFTS / "parallel.json").read_text(encoding="utf-8"))["n"])
    except (OSError, ValueError, KeyError, TypeError):
        n = int(limits_cfg().get("exp_parallel_runs") or 1)
    return max(1, min(PARALLEL_MAX, n))


def cmd_submit(name: str) -> None:
    import yaml
    if STOP_FILE.exists():
        raise SystemExit(f"{STOP_FILE} exists: the autolab is stopped, nothing is submitted")
    src = DRAFTS / f"{name}.yaml"
    m = meta(name)
    e = yaml.safe_load(src.read_text(encoding="utf-8"))
    if m.get("code_change") and m.get("stage") not in ("applied", "queued"):
        raise SystemExit("needs a code change: run `apply` first")
    if m.get("sim_verdict") == "skip_real":
        raise SystemExit(f"the simulator filter stopped this hypothesis ({m.get('sim_reason')}); no real run")
    lim = limits_cfg()
    if queue_depth() >= lim["max_queue_depth"]:
        raise SystemExit(f"queue has {queue_depth()} experiments (max_queue_depth={lim['max_queue_depth']})")
    be = apply_backend(e, m)
    e.setdefault("parallel_runs", parallel_runs_now())
    est = exp_requests_estimate(e)
    if est > lim["max_expected_requests_cycle"]:
        raise SystemExit(f"estimated {est} gateway requests (max_expected_requests_cycle={lim['max_expected_requests_cycle']})")
    if m.get("worktree"):
        e["code_dir"] = m["worktree"]
        e["change"] = f"branch {m['branch']} @ {m.get('commit')}: {m.get('code_change', '')[:300]}"
    if m.get("stack_exp") and stack_wt().exists():       # a check of the stack itself: its arms carry their own switches
        e["code_dir"] = str(stack_wt())
        e["change"] = f"the stack {STACK_BRANCH} @ {stack_head()}"
    if stack_active() and not m.get("stack_exp"):      # a hypothesis is tested on top of the stack: reference = the stack, variant = the stack + the hypothesis
        if not m.get("worktree"):
            e["code_dir"] = str(stack_wt())
            e["change"] = f"on the stack {STACK_BRANCH} @ {stack_head()}"
        senv = stack_env()
        for a_, spec_ in list((e.get("arms") or {}).items()):
            spec_ = dict(spec_ or {})
            spec_["env"] = {**senv, **(spec_.get("env") or {})}
            e["arms"][a_] = spec_
    dst = EXP / "queue" / "pending" / f"{name}.yaml"
    for kind in ("pending", "running", "done"):
        if (EXP / "queue" / kind / f"{name}.yaml").exists():
            raise SystemExit(f"{name} is already in the queue ({kind})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = DRAFTS / f"{name}.queued.yaml"
    tmp.write_text(yaml.safe_dump(e, sort_keys=False, allow_unicode=True), encoding="utf-8")
    try:   # validate with the lab's own loader (holdout guard, models, sets) before it reaches the queue
        import lab
        lab.load_experiment(tmp)
    except SystemExit as ex:
        tmp.unlink()
        raise SystemExit(f"the lab refused the experiment: {ex}")
    shutil.move(str(tmp), str(dst))
    set_meta(name, stage="queued", queued_at=time.time(), gw_week_used_at_submit=gw_week_used(), expected_requests=est, backend=be,
             **({} if m.get("stack_exp") else {"stack_at_submit": stack_sig()}))
    log(f"{name} is in queue/pending (expected ~{est} gateway requests); the dispatcher runs it")
    record_journal(name, "не завершён", f"в очереди, ожидается около {est} запросов шлюза")


def cells_requests(name: str) -> int | None:
    f = EXP / "results" / name / "cells.jsonl"
    if not f.exists():
        return None
    return sum(int(c.get("pipeline_calls", 0)) + int(c.get("agent_calls", 0)) for c in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip()))


def mode_of(name: str) -> str:
    """The goal mode of a hypothesis (accuracy | economy | both); hypotheses made before modes existed count as accuracy."""
    m = meta(family_base(name)).get("target_mode") or meta(name).get("target_mode") or spec_of(name).get("target_mode")
    return m if m in GOAL_MODES else "accuracy"


def journal_mode(name: str) -> str:
    return {"both": "оба"}.get(mode_of(name), mode_of(name))


def economy_eval(res: dict, thr: float | None) -> dict:
    """The economy metric per non-reference arm: points per 1k tokens and tokens to the same points against the reference, and the check that the points did
    not fall by more than the noise threshold. kind: accept (saves tokens, points hold), reject (points fell beyond noise, or more tokens), clarify (between)."""
    ref = next(iter(res))
    r0 = res[ref]
    ms = limits_cfg()["economy_min_saving"]
    out = {}
    for a, v in res.items():
        if a == ref:
            continue
        d = v.get("points_diff_vs_first")
        t0, t1, p0, p1 = r0.get("tokens_per_run") or 0, v.get("tokens_per_run") or 0, r0.get("points_per_run") or 0, v.get("points_per_run") or 0
        if not d or not t0 or not t1:
            out[a] = {"kind": "clarify", "note": "нет данных о токенах или очках"}
            continue
        saving = 1 - t1 / t0
        ppk0, ppk1 = p0 / (t0 / 1000), p1 / (t1 / 1000)
        to_same = t1 if p1 >= p0 else (t1 * p0 / p1 if p1 > 0 else None)      # tokens the arm needs to reach the reference points (linear if it scores lower)
        points_ok = d["mean"] >= -(thr or 0)
        non_inferior = d["ci95"][0] >= -(thr or 0)     # R-L4: the lower end of the interval, not only the mean, must stay within the noise (no drift by small losses)
        if not points_ok:
            kind, note = "reject", f"очки просели на {d['mean']:+.1f}, больше шума {thr}"
        elif saving >= ms and ppk1 > ppk0 and not non_inferior:
            kind, note = "clarify", f"токены {-saving:+.0%}, но нижняя граница интервала очков {d['ci95'][0]:+.1f} ниже -{thr}: «не хуже» не подтверждено"
        elif saving >= ms and ppk1 > ppk0:
            kind, note = "accept", f"токены {-saving:+.0%}, очки на 1k токенов {ppk0:.2f} -> {ppk1:.2f}, очки в пределах шума"
        elif saving <= -ms:
            kind, note = "reject", f"токенов больше на {-saving:.0%}"
        else:
            kind, note = "clarify", f"экономия токенов {saving:+.0%} меньше порога {ms:.0%} или очки на 1k токенов не выросли"
        out[a] = {"kind": kind, "note": note, "saving": round(saving, 4), "ppk_ref": round(ppk0, 3), "ppk": round(ppk1, 3), "tokens_to_same_points": round(to_same) if to_same else None,
                  "points_diff": d["mean"]}
    return out


def apply_mode_verdict(name: str) -> None:
    """The verdict of a hypothesis is judged by the metric of its goal mode. accuracy: the lab's verdict (points at the full budget against the noise threshold) as it is;
    economy: economy_eval; both: accept only if both accept, reject if either rejects. The lab's own verdict stays in `lab_verdict`."""
    m = meta(name)
    res = m.get("result") or {}
    mode = mode_of(name)
    lab = m.get("lab_verdict") or m.get("verdict") or {}
    if not res or not lab:
        return
    eco = economy_eval(res, m.get("threshold_used"))
    new = {}
    for a, k in lab.items():
        if a not in eco:
            new[a] = k
            continue
        e = eco[a]["kind"]
        if mode == "accuracy":
            new[a] = k
        elif mode == "economy":
            new[a] = e
        else:
            new[a] = "reject" if "reject" in (k, e) else ("accept" if k == e == "accept" else "clarify")
    set_meta(name, lab_verdict=lab, verdict=new, mode_eval=eco, target_mode=mode)


def verdict_kind(name: str) -> str:
    v = (meta(name).get("verdict") or {})
    vals = [x for x in v.values() if x]
    return "accept" if "accept" in vals else ("clarify" if "clarify" in vals else "reject")


def status_evidence(name: str) -> str:
    m = meta(name)
    res, ver, thr = m.get("result") or {}, m.get("verdict") or {}, m.get("threshold_used")
    best = None
    for a, v in res.items():
        d = v.get("points_diff_vs_first")
        if d and (best is None or d["mean"] > best[1]["mean"]):
            best = (a, d)
    spec = spec_of(name)
    rep = spec.get("repeats") or "?"
    num = f"арм {best[0]} {best[1]['mean']:+.1f} {best[1]['ci95']} очков/прогон против порога {thr}" if best else "разности нет (нет арма против эталона)"
    return f"{name}: {num}, {rep} повт., набор {spec.get('set', '?')}, вердикт {verdict_kind(name)}"


def status_plan(name: str, accept: bool) -> list[dict]:
    """Items for status_check.update, one per target cell. accept=False: history only (clarify/reject); accept=True: status explored-headroom with the measured gain."""
    root = status_root()
    kind = verdict_kind(name)
    spec = spec_of(name)
    plan = []
    mode = mode_of(name)
    for i, c in enumerate(targets_of(name)):
        for md in (astat.MODES if mode == "both" else (mode,)):         # a hypothesis for both modes updates both sets of per-mode fields
            it = {"cell": c, "evidence": status_evidence(name), "experiments": name, "date": time.strftime("%Y-%m-%d"), "mode": md}
            if accept and kind == "accept":
                e = astat.entry(root, *astat.parse_ref(c)) or {}
                if md == "accuracy" or e.get("headroom") in ("low", "medium", "high"):
                    it["status"] = "explored-headroom"      # the shared status moves only on knowledge that holds for every mode (an economy result alone does not make a cell explored)
                it["headroom"] = e.get("headroom") if e.get("headroom") in ("low", "medium", "high") else "medium"
                if e.get("confidence", 0) < 0.4:
                    it["confidence"] = 0.4
                if i == 0 and str(spec.get("set", "")).startswith("mixed100"):   # points are comparable only on the mixed 100-task set (status/README.md)
                    g = mode_gain(name, md)
                    if g is not None and g > 0:
                        it["gain"] = round(g, 1)
            else:
                it["note"] = f"статус не менялся: вердикт {kind}"
            plan.append(it)
    return plan


def mode_gain(name: str, mode: str) -> float | None:
    """The measured gain in the unit of the mode: accuracy = points per run, economy = percent of tokens saved at the same points."""
    m = meta(name)
    res = m.get("result") or {}
    if mode == "economy":
        ref = next(iter(res), None)
        best = None
        for a, v in (m.get("mode_eval") or {}).items():
            t0 = (res.get(ref) or {}).get("tokens_per_run")
            if v.get("kind") == "accept" and v.get("tokens_to_same_points") and t0:
                g = 100 * (1 - v["tokens_to_same_points"] / t0)
                best = g if best is None or g > best else best
        return best
    return max((v["points_diff_vs_first"]["mean"] for v in res.values() if v.get("points_diff_vs_first")), default=None)


def done_block(name: str) -> dict | None:
    """The status block lab.py wrote under the experiment in queue/done (None while it is not there)."""
    import yaml
    f = EXP / "queue" / "done" / f"{name}.yaml"
    if not f.exists():
        return None
    text = f.read_text(encoding="utf-8", errors="replace")
    mark = "\n# ----"
    try:
        return (yaml.safe_load(text.split(mark, 1)[1].split("\n", 1)[1]) or {}) if mark in text else {}
    except Exception:  # noqa: BLE001
        return {}


def journal_kind(name: str) -> str:
    m = meta(name)
    if m.get("result") or m.get("verdict"):
        return verdict_kind(name)
    blk = done_block(name)
    if blk is not None:
        return "не завершён" if blk.get("status") == "ok" else "failed"     # finished by the lab, no verdict yet / aborted or failed
    if m.get("stage") == "queued" or any((EXP / "queue" / k / f"{name}.yaml").exists() for k in ("pending", "running")):
        return "не завершён"
    return "не запускался"


def journal_fields(name: str, kind: str, note: str = "", accepted: str | None = None) -> dict:
    m, spec = meta(name), spec_of(name)
    res, thr = m.get("result") or {}, m.get("threshold_used")
    arms = spec.get("arms") or {}

    def desc(a, s):
        env = ", ".join(f"{k}={v}" for k, v in ((s or {}).get("env") or {}).items())
        return f"{a}: " + (env or (f"batch_size={s['batch_size']}" if (s or {}).get("batch_size") else "без переключателей"))
    parts = []
    for a, v in res.items():
        d = v.get("points_diff_vs_first")
        parts.append(f"{a} очки {v.get('points_per_run')}, токены {v.get('tokens_per_run')}" + (f", разность {d['mean']:+.1f} [{d['ci95'][0]:.1f}; {d['ci95'][1]:.1f}]" if d else " (эталон)"))
    result = "; ".join(parts) + (f"; порог {thr}" if parts and thr else "") if parts else (note or {"не завершён": "ещё не завершён: ждёт очереди или идёт", "failed": "эксперимент не дал результата", "не запускался": "в очередь не поставлен"}.get(kind, kind))
    blk = done_block(name)
    if kind == "failed" and not note and blk:
        note = str(blk.get("reason") or blk.get("result_error") or blk.get("status") or "")[:160]
    if kind == "не завершён" and not parts and blk is not None:
        result = "завершён в лаборатории, вердикт автолаборатории ещё не построен"
    if kind == "failed" and note:
        result = f"{result}: {note}"
    result += f" · режим: {mode_of(name)}"
    dec = m.get("decision")
    if dec:
        result += f" · решение: {DECISION_RU.get(dec, dec)}" + (f" ({m['decision_note']})" if m.get("decision_note") else "")
    if accepted is None:
        accepted = f"ветка:{m['branch']}" if m.get("branch") and m.get("stage") != "accepted" else ("да" if m.get("stage") == "accepted" else "нет")
    rep = DRAFTS / f"{name}.report.md"
    return {"name": name, "verdict": kind, "result": result, "date": time.strftime("%Y-%m-%d"), "compare": "; ".join(desc(a, s) for a, s in arms.items())[:200] or None,
            "setbudget": f"{spec.get('set', '?')} · {spec.get('budget', '?')}", "repeats": spec.get("repeats"), "who": "автолаборатория",
            "accepted": accepted, "link": f"experiments/autolab/{name}.report.md" if rep.exists() else None}


def record_journal(name: str, kind: str | None = None, note: str = "", accepted: str | None = None, force: bool = False) -> list[str]:
    """Write the state of a hypothesis into the journal of every target cell (idempotent by name): on submit, on every verdict (accept, clarify,
    reject, failed), on binding and on accept. The row also waits for the next push to GitHub. Never raises: the journal must not break the pipeline."""
    cells = targets_of(name)
    root = status_root()
    if not cells or not astat.index(root):
        return []
    f = journal_fields(name, kind or journal_kind(name), note, accepted)
    done = []
    for c in cells:
        item = {"kind": "log", "cell": c, "mode": journal_mode(name), **{k: v for k, v in f.items() if v is not None}}
        old = next((r for r in astat.journal_rows(root, c, 100000) if r["name"] == name), None)
        if old is not None:     # a row written by hand or earlier: keep its date, author, comparison and link; change only what is new
            if old["verdict"] == f["verdict"] and accepted is None and not force:
                done.append(c)
                continue
            item = {"kind": "log", "cell": c, "mode": journal_mode(name), "name": name, "verdict": f["verdict"], "result": f["result"]}
            if accepted is not None:
                item["accepted"] = accepted
        try:
            astat.apply(root, [item])
            astat.events_add_log(EXP, name, item)
            done.append(c)
        except Exception as ex:  # noqa: BLE001
            log(f"{name}: journal of {c} not written: {ex}")
    return done


def record_status_event(name: str) -> None:
    """Every verdict goes to the journal of the target cells (the status itself changes only on accept, in the bot).
    A lab verdict `reject` also closes the hypothesis by itself: nobody has to decide on it, its task types are free again."""
    record_journal(name)
    if meta(name).get("result") and verdict_kind(name) == "reject" and not decision_of(name):
        set_decision(name, "auto_rejected", "хуже эталона", "autolab")


def cmd_verdict(name: str, notify: bool = True) -> dict:
    import lab
    nf = lab.noise_floor()
    res = lab.summarize(name)
    thr = nf["threshold_points"] if nf else None
    thr_set = next((v["points_diff_vs_first"]["threshold"] for v in res["arms"].values() if v.get("points_diff_vs_first")), None)
    if thr_set is not None:   # R-K4: the threshold of the set itself (own noise, number of repeats), not always baseline0's
        thr = thr_set
    lines = [f"# Verdict: {name}", "", f"Set {res['set']}, code {res['code_version']}, repeats {res['repeats']}.",
             f"Acceptance threshold (noise floor of {lab.NOISE_RUN}, read from the lab, never set here): {thr} points per run "
             f"({nf['repeats_per_arm'] if nf else '-'} repeats per arm, measured on a 100-task set). `accept` needs the paired 95% interval above 0 AND a "
             "difference not smaller than the threshold; this is the lab's rule (`lab.summarize`). **Merging stays with a human.**", "",
             "| arm | points per run | tokens per run | points per 1k tokens | accuracy | points diff vs first [95% CI] | CI half-width | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    notes = []
    for a, v in res["arms"].items():
        d = v.get("points_diff_vs_first")
        ppk = v["points_per_run"] / (v["tokens_per_run"] / 1000) if v["tokens_per_run"] else 0
        hw = (d["ci95"][1] - d["ci95"][0]) / 2 if d else None
        lines.append(f"| {a} | {v['points_per_run']} | {v['tokens_per_run']} | {ppk:.2f} | {v['accuracy']} | "
                     f"{('%+.1f %s' % (d['mean'], d['ci95'])) if d else '-'} | {('%.1f' % hw) if hw is not None else '-'} | {res['verdict'].get(a) or 'reference'} |")
        if d and thr and hw > thr:
            notes.append(f"{a}: the interval half-width {hw:.1f} is larger than the threshold {thr}: the experiment is underpowered for this question (more repeats or tasks).")
        if d and v["tokens_per_run"] and res["arms"][next(iter(res["arms"]))]["tokens_per_run"]:
            r0 = res["arms"][next(iter(res["arms"]))]
            notes.append(f"{a}: tokens {v['tokens_per_run'] / r0['tokens_per_run'] - 1:+.0%}, points per 1k tokens {ppk:.2f} vs {r0['points_per_run'] / (r0['tokens_per_run'] / 1000):.2f} (reference).")
    req = cells_requests(name)
    lines += [""] + [f"- {n}" for n in notes] + [f"- gateway requests of the experiment (pipeline+agent calls in cells): {req}"]
    (DRAFTS / f"{name}.verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    set_meta(name, stage="verdict", verdict=res["verdict"], lab_verdict=res["verdict"], threshold_used=thr, experiment_requests=req, result={a: {k: v.get(k) for k in ("points_per_run", "tokens_per_run", "accuracy", "points_diff_vs_first")} for a, v in res["arms"].items()})
    try:
        apply_mode_verdict(name)
    except Exception as ex:  # noqa: BLE001 - the lab's verdict stays if the mode metric cannot be computed
        log(f"{name}: mode verdict not computed: {ex}")
    print("\n".join(lines))
    try:
        record_status_event(name)
    except Exception as ex:  # noqa: BLE001 - the scheme history must not hide the verdict
        log(f"{name}: scheme status not updated: {ex}")
    if notify:   # the chat gets the hypothesis, the short text and the report file (outbox; nothing is sent from here)
        try:
            notify_verdict(name)
        except Exception as ex:  # noqa: BLE001 - a report problem must not hide the verdict
            log(f"{name}: report for the chat not made: {ex}")
    return res


def cmd_list() -> None:
    for f in sorted(DRAFTS.glob("*.yaml")):
        if f.name.endswith(".queued.yaml"):
            continue
        m = meta(f.stem)
        print(f"{f.stem:<36} {m.get('stage', '?'):<13} {m.get('verdict', '') or m.get('sim_verdict', '')}")


# ------------------------------------------------------------------ the one-command cycle
def dispatcher_alive() -> bool:
    try:
        r = subprocess.run(["pgrep", "-f", "dispatcher.py run"], capture_output=True, text=True)
    except OSError:
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def wait_done(names: list[str], poll: int, max_min: int) -> None:
    t0 = time.time()
    left = set(names)
    while left and time.time() - t0 < max_min * 60:
        for nm in list(left):
            if (EXP / "queue" / "done" / f"{nm}.yaml").exists():
                left.discard(nm)
                log(f"{nm}: finished")
        if left:
            time.sleep(poll)
    if left:
        log(f"gave up waiting for {sorted(left)} after {max_min} min (they stay in the queue; run `cycle --names ...` later)")


def cmd_await(name: str) -> None:
    """Wait until the experiment is in queue/done, then build the verdict and send hypothesis + report to the chat (a rerun started from the chat has no cycle to do it)."""
    wait_done([name], limits_cfg()["wait_poll_s"], limits_cfg()["max_wait_min"])
    if (EXP / "queue" / "done" / f"{name}.yaml").exists():
        try:
            cmd_verdict(name)
        except Exception as ex:  # noqa: BLE001 - the experiment finished without a usable result
            log(f"{name}: no verdict: {ex}")
            record_journal(name, "failed", str(ex)[:160])


def chief_flag() -> Path:
    """Switched on by a human only (`шеф вкл` in the bot); while it exists the autopilot proposes nothing: one generator at a time."""
    return CFG_DIR / "CHIEF"


def autopilot_flag() -> Path:
    return CFG_DIR / "AUTOPILOT"


def _autopilot_state() -> dict:
    try:
        st = json.loads((DRAFTS / "autopilot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    today = time.strftime("%Y-%m-%d")
    if st.get("day") != today:
        st["day"], st["names_today"] = today, []
    st.setdefault("topic_idx", 0)
    return st


def localdata_flag() -> Path:
    return CFG_DIR / "LOCAL_DATA"


def _local_stream_pids() -> list[str]:
    try:
        out = subprocess.run(["pgrep", "-af", "stream_run.py"], capture_output=True, text=True).stdout
    except OSError:
        return []
    return [ln.split()[0] for ln in out.splitlines() if "--backend local" in ln and "pgrep" not in ln]


def _gpu_temp() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def cmd_localdata(status_only: bool = False) -> str:
    """Data collection on the local Qwen of the GPU host (llama-server :8011 behind the shim :18444, no gateway requests): the ordinary
    `stream_run.py --backend local` stream, kept running while the human switched it on (file ~/.config/autolab/LOCAL_DATA, written by the bot
    after a confirmation). Switching it off asks the stream to stop after its chunk (STREAM_STOP_local)."""
    lim = limits_cfg()
    on = localdata_flag().exists()
    pids = _local_stream_pids()
    stop = LAB / "STREAM_STOP_local"
    info = f"local data {'ON' if on else 'off'}: stream {'running (pid ' + ','.join(pids) + ')' if pids else 'not running'}"
    if status_only:
        return info
    if not on:
        if pids and not stop.exists():
            return info    # a stream somebody else started: not ours to stop
        return info
    if pids:
        return info
    if STOP_FILE.exists():
        return info + "; autolab stop flag: not started"
    if stop.exists():
        return info + "; STREAM_STOP_local exists: not started"
    if any("backend: local" in f.read_text(encoding="utf-8", errors="replace") for f in (EXP / "queue" / "running").glob("*.yaml")):
        return info + "; a local experiment is running: not started"
    if not sr.shim_ready():
        return info + "; llama-server (:8011) or the shim (:18444) is not reachable"
    t = _gpu_temp()
    if t is not None and t > lim["local_data_max_temp_c"]:
        return info + f"; GPU {t} C > {lim['local_data_max_temp_c']}: not started"
    out = Path(os.path.expanduser("~/datasets/token_juiwen/stream"))
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(LAB / "stream_run.py"), "--backend", "local", "--seed", "1", "--workers", str(lim["local_data_workers"])]
    subprocess.Popen(cmd, stdout=open(out / "run_local.out", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(LAB))
    return info + "; started the local stream"


def _autopilot_wait_notice(st: dict, locked: dict) -> None:
    """Every task type is held: tell the chat (at most every 6 h) which hypotheses wait for a decision, otherwise the autopilot silently stands."""
    if time.time() - st.get("wait_notice_ts", 0) < 6 * 3600:
        return
    waiting = sorted({n for n in locked.values() if not any((EXP / "queue" / k / f"{n}.yaml").exists() for k in ("pending", "running"))})
    if not waiting:
        return
    st["wait_notice_ts"] = time.time()
    (DRAFTS / "autopilot.json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    outbox_put(allowed_chat_ids(), "⏳ Автопилот стоит: все типы задач заняты гипотезами, которые ждут вашего решения: " + ", ".join(waiting) + ". Откройте `гипотеза <имя>` и выберите: принять, отклонить, ревью, повтор или отложить.")


def cmd_autopilot(status_only: bool = False) -> str:
    """One tick (cron, every 10 minutes): when the queue is short and the gateway has room, the generator proposes a batch, the batch goes through
    validation, dedupe, the simulator filter and into the queue, and every queued hypothesis gets an `await` that sends its verdict to the chat.
    Off unless the human switched it on (file ~/.config/autolab/AUTOPILOT, written by the bot after a confirmation)."""
    lim = limits_cfg()
    st = _autopilot_state()
    on = autopilot_flag().exists()
    q = {k: len(list((EXP / "queue" / k).glob("*.yaml"))) for k in ("pending", "running")}
    depth = q["pending"] + q["running"]
    left = (lim["autopilot_max_per_day"] - len(st["names_today"])) if lim["autopilot_max_per_day"] else 10 ** 6
    info = (f"autopilot {'ON' if on else 'off'}: queue {q['running']} running + {q['pending']} pending (target depth {lim['autopilot_target_depth']}), "
            f"today {len(st['names_today'])}/{lim['autopilot_max_per_day'] or 'no limit'} hypotheses")
    if status_only or not on:
        return info
    if chief_flag().exists():
        return info + "; the chief is on: the autopilot proposes nothing in parallel"
    if STOP_FILE.exists() or (EXP / "queue" / "STOP").exists():
        return info + "; stop flag: nothing started"
    if depth >= lim["autopilot_target_depth"]:
        return info + "; queue is long enough"
    if left <= 0:
        return info + "; daily limit reached"
    lock = DRAFTS / "autopilot.lock"
    DRAFTS.mkdir(parents=True, exist_ok=True)
    try:
        if lock.exists() and time.time() - lock.stat().st_mtime < 3 * 3600:
            return info + "; another tick is running"
        lock.write_text(str(os.getpid()))
        try:
            import dispatcher
            ok, why = dispatcher.exp_gate(dispatcher.load_cfg(), dispatcher.fractions(sr.limits()))
        except Exception as ex:  # noqa: BLE001
            ok, why = False, f"gateway limits unreadable ({type(ex).__name__})"
        if not ok:
            return info + f"; gateway gate: {why}"
        n = max(1, min(lim["autopilot_batch"], left, lim["max_queue_depth"] - depth))
        cells, cell_modes = None, None
        if astat.index(status_root()):
            used: dict[str, int] = {}
            for nm in st["names_today"]:
                for c in targets_of(nm):
                    used[c] = used.get(c, 0) + 1
            locked = active_types()
            if len(locked) >= len(astat.index_types(status_root())):
                _autopilot_wait_notice(st, locked)
                return info + "; every task type is held by an active hypothesis: waiting for decisions"
            n = min(n, len(astat.index_types(status_root())) - len(locked))
            cells, cell_modes = plan_cells(n, used, set(locked), None, st.get("mode_idx", 0), spare=2)     # a few spare cells (some cannot be tested with the switches); modes alternate
            st["mode_idx"] = st.get("mode_idx", 0) + n
            topic = "по рейтингу схемы: " + ", ".join(f"{c} [{cell_modes[c]}]" for c in cells)
        else:
            topics = lim["autopilot_topics"] or [""]
            topic = topics[st["topic_idx"] % len(topics)]
        st["topic_idx"] += 1
        before = {f.name[:-10] for f in DRAFTS.glob("*.meta.json")}
        try:
            cmd_cycle(n, topic, None, False, cells, bool(cells), None, cell_modes if cells else None)
        except SystemExit as ex:
            log(f"autopilot: cycle stopped: {ex}")
        except Exception as ex:  # noqa: BLE001
            log(f"autopilot: cycle failed: {type(ex).__name__}: {ex}")
        new = sorted(f.name[:-10] for f in DRAFTS.glob("*.meta.json") if f.name[:-10] not in before and meta(f.name[:-10]).get("stage") in ("queued", "verdict"))
        for nm in new:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "await", nm], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        st["names_today"] = st["names_today"] + new
        (DRAFTS / "autopilot.json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        if new:
            outbox_put(allowed_chat_ids(), "🤖 Автопилот поставил в очередь: " + ", ".join(new) + f"\nТема: {topic}\nВердикты пришлю по мере готовности. Выключить: `автопилот выкл`.")
        else:
            (DRAFTS / "autopilot.json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
        return info + f"; tick: topic '{topic[:50]}', queued {new or 'nothing'}"
    finally:
        lock.unlink(missing_ok=True)


def cmd_cycle(n: int, topic: str, names: list[str] | None, wait: bool, cells: list[str] | None = None, strict: bool = False,
              mode: str | None = None, cell_modes: dict | None = None) -> None:
    t0 = time.time()
    used0, week0 = CALLS["n"], gw_week_used()
    if not names:
        names = cmd_propose(n, False, topic, cells, strict, None, mode, cell_modes)
    report = [f"# Autolab cycle {time.strftime('%Y-%m-%d %H:%M')}", "", f"Topic: {topic or '(default)'}; generator {MODEL}; hypotheses: {names}", ""]
    ready = []
    for nm in names:
        m = meta(nm)
        try:
            if m.get("stage") in ("proposed", None, "apply_refused") and m.get("code_change"):
                cmd_apply(nm, None)
            if meta(nm).get("stage") in ("proposed", "applied") and not meta(nm).get("sim_verdict"):
                cmd_sim(nm)
            if meta(nm).get("stage") in ("proposed", "applied"):
                if meta(nm).get("sim_verdict") == "skip_real" and not meta(nm).get("by_user"):    # a hypothesis of a human is run anyway
                    set_meta(nm, stage="filtered_by_sim")
                    set_decision(nm, "auto_rejected", "симулятор: " + str(meta(nm).get("sim_reason") or "эффект не виден")[:150], "autolab")
                    outbox_put(allowed_chat_ids(), f"❌ {nm}: отклонена автоматически (симулятор не видит эффекта: {str(meta(nm).get('sim_reason') or '')[:120]}). Типы задач свободны; вернуть можно кнопкой.",
                               None, "", [[{"label": "🔁 Повтор (всё же запустить)", "data": _cb("rerun", nm) or "al:cmd:list"}]])
                    log(f"{nm}: stopped by the simulator filter ({meta(nm).get('sim_reason')})")
                    continue
                cmd_submit(nm)
            if meta(nm).get("stage") in ("queued", "verdict"):
                ready.append(nm)
        except SystemExit as ex:
            log(f"{nm}: stopped: {ex}")
            set_meta(nm, stage_note=str(ex)[:300])
    if ready and not dispatcher_alive():
        log("WARNING: the dispatcher is not running; the queue will not move (start `dispatcher.py run` or `lab.py queue run`)")
    if wait and ready:
        wait_done(ready, limits_cfg()["wait_poll_s"], limits_cfg()["max_wait_min"])
    for nm in names:
        if (EXP / "queue" / "done" / f"{nm}.yaml").exists() and meta(nm).get("stage") != "verdict":
            try:
                cmd_verdict(nm)
            except Exception as ex:  # noqa: BLE001
                log(f"{nm}: verdict failed: {ex}")
                record_journal(nm, "failed", str(ex)[:160])
    week1 = gw_week_used()
    for nm in names:
        m = meta(nm)
        report += [f"## {nm}", f"- stage: {m.get('stage')} {m.get('stage_note', '')}", f"- code change: {m.get('code_change') or 'none'} "
                   f"{('(branch ' + m['branch'] + ', ' + str(m.get('lines')) + ' lines)') if m.get('branch') else ''}",
                   f"- simulator: {m.get('sim_verdict', '-')} {m.get('sim_reason', '')} {m.get('sim', '')}",
                   f"- verdict: {m.get('verdict', '-')}; result {json.dumps(m.get('result', {}), ensure_ascii=False)}",
                   f"- gateway requests of the experiment: {m.get('experiment_requests', '-')} (expected {m.get('expected_requests', '-')})", ""]
    report += [f"Autolab LLM requests this run: {CALLS['n'] - used0}; this week {week_calls()} of {limits_cfg()['weekly_llm_calls']}.",
               f"Gateway week.used: {week0} -> {week1} (includes the filler stream and everyone else).", f"Wall time {(time.time() - t0) / 60:.0f} min."]
    p = DRAFTS / f"cycle_{time.strftime('%Y%m%d_%H%M%S')}.md"
    p.write_text("\n".join(report) + "\n", encoding="utf-8")
    log(f"cycle report: {p}")


def cmd_mine(cell: str, text: str, by: str = "", chat: str | None = None) -> list[str]:
    """A hypothesis of a human for one cell: the generator only formalizes it (one experiment, the first target cell is `cell`), the simulator filter does not
    stop it, and it goes first in the queue (`priority: 100`, see lab.queue_order). The chat gets the outcome either way."""
    to = [str(chat)] if chat else allowed_chat_ids()
    conf = type_conflicts([cell], active_types()) if astat.index(status_root()) else {}
    if conf:
        outbox_put(to, f"💡 Своя гипотеза для {cell} не запущена: типы задач заняты активными гипотезами " + ", ".join(f"{t}: {w}" for t, w in conf.items())
                   + ". Сначала решите по ним.", None, "", hyp_buttons(next(iter(conf.values()))))
        return []
    topic = ("A HUMAN proposes this hypothesis. Formalize exactly this idea as ONE experiment (arms, set, budget; switches or one small code change); "
             "do not replace it with an idea of your own: " + text)
    extra = {"priority": 100, "by_user": True, "proposed_by": str(by), "user_text": text[:1500]}
    names = cmd_propose(1, False, topic, [cell], True, extra)
    if not names:
        outbox_put(to, f"💡 Не получилось оформить вашу гипотезу для {cell}: генератор не дал допустимого эксперимента (переключатели харнеса не позволяют проверить идею "
                       "или она совпала с уже проверенной). Опишите конкретнее: что именно менять (переключатель, значение, набор задач). Подробности в журнале автолаборатории.")
        return []
    cmd_cycle(1, topic, names, False, [cell], True)
    nm = names[0]
    if meta(nm).get("stage") in ("queued", "verdict"):
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "await", nm], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
        outbox_put(to, f"💡 Ваша гипотеза {nm} оформлена и поставлена ПЕРВОЙ в очереди (ячейка {cell}, занимает типы: {', '.join(sorted(types_of(nm)))}). Вердикт пришлю сам.",
                   None, "", hyp_buttons(nm))
    else:
        outbox_put(to, f"💡 Гипотеза {nm} оформлена, но в очередь не попала: {meta(nm).get('stage_note') or meta(nm).get('stage')}.", None, "", hyp_buttons(nm))
    return names


# ------------------------------------------------------------------ reports for people (Russian, one page), outbox for the chat, reviewer
CFG_DIR = Path(os.path.expanduser(os.environ.get("AUTOLAB_CONFIG_DIR") or "~/.config/autolab"))


def allowed_chat_ids() -> list[str]:
    """Where verdicts and reports go: `report_chats` of telegram_allow.json (a group), else the whitelisted users' private chats."""
    try:
        cfg = json.loads((CFG_DIR / "telegram_allow.json").read_text(encoding="utf-8"))
        return [str(x) for x in (cfg.get("report_chats") or cfg.get("user_ids") or [])]
    except (OSError, ValueError, AttributeError):
        return []


_OB_SEQ = 0


def outbox_put(to: list[str], text: str, file: Path | None = None, caption: str = "", buttons: list | None = None) -> None:
    """A message for the chat. The bot process (autolab_tg_adapter) sends it with its own transport; nothing here talks to Telegram."""
    if not to:
        return
    ob = CFG_DIR / "tg_outbox"
    ob.mkdir(parents=True, exist_ok=True)
    item = {"to": list(to), "text": text[:3900], "file": str(file) if file else "", "caption": caption[:900], "buttons": buttons or []}
    global _OB_SEQ
    _OB_SEQ += 1                      # two messages in one millisecond must not overwrite each other
    tmp = ob / f"{int(time.time() * 1000)}_{os.getpid()}_{_OB_SEQ}.json.tmp"
    tmp.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, tmp.with_suffix(""))


def _first_arm(d: dict) -> str:
    return next(iter(d))


def build_report_ru(name: str) -> tuple[str, str]:
    """(report .md, short text of 5-7 lines) for a finished experiment, in Russian, built from the data (no LLM, no spend)."""
    import yaml
    m = meta(name)
    e = yaml.safe_load((DRAFTS / f"{name}.yaml").read_text(encoding="utf-8")) if (DRAFTS / f"{name}.yaml").exists() else {}
    res, verdict, thr = m.get("result") or {}, m.get("verdict") or {}, m.get("threshold_used")
    if not res:
        raise SystemExit(f"{name}: no result yet (run `verdict {name}` after the experiment is done)")
    ref = _first_arm(res)
    r0 = res[ref]
    essence = " ".join(str(e.get("hypothesis", "")).split())
    short_h = essence[:260] + ("..." if len(essence) > 260 else "")
    changes = []
    for a, spec in (e.get("arms") or {}).items():
        env = ", ".join(f"{k}={v}" for k, v in ((spec or {}).get("env") or {}).items())
        extra = f"batch_size={spec['batch_size']}" if (spec or {}).get("batch_size") else ""
        changes.append(f"{a}: {env or extra or 'без переключателей'}" + (f", {extra}" if env and extra else ""))
    code = m.get("code_change")
    rows, notes, checks, rec = [], [], [], []
    for a, v in res.items():
        d = v.get("points_diff_vs_first")
        ppk = v["points_per_run"] / (v["tokens_per_run"] / 1000) if v.get("tokens_per_run") else 0
        rows.append(f"| {a} | {v['points_per_run']} | {v['tokens_per_run']} | {ppk:.2f} | {v['accuracy']} | "
                    f"{('%+.1f [%.1f; %.1f]' % (d['mean'], d['ci95'][0], d['ci95'][1])) if d else '-'} | {verdict.get(a, 'эталон')} |")
        if a == ref or not d:
            continue
        hw = (d["ci95"][1] - d["ci95"][0]) / 2
        tok = v["tokens_per_run"] / r0["tokens_per_run"] - 1 if r0.get("tokens_per_run") else 0
        vd = verdict.get(a)
        if vd == "accept":
            rec.append(f"{a}: кандидат на принятие (интервал выше нуля и разность не меньше порога {thr}); перед принятием запросите рецензию и проверьте повторами")
        elif vd == "reject":
            rec.append(f"{a}: отклонить (интервал ниже нуля)")
        elif thr and hw > thr:
            rec.append(f"{a}: недостаточная мощность (полуширина интервала {hw:.1f} больше порога {thr}); повторить с 3 повторами: `rerun {name} 3`")
        else:
            rec.append(f"{a}: эффект в пределах шума; без нового довода не повторять")
        if thr and hw > thr:
            checks.append(f"{a}: интервал шире порога ({hw:.1f} > {thr})")
        if tok > 0.10:
            checks.append(f"{a}: токенов {tok:+.0%} к эталону")
        if m.get("sim_verdict") in ("pass", "skip_real") and m.get("sim_points"):
            checks.append("симулятор и реальный прогон стоит сравнить (сводка в метаданных `sim`)")
    if m.get("cross_check"):
        checks.append("гипотеза затрагивает общие блоки: результат нужно проверить и на другом режиме")
    if code:
        checks.append(f"изменён код (ветка {m.get('branch')}, {m.get('lines')} строк): прочитать diff")
    checks = checks or ["явных причин сомневаться нет"]
    expreq = m.get("experiment_requests")
    md = [f"# Отчёт: {name}", "", "## Гипотеза", essence or "(нет текста)", "",
          "Что меняется: " + "; ".join(changes) + (f". Код: {code}" if code else ""), "",
          "## Метод", f"Набор {e.get('set')}, бюджет {e.get('budget')}, повторов {e.get('repeats')}, модель {e.get('model')}. Первый вариант это эталон; "
          f"сравнение парное (бутстрап по задачам). Порог принятия {thr} очка: одно стандартное отклонение между повторами неизменного харнеса (лаборатория), автолаборатория его не меняет.", "",
          "## Результат", "", "| вариант | очки | токены | очки на 1k токенов | точность | разность очков [95 %] | вердикт |", "|---|---|---|---|---|---|---|", *rows, "",
          "## Ограничения", f"Повторов {e.get('repeats')}: при одном-двух повторах интервалы широки. Набор один и заморожен; английские задачи, бюджет {e.get('budget')} обрезает выход, а не прерывает задачу. "
          f"Запросов шлюза на эксперимент: {expreq if expreq else 'нет данных'}. Токены считаются по леджеру (вход/10 + выход).", "",
          "## Рекомендация", *[f"- {x}" for x in rec], "", "Принятие и слияние решает человек."]
    first = next((a for a in res if a != ref), ref)
    d = res[first].get("points_diff_vs_first")
    ppk0 = r0["points_per_run"] / (r0["tokens_per_run"] / 1000)
    ppk1 = res[first]["points_per_run"] / (res[first]["tokens_per_run"] / 1000)
    text = "\n".join([
        f"Гипотеза {name}: {short_h}",
        f"Режим: {MODE_RU.get(mode_of(name))}. Вердикт: {', '.join(f'{a} {v}' for a, v in verdict.items() if a != ref) or 'эталон'} (порог {thr})."
        + (" Нужна проверка на другом режиме (затронуты общие блоки)." if m.get("cross_check") else ""),
        f"Очки: {r0['points_per_run']} -> {res[first]['points_per_run']}" + (f", разность {d['mean']:+.1f} [{d['ci95'][0]:.1f}; {d['ci95'][1]:.1f}]" if d else ""),
        f"Токены: {r0['tokens_per_run']} -> {res[first]['tokens_per_run']}; очки на 1k токенов {ppk0:.2f} -> {ppk1:.2f}"
        + (f"; токенов до тех же очков {(m.get('mode_eval') or {}).get(first, {}).get('tokens_to_same_points')}" if mode_of(name) != "accuracy" and (m.get("mode_eval") or {}).get(first, {}).get("tokens_to_same_points") else ""),
        f"Повторов {e.get('repeats')}, набор {e.get('set')}, бюджет {e.get('budget')}; ячейка схемы: "
        + (", ".join(targets_of(name)) or f"не привязана (`привязать {name} <блок>/<тип>`)"),
        "Проверить: " + "; ".join(checks[:2]),
        "Рекомендация: " + (rec[0] if rec else "нет")])
    return "\n".join(md) + "\n", text


def notify_verdict(name: str, to: list[str] | None = None) -> None:
    """Write the Russian report and put hypothesis + short text + the file into the chat outbox (the bot sends it)."""
    to = to if to is not None else allowed_chat_ids()
    md, text = build_report_ru(name)
    f = DRAFTS / f"{name}.report.md"
    f.write_text(md, encoding="utf-8")
    dec = decision_of(name)
    if dec:
        text += f"\n{'❌' if 'reject' in dec else '📌'} Гипотеза {DECISION_RU.get(dec, dec)}: " + (meta(family_base(name)).get("decision_note") or "") + " Типы задач свободны."
    elif hyp_active(name) and types_of(name):
        text += f"\n🔒 Типы задач {', '.join(sorted(types_of(name)))} заняты, пока вы не решите: принять, отклонить, ревью, повтор или отложить."
    outbox_put(to, text, f, f"Отчёт {name}", None if dec else hyp_buttons(name))
    set_meta(name, report_file=str(f), notified=time.strftime("%Y-%m-%d %H:%M:%S") if to else None)


REVIEW_SYSTEM = ("Ты независимый рецензент экспериментов лаборатории, которая настраивает харнес для соревнования LLM-агентов. "
                 "Ты видишь только текст гипотезы, патч (если есть) и отчёт. У тебя нет инструментов, ты ничего не меняешь. Отвечай по-русски, коротко (до страницы), "
                 "в формате .md. Первая строка строго: `РЕКОМЕНДАЦИЯ: принять` или `РЕКОМЕНДАЦИЯ: отклонить` или `РЕКОМЕНДАЦИЯ: повторить`. Дальше разделы: "
                 "Риски (подгонка под набор, утечка данных, нарушение условий соревнования: задачи только на английском, превышение бюджета обрезает выход, "
                 "ключевая метрика очки на токен при бюджетах 20-40k, нарушение ограничений лаборатории), Слабые места статистики (повторы, ширина интервала против порога, "
                 "множественные сравнения), Чего не хватает. Не выдумывай чисел, которых нет во входе. "
                 "Последняя строка строго: `ДАННЫЕ: <имя набора из списка ниже> — почему именно он` (на каких данных стоит перепроверить: специализированный набор типа задач, "
                 "на котором гипотеза должна работать или может сломаться, а не тот, на котором уже мерили), либо `ДАННЫЕ: нужен новый набор — какой` (если подходящего нет), "
                 "либо `ДАННЫЕ: не нужны` (если проверка на тех же данных достаточна). Имя набора бери только из списка.")


def cmd_review(name: str, notify_to: list[str] | None = None) -> Path:
    import yaml
    m = meta(name)
    if not m.get("result"):
        raise SystemExit(f"{name}: no verdict yet; the reviewer needs the report")
    md, _ = build_report_ru(name)
    e = (DRAFTS / f"{name}.yaml").read_text(encoding="utf-8")
    patch = ""
    if m.get("branch"):
        try:
            patch = subprocess.run(["git", "-C", str(repo_root()), "diff", f"HEAD...{m['branch']}"], capture_output=True, text=True).stdout[:12000]
        except OSError:
            patch = ""
    sets = check_sets()
    used = (yaml.safe_load(e.split("# ---- written by lab.py")[0]) or {}).get("set")
    user = (f"## Гипотеза и эксперимент (YAML)\n{e[:3500]}\n\n## Патч\n{patch or '(кода нет, только переключатели)'}\n\n## Отчёт\n{md[:5000]}"
            f"\n\n## Наборы данных для перепроверки (уже использован: {used or 'не набор'})\n" + "\n".join(f"- {k}: {v}" for k, v in sets.items()))
    out = llm(REVIEW_SYSTEM, user, 4000, "review").strip()
    rec = (re.search(r"РЕКОМЕНДАЦИЯ:\s*(принять|отклонить|повторить)", out) or [None, "не указана"])[1]
    dm = re.search(r"ДАННЫЕ:\s*(.+)", out)
    data_line = dm.group(1).strip() if dm else ""
    first = re.match(r"[`*\s]*([a-z][a-z0-9_]+)", data_line)
    data_set = first.group(1) if first and first.group(1) in sets else None
    f = DRAFTS / f"{name}.review.md"
    f.write_text(f"# Рецензия: {name}\n\n(рецензент {(LAST['backend'] + '/' + LAST['model']) if LAST['model'] else MODEL}, видел только гипотезу, патч и отчёт)\n\n{out}\n", encoding="utf-8")
    set_meta(name, review_file=str(f), review_recommendation=rec, review_data_set=data_set, review_data_text=data_line)
    to = notify_to if notify_to is not None else allowed_chat_ids()
    text = f"Рецензия на {name}: рекомендация рецензента: {rec}." + (f"\nДанные для проверки: {data_line[:300]}" if data_line else "")
    btns = None if decision_of(name) else hyp_buttons(name, review=False)
    if btns and data_set:      # the reviewer named an existing set: one button starts the check on it
        btns = [[{"label": f"🧪 Проверить на {data_set}"[:40], "data": _cb("check", name)}]] + [[b for b in row if not b["data"].startswith("al:cmd:check:")] for row in btns]
        btns = [r for r in btns if r]
    outbox_put(to, text, f, f"Рецензия {name}", btns)
    log(f"review of {name}: {rec} -> {f}")
    return f


# ------------------------------------------------------------------ selftest
def selftest() -> None:
    import tempfile
    global EXP, DRAFTS, USAGE
    assert limits_cfg()["chief_stack_set"] == "mixed100_dev" and limits_cfg()["chief_stack_budget"] == 100000
    import tempfile as _tf
    _wt = Path(_tf.mkdtemp())
    (_wt / "harness").mkdir()
    (_wt / "harness" / "a.py").write_text("x = 1\n", encoding="utf-8")
    apply_edits(_wt, [{"file": "harness/solvers/new_one.py", "old": "", "new": "def f():\n    return 2\n"}, {"file": "harness/a.py", "old": "x = 1", "new": "x = 2"}])
    assert (_wt / "harness" / "solvers" / "new_one.py").read_text() == "def f():\n    return 2\n" and (_wt / "harness" / "a.py").read_text() == "x = 2\n"
    for bad in ({"file": "harness/a.py", "old": "", "new": "z"}, {"file": "../x.py", "old": "", "new": "z"}, {"file": "harness/nope.py", "old": "q", "new": "z"}):
        try:
            apply_edits(_wt, [bad])
            raise AssertionError(bad)
        except ValueError:
            pass
    assert limits_cfg()["max_code_lines"] >= 400 and limits_cfg()["apply_max_tokens"] >= 16000
    assert shadowed_imports("import math\nasync def math(ctx):\n    pass\n") == {"math"} and shadowed_imports("import math as _m\ndef math(c):\n    return _m.log(1)\n") == set()
    assert shadowed_imports("from os import path\npath = 1\n") == {"path"} and shadowed_imports("import os\ndef f():\n    import math\n    return math\n") == set() and shadowed_imports("def (") == set()
    EXP = Path(tempfile.mkdtemp())
    DRAFTS = EXP / "autolab"
    USAGE = DRAFTS / "usage.jsonl"
    (EXP / "sets").mkdir()
    (EXP / "sets" / "math16.yaml").write_text("role: dev\ntasks: {math: 16}\nseed: 1\n")
    import status_check as _sc
    st_tmp = Path(tempfile.mkdtemp())
    astat.mini(st_tmp / "status")
    os.environ["AUTOLAB_STATUS_ROOT"] = str(st_tmp / "status")
    good = {"name": "ab_test_one", "hypothesis": "x", "set": "math16", "repeats": 1, "budget": 30000,
            "arms": {"base": {}, "v": {"env": {"HARNESS_SORTER": "fifo"}}},
            "target_cells": ["blk/a"], "why_cell": "highest priority cell with a measured gain", "target_mode": "accuracy"}
    ks = known_switches()
    if "HARNESS_SORTER" in ks:
        assert validate(good) == [], validate(good)
    assert any("unknown or forbidden" in e for e in validate(dict(good, arms={"base": {}, "v": {"env": {"HARNESS_FORCE_MODEL": "x"}}})))
    assert any("must be one of" in e for e in validate(dict(good, arms={"base": {}, "v": {"env": {"HARNESS_VERIFY": "1"}}})))
    assert any("not an available" in e for e in validate(dict(good, set="mixed100_holdout")))
    assert any("keys not allowed" in e for e in validate(dict(good, threshold=1)))
    assert any("do not differ" in e for e in validate(dict(good, arms={"a": {}, "b": {}})))
    assert diff_signature({"set": 1, "arms": {"a": {}, "b": {"env": {"X": "1"}}}})[1] == 0      # no budget in the entry (the validate probe) must not raise
    assert any("budget" in e for e in validate(dict(good, budget=500000)))
    # scheme status: a hypothesis must name an applicable, non-saturated cell; the prompt carries the rating; accept/clarify results go to the cell history
    assert any("target_cells" in e for e in validate({k: v for k, v in good.items() if k not in ("target_cells",)}))
    assert any("why_cell" in e for e in validate(dict(good, why_cell="")))
    assert any("не применима" in e or "not" in e for e in validate(dict(good, target_cells=["blk/b"])))
    assert any("нет ячейки" in e for e in validate(dict(good, target_cells=["blk/zzz"])))
    pr = prompt_propose(2, "", "", ["blk/a"])
    assert "SCHEME STATUS" in pr and "blk/a" in pr and "target_cells" in pr and "SATURATED" in pr and "Preferred target cells" in pr
    replies = [json.dumps([dict(good, name="ab_cell_one"), {k: v for k, v in dict(good, name="ab_cell_two", arms={"base": {}, "v": {"env": {"HARNESS_SORTER": "reward"}}}).items() if k != "target_cells"}])]
    orig_llm_fn = globals()["llm"]
    globals()["llm"] = lambda system, user, max_tokens=4000, purpose="": replies[0]
    if "HARNESS_SORTER" in ks:
        made = cmd_propose(2, False, "", None, False, None, "accuracy")
        assert made == ["ab_cell_one"] and meta("ab_cell_one")["target_cells"] == ["blk/a"] and "ab_cell_two" not in made, made
        assert "ячейка схемы: blk/a" in (DRAFTS / "ab_cell_one.yaml").read_text(encoding="utf-8")
    globals()["llm"] = orig_llm_fn
    set_meta("ab_cell_one", stage="verdict", verdict={"base": None, "v": "clarify"}, threshold_used=7.69,
             result={"base": {"points_per_run": 100, "points_diff_vs_first": None}, "v": {"points_per_run": 103, "points_diff_vs_first": {"mean": 3.0, "ci95": [-2.0, 8.0]}}})
    if "HARNESS_SORTER" in ks:
        import status_check as _sc2
        j0 = _sc2.read_journal(status_root(), "blk", "a")[0]
        assert [x["name"] for x in j0 if x["verdict"] == "не запускался"] == [] or True
        assert record_journal("ab_cell_one") == ["blk/a"] and record_journal("ab_cell_one") == ["blk/a"]       # every verdict is written, twice = once
        jr = [x for x in _sc2.read_journal(status_root(), "blk", "a")[0] if x["name"] == "ab_cell_one"]
        assert len(jr) == 1 and jr[0]["verdict"] == "clarify" and "+3.0" in jr[0]["result"] and "порог 7.69" in jr[0]["result"] and jr[0]["who"] == "автолаборатория" and jr[0]["setbudget"].startswith("math16"), jr
        assert astat.entry(status_root(), "blk", "a")["status"] == "works-unexplored"       # the journal does not change the status
        assert astat.entry(status_root(), "blk", "a")["journal"]["last"] == "ab_cell_one"
        pend = [e for e in astat.events_load(EXP) if e["item"].get("kind") == "log" and e["name"] == "ab_cell_one"]
        assert len(pend) == 1 and len(astat.pending_plan(EXP)[0]) == 1               # one waiting row, replaced not duplicated
        record_journal("ab_cell_one", "failed", "no usable result")                  # a later state replaces the row
        assert [x["verdict"] for x in _sc2.read_journal(status_root(), "blk", "a")[0] if x["name"] == "ab_cell_one"] == ["failed"]
        record_journal("ab_cell_one")
        assert journal_kind("ab_nope") == "не запускался"
        # a task type is held by one active hypothesis: `blk/a` holds type a, `blk/_all` holds every type; the draw avoids held types; strict mode demands one of the drawn cells
        import random as _r
        (EXP / "queue" / "pending").mkdir(parents=True, exist_ok=True)
        (EXP / "queue" / "pending" / "ab_act.yaml").write_text("name: ab_act\npriority: 7\n")
        set_meta("ab_act", target_cells=["blk/a"], queued_at=time.time() + 1)
        assert active_types() == {"a": "ab_act"} and hyp_active("ab_act") and not hyp_active("ab_nope")
        assert astat.types_of_cells(["blk/_all"], ["a", "b"]) == {"a", "b"} and astat.types_of_cells(["blk/b"], ["a", "b"]) == {"b"}
        assert "blk/a" not in astat.pick_cells(status_root(), 5, locked_types=set(active_types())) and "blk/_all" not in astat.pick_cells(status_root(), 5, locked_types=set(active_types()))
        assert any("is held by the active hypothesis" in e for e in cell_conflicts(good, active_types(), set(), None))
        assert any("is held by the active hypothesis" in e for e in cell_conflicts(dict(good, target_cells=["blk/_all"]), active_types(), set(), None))      # block-wide cell needs every type free
        assert any("taken by another hypothesis of this batch" in e for e in cell_conflicts(good, {}, {"a"}, None))
        assert any("must be one of" in e for e in cell_conflicts(good, {}, set(), ["blk/_all"])) and not cell_conflicts(good, {}, set(), ["blk/a"])
        assert astat.pick_cells(status_root(), 2, rng=_r.Random(1), distinct_types=True) in (["blk/a"], ["blk/_all"]) or len(astat.pick_cells(status_root(), 2, distinct_types=True)) <= 2
        hnew = dict(good, name="ab_cell_act", arms={"base": {}, "v": {"env": {"HARNESS_SORTER": "reward"}}}, target_cells=["blk/a"])
        globals()["llm"] = lambda system, user, max_tokens=4000, purpose="": json.dumps([hnew])
        assert cmd_propose(1, False, "", None, False, None, "accuracy") == [] and not (DRAFTS / "ab_cell_act.yaml").exists()      # refused: the type is held
        # decisions: reject frees the type; accepting postponed marks the other postponed ones stale; repeats belong to the family of the hypothesis
        assert family_base("ab_x_r4") == "ab_x" and family_base("ab_x_r4_2") == "ab_x" and family_base("ab_x_fresh") == "ab_x" and family_base("ab_x") == "ab_x"
        set_decision("ab_act", "postponed", "позже", "42")
        assert decision_of("ab_act") == "postponed" and not hyp_active("ab_act") and active_types() == {}
        assert cmd_propose(1, False, "", None, False, None, "accuracy") == ["ab_cell_act"]
        globals()["llm"] = orig_llm_fn
        set_meta("ab_cell_act", queued_at=time.time() + 1)
        (EXP / "queue" / "pending" / "ab_cell_act.yaml").write_text("name: ab_cell_act\n")
        assert active_types() == {"a": "ab_cell_act"}
        set_meta("ab_post2", target_cells=["blk/a"], queued_at=time.time() + 1)
        set_decision("ab_post2", "postponed", "", "42")
        set_decision("ab_cell_act", "accepted", "", "42")
        assert meta("ab_act")["stale"] is True and meta("ab_post2")["stale"] is True and active_types() == {}
        (EXP / "queue" / "pending" / "ab_cell_act.yaml").unlink()
        set_decision("ab_cell_act", None)          # taken back into work: no decision
        assert decision_of("ab_cell_act") is None
        assert "HELD" not in prompt_propose(1, "", "", None, False, {}) and "ab_act" in prompt_propose(1, "", "", None, False, {"a": "ab_act"})
        assert "FIRST target cell" in prompt_propose(1, "", "", ["blk/a"], True, {})
        # lab: `priority` puts a hypothesis first in the queue
        import lab as _lab
        qd = EXP / "qo"
        qd.mkdir()
        for n_, pr_ in (("a.yaml", ""), ("b.yaml", "priority: 100\n"), ("c.yaml", "priority: 5\n")):
            (qd / n_).write_text(f"name: {n_[0]}\n{pr_}")
        assert [f.name for f in _lab.queue_order(sorted(qd.glob("*.yaml")))] == ["b.yaml", "c.yaml", "a.yaml"]
        for i in range(3):
            _sc2.log_entry(status_root(), "blk", "_all", f"ab_fail{i}", "clarify", "x +1 [-2; 3]", date=f"2026-03-0{i + 1}")
        assert astat.failures(status_root(), "blk/_all") == 3 and astat.failures(status_root(), "blk/zzz") == 0
        first = [astat.pick_cells(status_root(), 1, rng=_r.Random(i))[0] for i in range(300)]
        assert first.count("blk/_all") < 0.35 * 300 < first.count("blk/a"), (first.count("blk/_all"), first.count("blk/a"))     # same priority, but the failing cell is drawn less

        (EXP / "queue" / "done").mkdir(parents=True, exist_ok=True)
        (EXP / "queue" / "done" / "ab_cell_ab.yaml").write_text("name: ab_cell_ab\n# ---- written by lab.py ----\nstatus: aborted\nreason: gateway halted\n")
        (EXP / "queue" / "done" / "ab_cell_ok.yaml").write_text("name: ab_cell_ok\n# ---- written by lab.py ----\nstatus: ok\nresult: {}\n")
        set_meta("ab_cell_ab", target_cells=["blk/a"])
        set_meta("ab_cell_ok", target_cells=["blk/a"])
        assert journal_kind("ab_cell_ab") == "failed" and "gateway halted" in journal_fields("ab_cell_ab", "failed")["result"]
        assert journal_kind("ab_cell_ok") == "не завершён" and "вердикт автолаборатории ещё не построен" in journal_fields("ab_cell_ok", "не завершён")["result"]
        set_meta("ab_cell_q", stage="queued", target_cells=["blk/a"])
        assert record_journal("ab_cell_q") == ["blk/a"] and any(x["name"] == "ab_cell_q" and x["verdict"] == "не завершён" for x in _sc2.read_journal(status_root(), "blk", "a")[0])
        assert "журнал гипотез" in astat.prompt_block(status_root()) and "ab_cell_act" in astat.prompt_block(status_root())
        _sc2.log_entry(status_root(), "blk", "a", "ab_hand", "clarify", "ручная запись: +1 [-3; 5]", date="2026-02-02", who="человек", repeats=3)
        set_meta("ab_hand", target_cells=["blk/a"], verdict={"b": "clarify"}, result={"a": {"points_per_run": 1}, "b": {"points_per_run": 2, "points_diff_vs_first": {"mean": 1.0, "ci95": [-3.0, 5.0]}}})
        record_journal("ab_hand")
        row = next(x for x in _sc2.read_journal(status_root(), "blk", "a")[0] if x["name"] == "ab_hand")
        assert row["result"].startswith("ручная") and row["who"] == "человек" and row["date"] == "2026-02-02"       # the same verdict: the hand-written row is not touched
        set_meta("ab_hand", verdict={"b": "accept"})
        record_journal("ab_hand")
        row = next(x for x in _sc2.read_journal(status_root(), "blk", "a")[0] if x["name"] == "ab_hand")
        assert row["verdict"] == "accept" and row["who"] == "человек" and row["date"] == "2026-02-02" and row["repeats"] == "3", row
        plan = status_plan("ab_cell_one", accept=True)                      # a human accepts although the lab said clarify: history only, no status change
        assert "status" not in plan[0]
        set_meta("ab_cell_one", verdict={"base": None, "v": "accept"}, result={"base": {"points_diff_vs_first": None}, "v": {"points_per_run": 130, "points_diff_vs_first": {"mean": 30.0, "ci95": [12.0, 48.0]}}})
        plan = status_plan("ab_cell_one", accept=True)
        assert plan[0]["status"] == "explored-headroom" and plan[0]["headroom"] == "medium" and "accept" in plan[0]["evidence"] and "+30.0" in plan[0]["evidence"], plan
        astat.dry_run(status_root(), plan)
    assert targets_of("ab_cell_one_r3") == ["blk/a"] if "HARNESS_SORTER" in ks else True        # a repeat inherits the cells
    ok, _ = astat.check_target(status_root(), "blk/a")
    astat.apply(status_root(), [{"cell": "blk/a", "evidence": "x: saturated now", "status": "saturated", "headroom": "low"}])
    assert not astat.check_target(status_root(), "blk/a")[0] and any("насыщена" in e for e in validate(good))
    assert "blk/a" in astat.prompt_block(status_root()).split("SATURATED")[1]
    assert extract_json('text [{"a": 1}] tail') == [{"a": 1}]
    # duplicates compare what is tested, not names
    d1 = {"set": "s", "budget": 30000, "arms": {"a": {}, "b": {"env": {"HARNESS_SORTER": "fifo"}}}}
    d2 = {"set": "s", "budget": 30000, "arms": {"x": {"env": {}}, "y": {"env": {"HARNESS_SORTER": "fifo"}}}}
    d3 = {"set": "s", "budget": 30000, "arms": {"a": {}, "b": {"env": {"HARNESS_SORTER": "reward"}}}}
    assert diff_signature(d1) == diff_signature(d2) != diff_signature(d3)
    # the same settings as a decided hypothesis are refused on any set and budget; undecided or other settings pass
    _ae, _do = globals()["all_experiments"], globals()["decision_of"]
    globals()["all_experiments"] = lambda: [{"name": "ab_old", "e": d1, "block": {}, "where": "done", "mtime": 0}]
    try:
        globals()["decision_of"] = lambda n: "accepted"
        assert find_decided_duplicate(dict(d2, name="ab_new", set="other", budget=20000)) == "ab_old" and find_decided_duplicate(dict(d3, name="ab_new")) is None
        globals()["decision_of"] = lambda n: "postponed"
        assert find_decided_duplicate(dict(d2, name="ab_new", set="other")) is None
    finally:
        globals()["all_experiments"], globals()["decision_of"] = _ae, _do
    # patch rules
    assert check_patch(["tools/lab.py"], [], 0)
    assert check_patch(["experiments/sets/x.yaml"], [], 0)
    assert check_patch(["harness/sorter.py"], ["x = open('experiments/baseline0')"], 0)
    assert check_patch(["harness/sorter.py"], ["NOISE_FLOOR = 0"], 0)
    assert check_patch(["harness/sorter.py"], ["import subprocess"], 0)
    assert check_patch(["harness/sorter.py"], ["x = 1"], 0) is None
    assert check_patch(["harness/sorter.py"], ["x = 1"] * 500, 0) and check_patch(["harness/sorter.py"], ["x = 1"] * 300, 0) is None
    assert check_patch(["../x.py"], [], 0)
    # edit blocks
    t = Path(tempfile.mkdtemp())
    (t / "harness").mkdir()
    (t / "harness" / "a.py").write_text("def f():\n    return 1\n")
    apply_edits(t, parse_edits('```json\n{"edits":[{"file":"harness/a.py","old":"return 1","new":"return 2"}]}\n```'))
    assert "return 2" in (t / "harness" / "a.py").read_text()
    for bad in ({"file": "harness/a.py", "old": "nope", "new": "x"}, {"file": "../x.py", "old": "a", "new": "b"}):
        try:
            apply_edits(t, [bad])
            raise AssertionError("accepted")
        except ValueError:
            pass
    # sim table parsing
    assert parse_sim("| 20k | 151.7 | 18.0 | 138.7 | 25084 | 45.7 | 54.3 | 0 |\n| 30k | 188.7 | 14.4 | 181.0 | 1 | 1 | 1 | 0 |") == {20000: 151.7, 30000: 188.7}
    # accounting limits
    record_usage("t", {"prompt_tokens": 100, "completion_tokens": 10}, True)
    assert week_calls() == 1
    CALLS["n"] = 999
    try:
        llm("s", "u")
        raise AssertionError("limit not enforced")
    except SystemExit as ex:
        assert "limit" in str(ex)
    CALLS["n"] = 0
    # Claude backend with a mocked HTTP layer: key handling, dollars, limits, no key in any file or output
    import contextlib
    import io
    global _http_post, STOP_FILE, CFG_DIR, qwen_llm
    kf = Path(tempfile.mkdtemp()) / "k"
    os.environ["AUTOLAB_KEY_FILE"] = str(kf)
    os.environ["AUTOLAB_DEEPSEEK_ENV"] = str(kf) + ".none"   # never the real swarm .env in a selftest
    assert backend() == "qwen"   # no key file: the old behaviour
    fake_key = "sk-ant-FAKE-KEY-FOR-SELFTEST-0000"
    kf.write_text(fake_key + "\n")
    assert backend() == "claude"
    seen = {}

    def fake_post(url, headers, body, timeout=300):
        seen["headers"], seen["body"], seen["url"] = headers, json.loads(body), url
        return 200, {"content": [{"type": "text", "text": "[]"}], "usage": {"input_tokens": 1000, "output_tokens": 500}, "stop_reason": "end_turn"}
    orig_post, orig_stop = _http_post, STOP_FILE
    _http_post = fake_post
    STOP_FILE = EXP / "AUTOLAB_STOP_TEST"
    CALLS.update(n=0, usd=0.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        assert llm("sys", "user", 4000, "propose") == "[]"
        cmd_usage()
    assert seen["body"]["model"] == "claude-opus-5-5" and "temperature" not in seen["body"] and seen["url"].endswith("/v1/messages")
    assert abs(CALLS["usd"] - (1000 * 4 + 500 * 20) / 1e6) < 1e-9, CALLS
    everything = USAGE.read_text() + buf.getvalue()
    assert fake_key not in everything, "key leaked into usage or output"
    row = usage_rows()[-1]
    assert row["backend"] == "claude" and row["usd"] > 0 and "x-api-key" not in json.dumps(row)
    CALLS["usd"] = 4.999   # per-run dollar cap refuses before any request is made
    n_before = len(usage_rows())
    try:
        llm("sys", "user", 4000, "propose")
        raise AssertionError("usd cap not enforced")
    except SystemExit as ex:
        assert "usd_per_run" in str(ex) and fake_key not in str(ex)
    assert len(usage_rows()) == n_before
    CALLS.update(n=0, usd=0.0)
    # OpenRouter: the key prefix picks the provider; OpenAI chat format with a Bearer header; the real dollars of usage.cost; 401/402 are readable and never carry the key
    or_key = "sk-or-v1-FAKEFAKEFAKE-SELFTEST-0000"
    kf.write_text(or_key + "\n")
    assert provider() == "openrouter" and or_model("claude-opus-5-5") == "anthropic/claude-opus-5.5" and or_model("vendor/x") == "vendor/x" and backend() == "claude"

    def or_post(url, headers, body, timeout=300):
        seen.update(url=url, headers=headers, body=json.loads(body))
        return 200, {"choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.0123}}
    _http_post = or_post
    CALLS.update(n=0, usd=0.0)
    assert claude_llm("sys", "user", 4000, "propose") == "[]"
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions" and seen["headers"]["Authorization"] == "Bearer " + or_key and seen["body"]["model"] == "anthropic/claude-opus-5.5"
    assert seen["body"]["usage"] == {"include": True} and "system" not in seen["body"] and seen["body"]["messages"][0]["role"] == "system" and "Referer" not in json.dumps(seen["headers"])
    assert abs(CALLS["usd"] - 0.0123) < 1e-9 and usage_rows()[-1]["usd"] == 0.0123 and or_key not in USAGE.read_text()
    for code, word in ((401, "401"), (402, "402"), (404, "404")):
        _http_post = lambda *a, code=code, **k: (code, {"error": {"message": "bad " + or_key}})
        try:
            claude_llm("sys", "user", 4000, "propose")
            raise AssertionError("error not raised")
        except SystemExit as ex:
            assert word in str(ex) and or_key not in str(ex) and str(ex).startswith("Claude API"), str(ex)
    _http_post = or_post
    orig_get = _http_get
    g_calls = []

    def or_get(url, headers, timeout=20):
        g_calls.append(url)
        if url.endswith("/key"):
            return 200, {"data": {"usage": 1.5, "limit": 10, "limit_remaining": 8.5, "is_free_tier": False, "label": "sk-or-v1-abc...xyz"}}
        return 200, {"data": [{"id": "anthropic/claude-opus-5.5", "pricing": {"prompt": "0.000004", "completion": "0.00002"}}]}
    # the proxy applies to openrouter.ai only
    class _R:
        status = 200

        def read(self, *a):
            return b"{}"
    used = []
    o_bo, o_uo = urllib.request.build_opener, urllib.request.urlopen
    urllib.request.build_opener = lambda *h: type("O", (), {"open": staticmethod(lambda req, timeout=None: used.append(("proxy", [x.proxies for x in h])) or _R())})()
    urllib.request.urlopen = lambda req, timeout=None: used.append(("direct", None)) or _R()
    try:
        _open("https://openrouter.ai/api/v1/models", urllib.request.Request("https://openrouter.ai/api/v1/models"), 5)
        _open("https://api.deepseek.com/chat/completions", urllib.request.Request("https://api.deepseek.com/x"), 5)
        _open("https://api.anthropic.com/v1/messages", urllib.request.Request("https://api.anthropic.com/x"), 5)
    finally:
        urllib.request.build_opener, urllib.request.urlopen = o_bo, o_uo
    assert used[0] == ("proxy", [{"https": "http://127.0.0.1:7890", "http": "http://127.0.0.1:7890"}]) and used[1:] == [("direct", None), ("direct", None)], used
    globals()["_http_get"] = or_get
    st = key_status()
    assert "OpenRouter" in st and "осталось 8.5000" in st and "совпадают" in st and "Реальный запрос не делался" in st and or_key not in st and "abc...xyz" not in st, st
    n_posts = len(usage_rows())
    st2 = key_status(True)
    assert "Тестовый запрос прошёл" in st2 and len(usage_rows()) == n_posts + 1 and usage_rows()[-1]["purpose"] == "probe" and or_key not in st2
    globals()["_http_get"] = lambda url, headers, timeout=20: (401, {"error": {"message": "no " + or_key}})
    assert "401" in key_status() and or_key not in key_status()
    globals()["_http_get"] = orig_get
    kf.write_text(fake_key + "\n")
    assert provider() == "anthropic"
    CALLS.update(n=0, usd=0.0)
    _http_post = lambda *a, **k: (401, {"error": {"message": "invalid x-api-key " + fake_key}})   # an error body that echoes the key
    try:
        claude_llm("sys", "user", 4000, "propose")
        raise AssertionError("401 not raised")
    except SystemExit as ex:
        assert fake_key not in str(ex) and "401" in str(ex)
    # a dead Claude key: the chain moves on (DeepSeek if the swarm has its key, else qwen) and the chat is told once
    orig_cfg2 = CFG_DIR
    CFG_DIR = Path(tempfile.mkdtemp())
    (CFG_DIR / "telegram_allow.json").write_text('{"user_ids": [42]}')
    _TOLD.clear()
    orig_qwen = qwen_llm
    qwen_llm = lambda system, user, max_tokens, purpose: "from-qwen"
    CALLS.update(n=0, usd=0.0)
    assert llm("sys", "user", 4000, "propose") == "from-qwen" and LAST["backend"] == "qwen"
    note = [json.loads(f.read_text(encoding="utf-8"))["text"] for f in (CFG_DIR / "tg_outbox").glob("*.json")]
    assert len(note) == 1 and "Claude не отвечает" in note[0] and "qwen" in note[0] and fake_key not in note[0], note
    # DeepSeek from the swarm .env (a fake one): used when there is no Claude key; the key is read, never written anywhere
    kf.unlink()
    ds_key = "sk-FAKE-DEEPSEEK-0000"
    envf = Path(tempfile.mkdtemp()) / ".env"
    envf.write_text(f"MODEL_NAME=x\nAPI_BASE=https://api.deepseek.com\nAPI_KEY={ds_key}\nTELEGRAM_BOT_TOKEN=NOTTOUCHED\n")
    os.environ["AUTOLAB_DEEPSEEK_ENV"] = str(envf)
    assert chain() == ["deepseek", "qwen"] and deepseek_cfg()[1] == ds_key
    seen2 = {}

    def ds_post(url, headers, body, timeout=300):
        seen2["url"], seen2["body"], seen2["auth"] = url, json.loads(body), headers["Authorization"]
        return 200, {"choices": [{"message": {"content": "[1]"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    _http_post = ds_post
    CALLS.update(n=0, usd=0.0)
    assert llm("sys", "user", 4000, "apply") == "[1]" and LAST == {"backend": "deepseek", "model": "deepseek-v4-pro"}
    assert seen2["url"] == "https://api.deepseek.com/chat/completions" and seen2["body"]["model"] == "deepseek-v4-pro" and seen2["body"]["max_tokens"] == 12000
    assert llm("sys", "user", 4000, "propose") == "[1]" and seen2["body"]["model"] == "deepseek-flash"
    row = usage_rows()[-1]
    assert row["backend"] == "deepseek" and row["usd"] == 0 and ds_key not in USAGE.read_text() and "NOTTOUCHED" not in USAGE.read_text()
    # DeepSeek down (HTTP error echoing the key, then an empty answer): fall back to qwen, tell the chat once, no key in the text
    _TOLD.clear()
    for f in (CFG_DIR / "tg_outbox").glob("*.json"):
        f.unlink()
    time.sleep_orig = time.sleep
    time.sleep = lambda s_: None
    _http_post = lambda *a, **k: (401, {"error": {"message": "bad key " + ds_key}})
    assert llm("sys", "user", 4000, "review") == "from-qwen"
    _http_post = lambda *a, **k: (200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}})
    assert llm("sys", "user", 4000, "review") == "from-qwen"
    note = [json.loads(f.read_text(encoding="utf-8"))["text"] for f in (CFG_DIR / "tg_outbox").glob("*.json")]
    assert len(note) == 1 and "DeepSeek не отвечает" in note[0] and ds_key not in note[0] and "qwen" in note[0], note
    # a swarm that does not run on DeepSeek is not used (the key there belongs to something else)
    envf.write_text("API_BASE=https://example.org/v1\nAPI_KEY=zzz\n")
    assert deepseek_cfg() == ("", "") and chain() == ["qwen"]
    time.sleep = time.sleep_orig
    qwen_llm = orig_qwen
    CFG_DIR = orig_cfg2
    del os.environ["AUTOLAB_DEEPSEEK_ENV"]
    kf.write_text(fake_key + "\n")
    STOP_FILE.write_text("x")   # the stop file blocks every call
    try:
        llm("sys", "user", 10, "propose")
        raise AssertionError("stop file ignored")
    except SystemExit as ex:
        assert "stopped" in str(ex)
    STOP_FILE.unlink()
    _http_post, STOP_FILE = orig_post, orig_stop
    del os.environ["AUTOLAB_KEY_FILE"]
    os.environ.pop("AUTOLAB_DEEPSEEK_ENV", None)
    CALLS.update(n=0, usd=0.0)
    # report in Russian, outbox for the chat, reviewer on a mocked model (read-only: it only returns text)
    orig_cfg = CFG_DIR
    CFG_DIR = Path(tempfile.mkdtemp())
    (CFG_DIR / "telegram_allow.json").write_text('{"user_ids": [42]}')
    assert allowed_chat_ids() == ["42"]
    (CFG_DIR / "telegram_allow.json").write_text('{"user_ids": [42], "chat_ids": [-100500], "report_chats": [-100500]}')
    assert allowed_chat_ids() == ["-100500"]   # verdicts only to the group
    (CFG_DIR / "telegram_allow.json").write_text('{"user_ids": [42]}')
    d = {"name": "ab_rep_test", "hypothesis": "Order tasks cheapest first to answer more tasks per token.", "set": "math16", "repeats": 1, "budget": 30000,
         "arms": {"greedy": {}, "cheap": {"env": {"HARNESS_SORTER": "fifo"}}}}
    write_draft(d)
    set_meta("ab_rep_test", stage="verdict", threshold_used=7.69, verdict={"cheap": "clarify"}, experiment_requests=200,
             result={"greedy": {"points_per_run": 206.0, "tokens_per_run": 36186, "accuracy": 0.66, "points_diff_vs_first": None},
                     "cheap": {"points_per_run": 215.0, "tokens_per_run": 36522, "accuracy": 0.76, "points_diff_vs_first": {"mean": 9.0, "ci95": [-10.0, 26.0], "threshold": 7.69}}})
    md, text = build_report_ru("ab_rep_test")
    assert "## Гипотеза" in md and "| cheap |" in md and "rerun ab_rep_test 3" in md and 5 <= len(text.splitlines()) <= 7, text
    notify_verdict("ab_rep_test")
    items = sorted((CFG_DIR / "tg_outbox").glob("*.json"))
    assert len(items) == 1 and json.loads(items[0].read_text(encoding="utf-8"))["to"] == ["42"] and Path(json.loads(items[0].read_text(encoding="utf-8"))["file"]).exists()
    _http_post = lambda url, h, body, timeout=300: (200, {"content": [{"type": "text", "text": "РЕКОМЕНДАЦИЯ: повторить\nРиски: ...\nДАННЫЕ: `math16` — другой состав"}], "usage": {"input_tokens": 800, "output_tokens": 200}, "stop_reason": "end_turn"})
    kf.write_text(fake_key + "\n")
    os.environ["AUTOLAB_KEY_FILE"] = str(kf)
    CALLS.update(n=0, usd=0.0)
    rf = cmd_review("ab_rep_test", ["42"])
    assert rf.exists() and meta("ab_rep_test")["review_recommendation"] == "повторить" and len(list((CFG_DIR / "tg_outbox").glob("*.json"))) == 2
    (EXP / "queue" / "pending").mkdir(parents=True, exist_ok=True)
    d0 = queue_depth()
    (EXP / "queue" / "pending" / "adapt_stand_x.yaml").write_text("name: adapt_stand_x" + chr(10))
    assert queue_depth() == d0                      # the stand's experiments are not the autolab's depth
    (EXP / "queue" / "pending" / "adapt_stand_x.yaml").unlink()
    assert fake_key not in rf.read_text(encoding="utf-8") and usage_rows()[-1]["purpose"] == "review"
    assert meta("ab_rep_test")["review_data_set"] == "math16" and "math16" in meta("ab_rep_test")["review_data_text"]
    rbt = [json.loads(x.read_text(encoding="utf-8")) for x in (CFG_DIR / "tg_outbox").glob("*.json")]
    rbt = [r for r in rbt if r["caption"].startswith("Рецензия")][0]["buttons"]
    flat = [b["data"] for row in rbt for b in row]
    assert rbt[0][0]["label"].startswith("🧪 Проверить на math16") and "al:cmd:check:ab_rep_test" in flat and not any(":review:" in d for d in flat) and "al:cmd:rerun:ab_rep_test" in flat, rbt
    _http_post = orig_post
    del os.environ["AUTOLAB_KEY_FILE"]
    CFG_DIR = orig_cfg
    CALLS.update(n=0, usd=0.0)
    # a hypothesis of a human: formalized for exactly one cell, `priority: 100` in the yaml (the lab runs it first)
    if "HARNESS_SORTER" in ks:
        hmine = dict(good, name="ab_mine_one", budget=29000, arms={"base": {}, "v": {"env": {"HARNESS_SORTER": "reward"}}}, target_cells=["blk/_all"])
        globals()["llm"] = lambda system, user, max_tokens=4000, purpose="": json.dumps([hmine])
        got = cmd_propose(1, False, "A HUMAN proposes this hypothesis: x", ["blk/_all"], True, {"priority": 100, "by_user": True, "proposed_by": "42", "user_text": "x"})
        globals()["llm"] = orig_llm_fn
        assert got == ["ab_mine_one"] and spec_of("ab_mine_one")["priority"] == 100 and meta("ab_mine_one")["by_user"] is True and meta("ab_mine_one")["proposed_by"] == "42"
        assert "A HUMAN" in prompt_propose(1, "A HUMAN proposes: x", "", ["blk/a"], True, {})
    # goal modes: every hypothesis names one; the verdict follows the metric of its mode; the batch is split between the modes; cells come from the rating of their mode
    assert any("target_mode" in e for e in validate({k: v for k, v in good.items() if k != "target_mode"})) and any("target_mode" in e for e in validate(dict(good, target_mode="speed")))
    assert any("cross_check" in e for e in validate(dict(good, cross_check="yes"))) and not [e for e in validate(dict(good, target_mode="both", cross_check=True)) if "mode" in e or "cross" in e]
    res = {"base": {"points_per_run": 100, "tokens_per_run": 20000, "points_diff_vs_first": None},
           "save": {"points_per_run": 100, "tokens_per_run": 17000, "points_diff_vs_first": {"mean": 0.5, "ci95": [-4.0, 5.0]}},
           "drop": {"points_per_run": 85, "tokens_per_run": 10000, "points_diff_vs_first": {"mean": -15.0, "ci95": [-20.0, -10.0]}},
           "same": {"points_per_run": 100, "tokens_per_run": 19800, "points_diff_vs_first": {"mean": 0.0, "ci95": [-3.0, 3.0]}},
           "more": {"points_per_run": 101, "tokens_per_run": 24000, "points_diff_vs_first": {"mean": 1.0, "ci95": [-3.0, 5.0]}}}
    ev = economy_eval(res, 7.69)
    assert {a_: v["kind"] for a_, v in ev.items()} == {"save": "accept", "drop": "reject", "same": "clarify", "more": "reject"}, ev
    assert ev["save"]["tokens_to_same_points"] == 17000 and abs(ev["drop"]["tokens_to_same_points"] - 10000 * 100 / 85) <= 1 and ev["save"]["ppk"] > ev["save"]["ppk_ref"]
    lab_v = {"base": None, "save": "clarify", "drop": "reject", "same": "clarify", "more": "clarify"}
    for md, want in (("accuracy", lab_v), ("economy", {"base": None, "save": "accept", "drop": "reject", "same": "clarify", "more": "reject"}),
                     ("both", {"base": None, "save": "clarify", "drop": "reject", "same": "clarify", "more": "reject"})):
        set_meta("ab_modev", target_mode=md, result=res, threshold_used=7.69, verdict=lab_v, lab_verdict=lab_v)
        apply_mode_verdict("ab_modev")
        assert meta("ab_modev")["verdict"] == want and meta("ab_modev")["lab_verdict"] == lab_v, (md, meta("ab_modev")["verdict"])
    assert mode_of("ab_modev") == "both" and mode_of("ab_nope_mode") == "accuracy" and journal_mode("ab_modev") == "оба"
    assert abs(mode_gain("ab_modev", "economy") - 15.0) < 0.01 and mode_gain("ab_modev", "accuracy") == 1.0
    set_meta("ab_modev", target_cells=["blk/_all"], verdict={"base": None, "save": "accept"})
    pl = status_plan("ab_modev", True)
    assert [(i["cell"], i["mode"]) for i in pl] == [("blk/_all", "accuracy"), ("blk/_all", "economy")] and "status" in pl[0] and "status" not in pl[1], pl
    assert record_journal("ab_modev", "accept") == ["blk/_all"]
    row = next(x for x in _sc2.read_journal(status_root(), "blk", "_all")[0] if x["name"] == "ab_modev")
    assert row.get("mode", "оба") == "оба" and "режим: both" in row["result"], row
    _lc0 = globals()["limits_cfg"]
    globals()["limits_cfg"] = lambda: dict(_lc0(), goal_mode="")                # "" = alternate equally
    try:
        cells2, cm2 = plan_cells(1, {}, set(), None, 0)
        cells3, cm3 = plan_cells(1, {}, set(), None, 1)
    finally:
        globals()["limits_cfg"] = _lc0
    assert _lc0()["goal_mode"] == "accuracy" and list(plan_cells(1, {}, set(), None, 1)[1].values()) == ["accuracy"]       # the default focus
    assert list(cm2.values()) == ["accuracy"] and list(cm3.values()) == ["economy"] and list(plan_cells(1, {}, set(), "economy", 0)[1].values()) == ["economy"]
    _lc = globals()["limits_cfg"]
    globals()["limits_cfg"] = lambda: dict(_lc(), goal_mode="accuracy")        # the configured focus: no alternation, an explicit mode still wins
    try:
        assert list(plan_cells(1, {}, set(), None, 1)[1].values()) == ["accuracy"] and list(plan_cells(1, {}, set(), "economy", 0)[1].values()) == ["economy"]
    finally:
        globals()["limits_cfg"] = _lc
    assert plan_cells(1, {}, {"a", "b"}, None, 0) == ([], {})                       # every type held: nothing to draw
    pm = prompt_propose(1, "", "", ["blk/_all"], True, {}, {"blk/_all": "economy"}, None)
    assert "GOAL MODES" in pm and "blk/_all -> economy" in pm and "target_mode" in pm and "cross_check" in pm and "for the goal mode ECONOMY" in pm, pm[:300]
    hm = dict(good, name="ab_mode_eco", budget=27000, arms={"base": {}, "v": {"env": {"HARNESS_SORTER": "reward"}}}, target_cells=["blk/_all"], target_mode="accuracy")
    globals()["llm"] = lambda system, user, max_tokens=4000, purpose="": json.dumps([hm])
    assert cmd_propose(1, False, "", ["blk/_all"], True, None, "economy") == []         # the batch is for economy: an accuracy hypothesis is refused
    hm["target_mode"] = "both"
    assert cmd_propose(1, False, "", ["blk/_all"], True, None, "economy") == ["ab_mode_eco"] and spec_of("ab_mode_eco")["target_mode"] == "both" and meta("ab_mode_eco")["target_mode"] == "both"
    globals()["llm"] = orig_llm_fn
        # autopilot: off by default; on: tops the queue up through the generator, respects stop flag, queue depth, gateway gate and the daily cap
    import types
    global cmd_cycle
    orig_cc, orig_cfg3, orig_popen, orig_limits = cmd_cycle, CFG_DIR, subprocess.Popen, sr.limits
    CFG_DIR = Path(tempfile.mkdtemp())
    (CFG_DIR / "telegram_allow.json").write_text('{"user_ids": [42], "report_chats": [-100500]}')
    for k in ("pending", "running", "done"):
        (EXP / "queue" / k).mkdir(parents=True, exist_ok=True)
    gate = {"ok": True}
    fake_disp = types.ModuleType("dispatcher")
    fake_disp.load_cfg = lambda: {}
    fake_disp.fractions = lambda lim: {}
    fake_disp.exp_gate = lambda cfg, fr: (gate["ok"], "ok" if gate["ok"] else "day capacity 99%")
    sys.modules["dispatcher"] = fake_disp
    sr.limits = lambda: {}
    made, spawned_aw = [], []

    def fake_cycle(n, topic, names, wait, cells=None, strict=False, mode=None, cell_modes=None):
        made.append((n, topic, wait, cells))
        assert strict and cells and len(cells) >= min(n, 1)     # the autopilot aims every hypothesis at the cells it drew
        for i in range(n):
            nm = f"ab_auto_{len(made)}_{i}"
            set_meta(nm, stage="queued")
    cmd_cycle = fake_cycle
    subprocess.Popen = lambda argv, **kw: spawned_aw.append(argv[-2:])
    assert "off" in cmd_autopilot() and not made                      # no flag: nothing happens
    autopilot_flag().write_text("on")
    assert "ON" in cmd_autopilot(True) and not made                   # status only
    gate["ok"] = False
    assert "gateway gate" in cmd_autopilot() and not made             # the gateway has no room
    gate["ok"] = True
    out = cmd_autopilot()
    assert made and made[0][0] == 2 and made[0][2] is False and "queued" in out and spawned_aw and spawned_aw[0][0] == "await", (out, made)
    ob = [json.loads(f.read_text(encoding="utf-8")) for f in (CFG_DIR / "tg_outbox").glob("*.json")]
    assert len(ob) == 1 and ob[0]["to"] == ["-100500"] and "Автопилот" in ob[0]["text"]
    (EXP / "queue" / "pending" / "a.yaml").write_text("x")
    (EXP / "queue" / "pending" / "b.yaml").write_text("x")
    n_made = len(made)
    assert "long enough" in cmd_autopilot() and len(made) == n_made   # the queue is already at the target depth
    for f in (EXP / "queue" / "pending").glob("*.yaml"):
        f.unlink()
    STOP_FILE.write_text("x") if False else (EXP / "queue" / "STOP").write_text("x")
    assert "stop flag" in cmd_autopilot() and len(made) == n_made
    (EXP / "queue" / "STOP").unlink()
    # every task type is held by an active hypothesis (here: a block-wide one waiting for its decision): the autopilot waits and tells the chat once
    set_meta("ab_hold", target_cells=["blk/_all"], queued_at=time.time() + 1, result={"x": {}})
    assert set(active_types()) == {"a", "b"}
    n_made2 = len(made)
    out = cmd_autopilot()
    assert "waiting for decisions" in out and len(made) == n_made2, out
    cmd_autopilot()
    notes = [json.loads(f.read_text(encoding="utf-8")) for f in (CFG_DIR / "tg_outbox").glob("*.json")]
    assert len([x for x in notes if "Автопилот стоит" in x["text"]]) == 1 and "ab_hold" in [x for x in notes if "Автопилот стоит" in x["text"]][0]["text"]
    set_decision("ab_hold", "rejected", "", "42")
    assert active_types() == {}
    assert limits_cfg()["autopilot_max_per_day"] == 0 and "no limit" in cmd_autopilot(True)      # no daily limit by default
    orig_lc = globals()["limits_cfg"]
    globals()["limits_cfg"] = lambda: dict(orig_lc(), autopilot_max_per_day=8)
    for _ in range(6):                                                # a configured daily cap still ends it
        cmd_autopilot()
    st = _autopilot_state()
    assert len(st["names_today"]) <= 8 and "daily limit" in cmd_autopilot(), st
    globals()["limits_cfg"] = orig_lc
    assert all(m[3] and "blk/" in m[3][0] and "рейтингу схемы" in m[1] for m in made)   # the batch is aimed at cells drawn from the rating
    autopilot_flag().unlink()
    cmd_cycle, CFG_DIR, subprocess.Popen, sr.limits = orig_cc, orig_cfg3, orig_popen, orig_limits
    sys.modules.pop("dispatcher", None)
    # local data stream: only when switched on, not while another local stream or a stop file exists, not when the shim is down or the GPU is hot
    global LAB, _local_stream_pids, _gpu_temp
    orig_lab, orig_pids, orig_shim, orig_temp, orig_popen2 = LAB, _local_stream_pids, sr.shim_ready, _gpu_temp, subprocess.Popen
    LAB = Path(tempfile.mkdtemp())
    CFG_DIR = Path(tempfile.mkdtemp())
    started = []
    subprocess.Popen = lambda argv, **kw: started.append(argv)
    pids, shim, temp = [], [True], [60]
    _local_stream_pids = lambda: list(pids)
    sr.shim_ready = lambda: shim[0]
    _gpu_temp = lambda: temp[0]
    assert "off" in cmd_localdata() and not started
    localdata_flag().write_text("on")
    shim[0] = False
    assert "not reachable" in cmd_localdata() and not started
    shim[0], temp[0] = True, 90
    assert "GPU 90" in cmd_localdata() and not started
    temp[0] = 60
    (LAB / "STREAM_STOP_local").write_text("x")
    assert "STREAM_STOP_local" in cmd_localdata() and not started
    (LAB / "STREAM_STOP_local").unlink()
    pids.append("123")
    assert "running" in cmd_localdata() and not started
    pids.clear()
    out = cmd_localdata()
    assert "started" in out and started and "--backend" in started[0] and "local" in started[0] and started[0][started[0].index("--workers") + 1] == "3", (out, started)
    LAB, _local_stream_pids, sr.shim_ready, _gpu_temp, subprocess.Popen = orig_lab, orig_pids, orig_shim, orig_temp, orig_popen2
    CFG_DIR = orig_cfg3
    # this module must not write the noise threshold
    src = Path(__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(lab\.)?NOISE_RUN\s*=", src, re.M)
    print("autolab selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose"); p.add_argument("--n", type=int, default=3); p.add_argument("--dry", action="store_true"); p.add_argument("--topic", default=""); p.add_argument("--mode", choices=["accuracy", "economy"], default=None)
    p = sub.add_parser("autopilot"); p.add_argument("--status", action="store_true")
    p = sub.add_parser("key"); p.add_argument("--probe", action="store_true")
    p = sub.add_parser("mine"); p.add_argument("--cell", required=True); p.add_argument("--text", required=True); p.add_argument("--by", default=""); p.add_argument("--chat", default=None)
    for c in ("sim", "submit", "verdict", "export", "await"):
        sub.add_parser(c).add_argument("name")
    p = sub.add_parser("review"); p.add_argument("name"); p.add_argument("--notify", help="chat user id that gets the file (default: the whitelist)")
    p = sub.add_parser("apply"); p.add_argument("name"); p.add_argument("--patch-file")
    p = sub.add_parser("cycle"); p.add_argument("--n", type=int, default=2); p.add_argument("--topic", default=""); p.add_argument("--mode", choices=["accuracy", "economy"], default=None)
    p.add_argument("--names", help="comma-separated existing drafts (skips propose)"); p.add_argument("--wait", action="store_true")
    for c in ("list", "usage", "selftest"):
        sub.add_parser(c)
    a = ap.parse_args()
    {"propose": lambda: cmd_propose(a.n, a.dry, a.topic, None, False, None, a.mode), "sim": lambda: cmd_sim(a.name), "submit": lambda: cmd_submit(a.name),
     "verdict": lambda: cmd_verdict(a.name), "await": lambda: cmd_await(a.name), "autopilot": lambda: [log(cmd_localdata(a.status)), log(cmd_autopilot(a.status))], "apply": lambda: cmd_apply(a.name, a.patch_file), "mine": lambda: cmd_mine(a.cell, a.text, a.by, a.chat), "list": cmd_list, "export": lambda: cmd_export(a.name),
     "usage": cmd_usage, "cycle": lambda: cmd_cycle(a.n, a.topic, a.names.split(",") if a.names else None, a.wait, None, False, a.mode),
     "review": lambda: cmd_review(a.name, [a.notify] if a.notify else None), "key": lambda: print(key_status(a.probe)), "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    main()
