#!/usr/bin/env python3
"""Control of the autolab from a chat (Telegram), independent of the transport. Lab tool, runs on the GPU host next to autolab.py.

The transport adapter (tools/autolab_tg_adapter.py) only forwards `(user_id, text)` to `handle()` and button presses to
`handle_callback()`; everything that decides lives here and is testable without Telegram or Claude (`selftest`).

Commands (`/autolab <command>`):
  status                 gateway share, dispatcher, queue, stop flags, autolab spend
  queue                  pending / running / last finished experiments
  usage                  gateway requests and Claude dollars of the autolab
  list                   drafts with stage, verdict and branch
  report <name>          verdict file of an experiment
  verdict <name>         (re)build the verdict of a finished experiment (reads results, spends nothing)
  propose [n] [topic]    generator only (spends Claude dollars)            -> asks for confirmation
  cycle [n] [topic]      the whole cycle on a new batch of hypotheses       -> asks for confirmation
  rerun <name> <n>       the same experiment with n repeats under a new name -> asks for confirmation
  report <name> | отчёт  the report (Russian .md file + 5-7 lines); a finished hypothesis is also sent automatically
  review <name> | рецензия   an independent reviewer (Claude, read-only: it sees only the hypothesis, the patch and the report) -> .md file
  accept <name> | принять    merge the branch into the host copy of the lab (~/harness-lab/autolab_repo) AND push master to GitHub; one button
  stop | resume          touch / remove the stop files (no more LLM calls and no submits; the queue leaves after the running experiment)
Rules enforced here:
  * only Telegram user ids listed in ~/.config/autolab/telegram_allow.json ({"user_ids": [123]}) are served; a missing or empty
    file rejects everybody; rejected users get no reply (the denial is journaled);
  * every command and decision goes to ~/.config/autolab/tg_journal.jsonl (outside the repository, no secrets);
  * a command that spends (LLM or gateway requests) is first answered with an estimate and a one-time code; it runs only after the code
    comes back (button or `/autolab ok <code>`) from the SAME user within 10 minutes; a code works once;
  * `accept` shows the verdict, the reviewer's recommendation (if a review exists; it never blocks) and the diffstat, and needs ONE button
    (one-time code bound to the user and to the branch commit that was shown). It merges into ~/harness-lab/autolab_repo, then applies the same
    patch on a clone of the GitHub repository and pushes master (never --force). Without the deploy key it merges and says "push не выполнен: нет
    ключа". A patch that touches anything but harness/, profiles/, run_agent.py (the deny-list of `autolab.py apply`), the lab, sets or
    thresholds is refused. Every decision is journaled with the user id.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import autolab as al  # noqa: E402
import autolab_status as astat  # noqa: E402

CFG_DIR = Path(os.path.expanduser(os.environ.get("AUTOLAB_CONFIG_DIR") or "~/.config/autolab"))
CODE_TTL = 600       # seconds a one-time code lives


def _allow_file() -> Path:
    return CFG_DIR / "telegram_allow.json"


def _cfg() -> dict:
    try:
        d = json.loads(_allow_file().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}   # no file or a broken file: everybody is rejected


def allowed(uid, chat=None, chat_type: str = "private") -> bool:
    """The user must be whitelisted; in a group the chat must be listed too (`chat_ids`), a private chat is always fine.
    Nobody is allowed when the list is empty or the file is missing."""
    cfg = _cfg()
    if str(uid) not in {str(x) for x in (cfg.get("user_ids") or [])}:
        return False
    if chat_type in ("group", "supergroup") and str(chat) not in {str(x) for x in (cfg.get("chat_ids") or [])}:
        return False
    return True


def whoami(uid, chat, chat_type: str) -> dict:
    """Answers anybody with the ids they need for the config (their own user id and this chat id); changes nothing."""
    journal(uid, "whoami", chat=str(chat), chat_type=chat_type)
    return reply(f"Ваш user_id: {uid}\nЭтот чат: {chat} ({chat_type})\n"
                 "Эти числа нужно вписать в ~/.config/autolab/telegram_allow.json (user_ids, chat_ids, report_chats).")


def journal(uid, event: str, **kw) -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CFG_DIR / "tg_journal.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "user": str(uid), "event": event, **kw}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ one-time codes
def _pending() -> dict:
    try:
        return json.loads((CFG_DIR / "tg_pending.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_pending(p: dict) -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    p = {k: v for k, v in p.items() if v["expires"] > now}
    tmp = CFG_DIR / "tg_pending.json.tmp"
    tmp.write_text(json.dumps(p), encoding="utf-8")
    os.replace(tmp, CFG_DIR / "tg_pending.json")


def issue(uid, action: str, args: dict, ttl: int = CODE_TTL) -> str:
    p = _pending()
    code = secrets.token_hex(3)
    p[code] = {"user": str(uid), "action": action, "args": args, "expires": time.time() + ttl}
    _save_pending(p)
    return code


def consume(uid, code: str, actions: tuple) -> tuple[dict | None, str]:
    """(entry, why-not). The code is removed on every successful AND wrong-user attempt is not: only the owner can burn it."""
    p = _pending()
    e = p.get(code)
    if not e or e["expires"] < time.time():
        return None, "unknown or expired code"
    if e["user"] != str(uid):
        return None, "this code belongs to another user"
    if e["action"] not in actions:
        return None, f"this code needs another kind of confirmation ({e['action']})"
    del p[code]
    _save_pending(p)
    return e, ""


def reply(text: str, buttons: list | None = None, document: str | Path | None = None) -> dict:
    return {"text": text[:3900], "buttons": buttons or [], "document": str(document) if document else ""}


def confirm_buttons(code: str, label: str = "Confirm") -> list:
    return [[{"label": label, "data": f"al:ok:{code}"}, {"label": "Отмена", "data": f"al:cancel:{code}"}]]


# ------------------------------------------------------------------ read-only commands
def _capture(fn, *a) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(*a)
        except SystemExit as ex:
            print(f"(stopped: {ex})")
    return buf.getvalue().strip()


def _stop_files() -> list[Path]:
    return [al.STOP_FILE, al.EXP / "queue" / "STOP"]


STAGE_RU = {"proposed": "предложена", "applied": "патч готов", "sim_skip": "отсеяна симулятором", "queued": "в очереди", "verdict": "есть вердикт",
            "accepted": "принята", "apply_refused": "патч не применился", "sim_pass": "прошла симулятор", "rejected": "отклонена", "failed": "ошибка"}
VERDICT_RU = {"accept": "лучше (за порогом)", "reject": "хуже (за порогом)", "clarify": "неясно (в пределах шума)", "inconclusive": "неясно"}


def _verdict_ru(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}: {VERDICT_RU.get(x, x)}" for k, x in v.items())
    return VERDICT_RU.get(str(v), str(v)) if v else ""


def cmd_status() -> str:
    lines = ["🧪 Лаборатория"]
    try:
        import dispatcher
        import stream_run as sr
        fr = dispatcher.fractions(sr.limits())
        n = lambda x: f"{x:,}".replace(",", " ")
        lines.append(f"Шлюз: неделя {fr['week_pct']:.1f}% ({n(fr['week_used'])} из {n(fr['week_limit'])} запросов), сутки {fr['day_pct']:.0f}%, в окне осталось {n(fr['session_left'])}")
    except Exception as ex:  # noqa: BLE001
        lines.append(f"Шлюз: не читается ({type(ex).__name__})")
    lines.append("Диспетчер: " + ("работает ✅" if al.dispatcher_alive() else "не запущен ⛔"))
    q = {k: sorted(p.stem for p in (al.EXP / "queue" / k).glob("*.yaml")) for k in ("pending", "running")}
    lines.append("Идёт сейчас: " + (", ".join(q["running"]) or "ничего"))
    lines.append("Ждёт в очереди: " + (", ".join(q["pending"]) or "ничего"))
    stops = [p.name for p in _stop_files() if p.exists()]
    lines.append("Стоп-флаги: " + (("🛑 " + ", ".join(stops)) if stops else "нет"))
    cfg = al.limits_cfg()
    be, model = al.backend_desc("propose")
    money = f", Claude ${al.week_usd():.2f} из ${cfg['usd_per_week']:.0f}" if be.startswith("claude") else ""
    lines.append(f"Автолаборатория: модель {be}/{model}, запросов LLM за неделю {al.week_calls()} из {cfg['weekly_llm_calls']}{money}")
    return "\n".join(lines)


def cmd_queue() -> str:
    out = []
    for k, title in (("running", "▶️ Идёт"), ("pending", "⏳ Ждёт"), ("done", "✅ Готово (последние 6)")):
        files = sorted((al.EXP / "queue" / k).glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=(k == "done"))
        rows = []
        for f in (files[:6] if k == "done" else files):
            e, blk = al._split_lab_block(f.read_text(encoding="utf-8"))
            v = _verdict_ru((blk.get("result") or {}).get("verdict"))
            rows.append(f"• {f.stem}" + (f" — {v}" if v else ""))
        if rows:
            out.append(title + "\n" + "\n".join(rows))
    return "\n\n".join(out) or "Очередь пуста."


LIST_GROUPS = [("decide", "🔔", "Ждут вашего решения (держат типы задач)"), ("running", "🔄", "Идут или ждут очереди (держат типы)"),
               ("noverdict", "🕓", "Закончились в лаборатории, вердикт не построен (старые, типы не держат)"),
               ("accept", "✅", "Лучше эталона (accept)"), ("clarify", "⚖️", "Неясно, в пределах шума (clarify)"),
               ("reject", "❌", "Хуже эталона (reject)"), ("failed", "⛔", "Без результата (ошибка или прервано)"), ("draft", "📝", "Придуманы, ещё не запускались"),
               ("skipped", "🚫", "Отсеяны до запуска"), ("postponed", "⏸", "Отложены (не активны; полный список: отложенные)"), ("closed", "📦", "Приняты или отклонены")]


def _kind_group(name: str) -> str:
    m = al.meta(name)
    if m.get("stage") in ("filtered_by_sim", "sim_skip", "apply_refused"):
        return "skipped"
    k = al.journal_kind(name)
    return {"accept": "accept", "clarify": "clarify", "reject": "reject", "failed": "failed", "не завершён": "running", "не запускался": "draft"}.get(k, "draft")


def _hyp_group(name: str) -> str:
    """Where a hypothesis stands in the list: closed by a decision, postponed, active (holds task types), or by its result."""
    dec = al.decision_of(name)
    if dec == "postponed":
        return "postponed"
    if dec:
        return "closed"
    if al.hyp_active(name):
        return "decide" if al.meta(name).get("result") else "running"
    k = _kind_group(name)
    if k == "running" and not any((al.EXP / "queue" / d / f"{name}.yaml").exists() for d in ("pending", "running")):
        return "noverdict"                  # finished in the lab before decisions existed, no verdict built: holds nothing
    return k


def _hyp_line(name: str, group: str) -> str:
    if group == "noverdict":
        group = "running"
    if group in ("decide", "postponed", "closed"):
        kind = _kind_group(name)
        body = _hyp_line(name, kind if kind != "running" else "running")
        g = al.hyp_gain(name)
        tag = {"decide": "🔒 " + ", ".join(sorted(al.types_of(name))), "postponed": "⚠️ устарела" if al.meta(al.family_base(name)).get("stale") else "актуальна",
               "closed": al.DECISION_RU.get(al.decision_of(name) or "", "")}[group]
        return body + (f"{chr(10)}   {tag}" if tag.strip() else "")
    g = al.hyp_gain(name) if group in ("running", "draft") else None
    m = al.meta(name)
    cells = ", ".join(al.targets_of(name)) or "без ячейки"
    out = f"• {name}\n   ячейка: {cells} · режим: {al.MODE_RU.get(al.mode_of(name))}"
    res = m.get("result") or {}
    best = None
    for a_, v in res.items():
        d = v.get("points_diff_vs_first")
        if d and (best is None or d["mean"] > best[1]["mean"]):
            best = (a_, d)
    if group in ("accept", "clarify", "reject") and best:
        out += f"\n   {best[0]}: {best[1]['mean']:+.1f} очков [{best[1]['ci95'][0]:.1f}; {best[1]['ci95'][1]:.1f}], порог {m.get('threshold_used')}"
    elif group == "running":
        out += "\n   " + ("идёт сейчас" if (al.EXP / "queue" / "running" / f"{name}.yaml").exists() else ("ждёт вердикта" if al.done_block(name) is not None else "в очереди"))
    elif group == "failed":
        blk = al.done_block(name) or {}
        out += "\n   " + str(blk.get("reason") or blk.get("result_error") or "нет результата")[:90]
    elif group == "skipped":
        out += "\n   " + str(m.get("sim_reason") or m.get("stage_note") or STAGE_RU.get(m.get("stage", ""), "отсеяна"))[:90]
    elif group == "draft":
        h = " ".join(str(al.spec_of(name).get("hypothesis") or "").split("[ячейка схемы")[0].split())
        if h:
            out += "\n   " + h[:90] + ("…" if len(h) > 90 else "")
    if g is not None:
        out += f"\n   ожидаемый выигрыш: {g:+.1f}"
    return out


def lock_summary() -> str:
    """Which task types are held by an active hypothesis and which are free: at most one active hypothesis per type."""
    root = _status_root()
    if not astat.index(root):
        return ""
    held = al.active_types()
    free = [t for t in astat.index_types(root) if t not in held]
    return ("🔒 Заняты: " + ", ".join(f"{t} ({w})" for t, w in held.items()) if held else "🟢 Все типы задач свободны") + ("\n🟢 Свободны: " + ", ".join(free) if held and free else "")


def _gain_key(n: str) -> float:
    g = al.hyp_gain(n)
    return -g if g is not None else float("inf")        # the biggest gain first, unknown last


def cmd_list(show_all: bool = False) -> str:
    names = [f.stem for f in sorted(al.DRAFTS.glob("*.yaml"), key=lambda f: -f.stat().st_mtime) if not f.name.endswith(".queued.yaml")]
    if not names:
        return "Гипотез пока нет."
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(_hyp_group(n), []).append(n)
    cap = 50 if show_all else 4
    if not show_all:        # only what needs a decision or is in flight; the rest lives in the journal
        npost = len(groups.get("postponed", []))
        act = {g: groups[g] for g in ("decide", "running") if g in groups}
        out = [f"📋 Активные гипотезы: {sum(len(v) for v in act.values())} (по убыванию выигрыша); остальные в журнале", lock_summary(), ""]
        if not act:
            out += ["Активных гипотез нет: решать нечего. Новые придумает автопилот, или предложите свою: `своя <блок>/<тип> <текст>`."]
        for g, ic, title in LIST_GROUPS:
            if g in act:
                out.append(f"{ic} {title} — {len(act[g])}")
                out += [_hyp_line(n, g) for n in sorted(act[g], key=_gain_key)]
                out.append("")
        out.append(f"Кнопки ниже открывают карточку с решениями. Отложено: {npost}. Все гипотезы: `гипотезы все`.")
        return "\n".join(x for x in out if x is not None)
    out = [f"📋 Гипотезы: всего {len(names)} (внутри групп по убыванию выигрыша)", " · ".join(f"{ic} {len(groups[g])}" for g, ic, _ in LIST_GROUPS if g in groups)]
    ls = lock_summary()
    if ls:
        out.append(ls)
    out.append("")
    for g, ic, title in LIST_GROUPS:
        if g not in groups:
            continue
        out.append(f"{ic} {title} — {len(groups[g])}")
        if g in ("postponed", "closed") and not show_all:
            out.append("   " + ("`отложенные` покажет список" if g == "postponed" else "`список все` покажет их"))
            out.append("")
            continue
        out += [_hyp_line(n, g) for n in sorted(groups[g], key=_gain_key)[:cap]]
        if len(groups[g]) > cap:
            out.append(f"   … и ещё {len(groups[g]) - cap} (список все)")
        out.append("")
    out.append("Карточка с кнопками: гипотеза <имя> · отчёт <имя> · журнал <блок> <тип>")
    return "\n".join(out)


def cmd_usage() -> str:
    rows = al.usage_rows()
    ws = al.week_start_ts()
    wk = [r for r in rows if r["ts"] >= ws]
    cfg = al.limits_cfg()
    cl = [r for r in rows if r.get("backend") == "claude"]
    lines = ["💸 Расход",
             f"Claude: запросов {len(cl)}, всего ${sum(r.get('usd', 0) for r in cl):.2f}; за неделю ${al.week_usd():.2f} из ${cfg['usd_per_week']:.0f} (за один запуск не больше ${cfg['usd_per_run']:.0f})",
             f"Запросы автолаборатории (все модели): всего {sum(r.get('requests', 1) for r in rows)}, за неделю {sum(r.get('requests', 1) for r in wk)} из {cfg['weekly_llm_calls']}",
             "По назначению:"]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["purpose"], []).append(r)
    for pur, v in sorted(by.items()):
        lines.append(f"  • {pur}: запросов {sum(x.get('requests', 1) for x in v)}")
    exp = sum((al.meta(f.name[:-10]).get("experiment_requests") or 0) for f in al.DRAFTS.glob("*.meta.json"))
    ds = [r for r in rows if r.get("backend") == "deepseek"]
    if ds:
        lines.insert(2, f"DeepSeek: запросов {len(ds)}, токенов вход/выход {sum(r.get('in', 0) for r in ds)}/{sum(r.get('out', 0) for r in ds)} (деньги не считаются)")
    lines.append(f"Эксперименты автолаборатории: около {exp} запросов к шлюзу (нижняя оценка)")
    return "\n".join(lines)


def _safe_name(name: str) -> str | None:
    import re
    return name if re.fullmatch(r"[a-z][a-z0-9_]{2,60}", name or "") else None


def cmd_report(name: str) -> dict:
    name = _safe_name(name)
    if not name:
        return reply("Формат: отчёт <имя>")
    if not al.meta(name).get("result") and (al.EXP / "queue" / "done" / f"{name}.yaml").exists():
        _capture(al.cmd_verdict, name, False)   # builds the result in the metadata; the chat gets the file below, not a second copy from the outbox
    try:
        md, text = al.build_report_ru(name)
    except SystemExit as ex:
        return reply(str(ex))
    f = al.DRAFTS / f"{name}.report.md"
    f.write_text(md, encoding="utf-8")
    return reply(text, al.hyp_buttons(name), document=f)


def cmd_verdict(name: str) -> str:
    name = _safe_name(name)
    if not name:
        return "Формат: вердикт <имя>"
    if not al.meta(name).get("result") and (al.EXP / "queue" / "done" / f"{name}.yaml").exists():
        _capture(al.cmd_verdict, name, False)
    try:
        return al.build_report_ru(name)[1]
    except SystemExit as ex:
        return f"{name}: вердикта пока нет ({ex})"


# ------------------------------------------------------------------ spending commands: estimate, then confirmation
def _money_note(be: str, cfg: dict) -> str:
    if be.startswith("claude"):
        return f"не больше ${cfg['usd_per_run']:.0f} за запуск (Claude за неделю ${al.week_usd():.2f} из ${cfg['usd_per_week']:.0f})"
    if be == "deepseek":
        return "DeepSeek без денежного лимита (если не ответит, перейду на qwen и напишу об этом)"
    return "qwen через шлюз, деньги не тратятся"


def _est_cycle(n: int, with_exp: bool) -> str:
    cfg = al.limits_cfg()
    n = max(1, min(n, cfg["max_hyp_per_cycle"]))
    be, model = al.backend_desc("propose")
    s = (f"гипотез: {n}; генератор {be}/{model}; не больше {cfg['max_llm_calls_per_cycle']} запросов LLM за запуск; " + _money_note(be, cfg))
    if with_exp:
        s += f"\nДалее эксперименты: примерно {n * 300} запросов к шлюзу ({n * 300 / 10000:.1f}% трёхчасового окна, {n * 300 / 1e6:.2%} недели), около {n * 6} минут на каждую в очереди"
    else:
        s += "\nЭксперименты не запускаются: только идеи, проверка и симулятор."
    return s


def ask_cycle(uid, args: list[str], with_exp: bool) -> dict:
    n = 2
    if args and args[0].isdigit():
        n, args = int(args[0]), args[1:]
    mode = next((a.lower() for a in args if a.lower() in astat.MODES), None)
    args = [a for a in args if a.lower() not in astat.MODES]
    topic = " ".join(args)[:300]
    n = max(1, min(n, al.limits_cfg()["max_hyp_per_cycle"]))
    if al.STOP_FILE.exists():
        return reply("Автолаборатория остановлена (флаг AUTOLAB_STOP). Сначала `продолжить`.")
    code = issue(uid, "cycle" if with_exp else "propose", {"n": n, "topic": topic, "mode": mode})
    return reply(f"{'🔁 Цикл (идеи, проверка, эксперименты)' if with_exp else '💡 Идеи (без экспериментов)'}\n{_est_cycle(n, with_exp)}\nРежим: {al.MODE_RU[mode] if mode else 'поровну 🎯 accuracy и 💸 economy'}\nТема: {topic or 'по умолчанию'}\nКод {code}, действует 10 минут.",
                 confirm_buttons(code, "Запустить"))


def _conflict_reply(conf: dict, intro: str) -> dict:
    """The task types are held by an active hypothesis: say it plainly and put its five decisions under the message."""
    who = sorted(set(conf.values()))
    lines = [intro, "", "⛔ Типы задач заняты активными гипотезами (одновременно в работе одна гипотеза на тип):"]
    lines += [f"• {t}: {w} ({_state_ru(w)})" for t, w in sorted(conf.items())]
    lines += ["", "Сначала решите по активной гипотезе: принять, отклонить, ревью, повтор или отложить, чтобы новая с ней не конфликтовала. Кнопки ниже."]
    btns: list = []
    for w in who[:2]:
        btns += [[{"label": f"📄 {w}"[:40], "data": al._cb("hy", w) or "al:cmd:list"}]] + al.hyp_buttons(w)
    return reply("\n".join(lines), btns)


def _state_ru(name: str) -> str:
    base = al.family_base(name)
    dec = al.decision_of(base)
    if dec:
        return al.DECISION_RU.get(dec, dec) + (", устарела" if al.meta(base).get("stale") else "")
    if (al.EXP / "queue" / "running" / f"{name}.yaml").exists():
        return "идёт эксперимент"
    if (al.EXP / "queue" / "pending" / f"{name}.yaml").exists():
        return "в очереди"
    if al.meta(name).get("result"):
        return f"ждёт вашего решения, вердикт {al.journal_kind(name)}"
    return "ждёт вердикта"


def rerun_plan(name: str) -> dict:
    """How to collect more data for an experiment: the width of the paired interval falls as 1/sqrt(repeats), so the repeats needed follow from the
    current width against the target (a lower bound above 0, or half the threshold when the estimate is not positive). Returns n, a text and the facts."""
    import math
    import yaml
    cfg = al.limits_cfg()
    cap = cfg["max_rerun_repeats"]
    m, spec = al.meta(name), al.spec_of(name)
    r = int(spec.get("repeats") or 1)
    thr = m.get("threshold_used")
    best = None
    for a_, v in (m.get("result") or {}).items():
        d = v.get("points_diff_vs_first")
        if d and (best is None or d["mean"] > best[1]["mean"]):
            best = (a_, d)
    out: dict = {"repeats_now": r, "cap": cap, "threshold": thr}
    if not best:
        n = max(2, min(r, cap))
        out.update(n=n, text=f"Результата у {name} ещё нет, поэтому повтор просто запустит эксперимент заново: {n} повторов.")
    else:
        d, (lo, hi) = best[1]["mean"], best[1]["ci95"]
        hw = (hi - lo) / 2
        if d > 0:
            target, goal = 0.87 * d, f"нижняя граница интервала выше 0 (ширина ±{0.87 * d:.1f} при оценке {d:+.1f})"
        else:
            target, goal = (thr or hw) / 2, f"ширина интервала не больше половины порога (±{(thr or hw) / 2:.1f}), чтобы уверенно сказать, что эффекта нет"
        need = max(r + 1, math.ceil(r * (hw / target) ** 2)) if target > 0 and hw > 0 else r + 1
        n = min(max(need, 2), cap)
        why = ("интервал включает 0" if lo <= 0 else "интервал выше 0, но оценка ниже порога") if not (lo > 0 and thr is not None and d >= thr) else "результат уже accept"
        t = [f"Сейчас: {best[0]} {d:+.1f} очков [{lo:.1f}; {hi:.1f}], порог {thr}, повторов {r}. Проблема: {why}.",
             f"Ширина интервала падает как 1/√повторов (грубая оценка). Цель: {goal}.",
             f"Нужно примерно {need} повторов (сейчас {r})."]
        if need > cap:
            t.append(f"Лимит {cap} повторов за один запуск (max_rerun_repeats): предлагаю {n}; этого может не хватить, поднять лимит может только человек.")
        if thr is not None and d < thr:
            t.append(f"Точечная оценка ниже порога {thr}: повтор уточнит её, но accept возможен, только если истинный эффект на самом деле выше.")
        if lo > 0 and thr is not None and d >= thr:
            t.append("Эксперимент уже даёт accept: повтор нужен только как перепроверка.")
        out.update(n=n, need=need, text="\n".join(t))
    y = dict(spec)
    y["repeats"] = out["n"]
    out["est"] = al.exp_requests_estimate(y)
    return out


def _run_now(uid, action: str, args: dict, do, preview: str) -> dict:
    """Actions that spend only gateway / local Qwen tokens (free in the competition) need no confirmation: the plan is shown and the action runs at once."""
    journal(uid, "decision", what=action, args=json.dumps(args, ensure_ascii=False))
    out = do(uid, {"args": args})
    journal(uid, "result", what=action, out=out[:200])
    return reply(preview + "\n\n" + out)


def ask_rerun(uid, args: list[str]) -> dict:
    """`повтор <имя>` — the bot works out how many repeats to collect and what it costs; `повтор <имя> <n>` — a number of your own."""
    cfg = al.limits_cfg()
    if not args or len(args) > 2 or not _safe_name(args[0]) or (len(args) == 2 and not args[1].isdigit()):
        return reply("Формат: повтор <имя> (сам посчитаю, сколько данных нужно) или повтор <имя> <число повторов>")
    name = args[0]
    if not (al.DRAFTS / f"{name}.yaml").exists():
        return reply(f"Нет гипотезы {name}")
    base = al.family_base(name)
    conf = {t: w for t, w in al.type_conflicts(sorted(al.targets_of(name)), al.active_types()).items() if al.family_base(w) != base}
    if conf:
        return _conflict_reply(conf, f"🔁 Повтор {name} не запущен.")
    if al.meta(base).get("stale") and al.meta(name).get("branch"):
        return reply(f"⚠️ {name} устарела: после неё принята другая гипотеза того же типа, а у этой есть свой патч кода, сделанный на старой основе. Её нужно придумать заново: `идеи 1 <тема>` или кнопка «💡 Своя гипотеза» в ячейке.")
    plan = rerun_plan(name)
    reps = int(args[1]) if len(args) == 2 else plan["n"]
    if not 2 <= reps <= cfg["max_rerun_repeats"]:
        return reply(f"Повторов должно быть от 2 до {cfg['max_rerun_repeats']}")
    import yaml
    e = yaml.safe_load((al.DRAFTS / f"{name}.yaml").read_text(encoding="utf-8"))
    e["repeats"] = reps
    est = al.exp_requests_estimate(e)
    fresh = "\n♻️ Гипотеза устарела: повтор пересчитает всё с нуля на текущем коде (эталон и вариант в одном прогоне)." if al.meta(base).get("stale") else ""
    types = sorted(al.types_of(name))
    return _run_now(uid, "rerun", {"name": name, "repeats": reps}, do_rerun, f"🔁 Повтор {name}\n{plan['text'] if len(args) == 1 else 'Число повторов задано вручную.'}\n\n▶ Запуск: {reps} повторов как новый эксперимент (старые данные не суммируются), около {est} запросов к шлюзу ({est / 10000:.1f}% трёхчасового окна), без вызовов LLM."
                 f"\nГипотеза снова станет активной и займёт типы: {', '.join(types) or 'нет'}.{fresh}")


def ask_key(uid) -> dict:
    """`ключ`: provider, role models, balance and limit of the key without spending (OpenRouter GET /key); a real 1-token request only after the button."""
    text = al.key_status(False)
    if not al.read_key():
        return reply(text)
    code = issue(uid, "keyprobe", {})
    return reply(text + "\n\n▶ Реальный запрос: один вызов, ответ до 8 токенов, около 0,0001 $ (идёт в расход автолаборатории).\nКод " + code + ", действует 10 минут.", confirm_buttons(code, "🔑 Проверить запросом"))


def do_keyprobe(uid, e: dict) -> str:
    return al.key_status(True)


def _check_choose(name: str, sets: dict) -> dict:
    """No set named by the reviewer or by you: offer the sets that fit the task types of the hypothesis (a button each), then the rest as text."""
    import yaml
    try:
        used = (yaml.safe_load((al.DRAFTS / f"{name}.yaml").read_text(encoding="utf-8").split("# ---- written by lab.py")[0]) or {}).get("set")
    except Exception:  # noqa: BLE001
        used = None
    types = al.types_of(name)
    fit = [k for k, v in sets.items() if k != used and types and any(f"'{t}'" in v.split("tasks", 1)[-1].split("}", 1)[0] for t in types)]
    cells = set(al.targets_of(name))
    fit.sort(key=lambda k: (-sum(c in sets[k].split("loads cells:", 1)[-1] for c in cells) if "loads cells:" in sets[k] else 0,   # a narrow set that loads the target cell of the hypothesis first
                            len(re.findall(r"'\w+':", sets[k].split("}", 1)[0])), k))      # then the most specialised (fewest task types)
    btns = []
    for k in fit[:6]:
        d = f"al:cmd:check:{name}:{k}"
        if len(d.encode("utf-8")) <= 64:
            btns.append([{"label": f"🧪 {k}"[:40], "data": d}])
    head = f"Рецензия не назвала набор для проверки {name} (уже использован: {used or 'не набор'}). " + ("Выберите набор под типы задач " + ", ".join(sorted(types)) + " кнопкой или напишите: проверка <имя> <набор>." if btns else "Напишите: проверка <имя> <набор>.")
    return reply(head + "\nДоступные наборы:\n" + "\n".join(f"• {k}: {v}" for k, v in sets.items()), btns or None)


def ask_check(uid, args: list[str]) -> dict:
    """`проверка <имя> [набор]` — the same hypothesis on other data (a specialised or fresh set); without a set the reviewer's advice is used."""
    if args and args[0].lower() in ("ключа", "ключ"):
        return ask_key(uid)
    if not args or len(args) > 2 or not _safe_name(args[0]):
        return reply("Формат: проверка <имя> [набор] (набор по умолчанию — тот, что посоветовал рецензент)")
    name = args[0]
    if not (al.DRAFTS / f"{name}.yaml").exists():
        return reply(f"Нет гипотезы {name}")
    sets = al.check_sets()
    ds = args[1] if len(args) == 2 else al.meta(name).get("review_data_set")
    if not ds:
        return _check_choose(name, sets)
    if ds not in sets:
        return reply(f"Нет набора {ds} (holdout не используется). Доступные:\n" + "\n".join(f"• {k}: {v}" for k, v in sets.items()))
    base = al.family_base(name)
    conf = {t: w for t, w in al.type_conflicts(sorted(al.targets_of(name)), al.active_types()).items() if al.family_base(w) != base}
    if conf:
        return _conflict_reply(conf, f"🧪 Проверка {name} не запущена.")
    if al.meta(base).get("stale") and al.meta(name).get("branch"):
        return reply(f"⚠️ {name} устарела: после неё принята другая гипотеза того же типа, а у этой есть патч на старой основе. Придумайте её заново: `идеи 1 <тема>`.")
    import yaml
    e = yaml.safe_load((al.DRAFTS / f"{name}.yaml").read_text(encoding="utf-8").split("# ---- written by lab.py")[0])
    for k in ("tasks", "sources", "unanswerable", "rewards", "seed", "exclude", "exclude_stream", "adversarial"):
        e.pop(k, None)
    e["set"] = ds
    est = al.exp_requests_estimate(e)
    types = sorted(al.types_of(name))
    return _run_now(uid, "check", {"name": name, "set": ds}, do_check, f"🧪 Проверка {name} на других данных\nНабор: {ds} ({sets[ds]}). Эталон и вариант в одном прогоне, {e.get('repeats', 1)} повт., около {est} запросов к шлюзу ({est / 10000:.1f}% трёхчасового окна), без вызовов LLM."
                 f"\nГипотеза снова станет активной и займёт типы: {', '.join(types) or 'нет'}.")


def card_text(name: str) -> str:
    m = al.meta(name)
    base = al.family_base(name)
    cells = ", ".join(al.targets_of(name)) or "без ячейки"
    types = ", ".join(sorted(al.types_of(name))) or "-"
    g = al.hyp_gain(name)
    measured = bool(m.get("result"))
    out = [f"🧪 {name}", f"Состояние: {_state_ru(name)}", f"Режим: {al.MODE_RU.get(al.mode_of(name))}" + (" · ⚠️ затрагивает общие блоки: нужна проверка на другом режиме" if m.get("cross_check") else ""),
           f"Ячейки: {cells} · типы задач: {types}"]
    if g is not None:
        out.append(f"Выигрыш: {g:+.1f} очков за прогон ({'измерено' if measured else 'ожидается по рейтингу ячейки'})")
    out.append(_hyp_line(name, _hyp_group(name)).split("\n", 1)[1].replace("   ", "", 1) if "\n" in _hyp_line(name, _hyp_group(name)) else "")
    h = " ".join(str(al.spec_of(name).get("hypothesis") or "").split("[ячейка схемы")[0].split())
    if h:
        out.append("Идея: " + h[:260] + ("…" if len(h) > 260 else ""))
    dec = al.decision_of(base)
    if dec:
        out.append(f"Решение: {al.DECISION_RU.get(dec, dec)}" + (f" ({m.get('decision_note') or al.meta(base).get('decision_note')})" if al.meta(base).get("decision_note") else "")
                   + (f"\n⚠️ Устарела: после неё принята {al.meta(base).get('stale_by')}; перепроверьте повтором (с нуля)." if al.meta(base).get("stale") else ""))
    elif al.hyp_active(name):
        out.append("🔒 Держит типы задач: " + types + ". Пока вы не решите, эти типы заняты.")
    return "\n".join(x for x in out if x)


def cmd_card(args: list[str]) -> dict:
    name = _safe_name(args[0] if args else "")
    if not name or not (al.DRAFTS / f"{name}.yaml").exists():
        return reply("Формат: гипотеза <имя> (карточка с кнопками решений)" + ("" if not name else f"\nНет гипотезы {name}"))
    extra = [[{"label": "📄 Отчёт", "data": al._cb("report", name) or "al:cmd:list"}]] if al.meta(name).get("result") else []
    return reply(card_text(name), al.hyp_buttons(name) + extra)


def _cancel_pending(base: str) -> list[str]:
    """A hypothesis closed by a decision must not start later: its waiting experiment goes to queue/cancelled (a running one cannot be stopped from here)."""
    out = []
    pd = al.EXP / "queue" / "pending"
    for f in sorted(pd.glob("*.yaml")) if pd.exists() else []:
        if al.family_base(f.stem) == base:
            dst = al.EXP / "queue" / "cancelled"
            dst.mkdir(parents=True, exist_ok=True)
            f.replace(dst / f.name)
            out.append(f.stem)
    return out


def decide(uid, args: list[str], decision: str) -> dict:
    """`отклонить <имя>` / `отложить <имя>`: no money is spent and nothing leaves the host, so no confirmation."""
    name = _safe_name(args[0] if args else "")
    word = {"rejected": "отклонить", "postponed": "отложить"}[decision]
    if not name or not (al.DRAFTS / f"{name}.yaml").exists():
        return reply(f"Формат: {word} <имя>" + ("" if not name else f"\nНет гипотезы {name}"))
    base = al.family_base(name)
    cur = al.decision_of(base)
    if cur == "accepted":
        return reply(f"{name} уже принята: изменить решение этой кнопкой нельзя.")
    if cur == decision:
        return reply(f"{name} уже {al.DECISION_RU[decision]}.", [[{"label": "📄 Карточка", "data": al._cb("hy", name) or "al:cmd:list"}]])
    cancelled = _cancel_pending(base)
    running = any(al.family_base(f.stem) == base for f in (al.EXP / "queue" / "running").glob("*.yaml"))
    types = ", ".join(sorted(al.types_of(name))) or "-"
    al.set_decision(base, decision, "", str(uid))
    journal(uid, "decision", what=decision, name=base)
    if decision == "rejected":
        txt = f"❌ {base} отклонена. Типы задач ({types}) свободны."
    else:
        txt = (f"⏸ {base} отложена: она не занимает типы задач ({types}). Найти её можно командой `отложенные`; она устареет, как только будет принята любая гипотеза этих типов, "
               "и тогда её нужно перепроверить с нуля. Вернуть в работу: 🔁 Повтор или 🔍 Ревью в карточке.")
    if cancelled:
        txt += f"\n🗑 Из очереди убрано: {', '.join(cancelled)}."
    if running:
        txt += "\n⚠️ Эксперимент уже идёт и закончится, но вердикт ничего не заблокирует."
    return reply(txt, [[{"label": "📄 Карточка", "data": al._cb("hy", base) or "al:cmd:list"}, {"label": "📋 Гипотезы", "data": "al:cmd:list"}]])


# a hypothesis of a human: `своя <блок>/<тип> <текст>`, or the button under a cell and then the next message of that user in that chat is the text
CAPTURE_TTL = 600


def _capture_file() -> Path:
    return al.CFG_DIR / "tg_mine.json"


def _captures() -> dict:
    try:
        d = json.loads(_capture_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in d.items() if time.time() - v.get("ts", 0) < CAPTURE_TTL}


def _save_captures(d: dict) -> None:
    al.CFG_DIR.mkdir(parents=True, exist_ok=True)
    _capture_file().write_text(json.dumps(d), encoding="utf-8")


def has_capture(uid, chat) -> bool:
    return f"{uid}:{chat}" in _captures()


def _drop_capture(uid, chat) -> None:
    d = _captures()
    if d.pop(f"{uid}:{chat}", None) is not None or _capture_file().exists():
        _save_captures(d)


def handle_capture(uid, chat, text: str, chat_type: str = "private") -> dict | None:
    """The message of a user who pressed «Своя гипотеза»: it is the text of the hypothesis. None when this user is not composing one."""
    if not allowed(uid, chat, chat_type):
        return None
    d = _captures()
    st = d.get(f"{uid}:{chat}")
    if not st:
        return None
    t = (text or "").strip()
    if t.lower() in ("отмена", "cancel", "/cancel"):
        _drop_capture(uid, chat)
        return reply("Хорошо, своя гипотеза отменена.")
    if len(t) < 15:
        return reply("Слишком коротко: опишите, что именно менять и почему это должно улучшить результат (от 15 символов), или напишите `отмена`.")
    _drop_capture(uid, chat)
    return ask_mine(uid, [st["cell"], t[:1500]], chat, chat_type)


def ask_mine(uid, args: list[str], chat=None, chat_type: str = "private") -> dict:
    root = _status_root()
    if not astat.index(root):
        return _no_status()
    ref = astat.parse_ref(args[0]) if args else None
    rest = args[1:]
    if not ref and len(args) > 1:
        ref, rest = astat.parse_ref(" ".join(args[:2])), args[2:]
    if not ref:
        return reply("Формат: своя <блок>/<тип> <текст гипотезы> (или кнопка «💡 Своя гипотеза» в `ячейка <блок> <тип>`)")
    cell = f"{ref[0]}/{ref[1]}"
    ok, msg = astat.check_target(root, cell)
    if not ok:
        return reply(f"💡 Своя гипотеза: {msg}")
    if al.STOP_FILE.exists():
        return reply("Автолаборатория остановлена (флаг AUTOLAB_STOP). Сначала `продолжить`.")
    conf = al.type_conflicts([cell], al.active_types())
    if conf:
        return _conflict_reply(conf, f"💡 Своя гипотеза для ячейки {cell}.")
    text = " ".join(rest).strip()
    if not text:
        d = _captures()
        d[f"{uid}:{chat or uid}"] = {"cell": cell, "ts": time.time()}
        _save_captures(d)
        return reply(f"💡 Своя гипотеза для ячейки {cell}. Типы задач свободны.\nНапишите ОДНИМ следующим сообщением, что именно изменить и почему это должно помочь "
                     "(переключатель, значение, набор задач). Она встанет первой в очереди. Отмена: `отмена` (10 минут).")
    be, model = al.backend_desc("propose")
    types = ", ".join(sorted(astat.types_of_cells([cell], astat.index_types(root))))
    est = al.exp_requests_estimate({"set": "math16", "budget": 30000, "arms": {"a": 1, "b": 1}, "repeats": 1})
    code = issue(uid, "mine", {"cell": cell, "text": text, "chat": str(chat or uid)})
    return reply(f"💡 Своя гипотеза для ячейки {cell}\nТекст: «{text[:400]}»\n\nЧто будет: генератор {be}/{model} оформит её в эксперимент (один запрос, {_money_note(be, al.limits_cfg())}); "
                 f"отсев симулятором для неё отключён; эксперимент встанет ПЕРВЫМ в очереди и займёт типы задач: {types}. Эксперимент потратит порядка {est}+ запросов шлюза.\nКод {code}, действует 10 минут.",
                 confirm_buttons(code, "Оформить и запустить"))


def do_mine(uid, e: dict) -> str:
    a = e["args"]
    return _spawn(["mine", "--cell", a["cell"], "--text", a["text"], "--by", str(uid), f"--chat={a['chat']}"], "mine")


def cmd_postponed() -> dict:
    names = [f.name[:-10] for f in al.DRAFTS.glob("*.meta.json") if al.meta(f.name[:-10]).get("decision") == "postponed"]
    names.sort(key=lambda n: -(al.hyp_gain(n) if al.hyp_gain(n) is not None else -1e9))
    if not names:
        return reply("⏸ Отложенных гипотез нет. Отложить можно кнопкой ⏸ в карточке гипотезы.")
    out = [f"⏸ Отложенные гипотезы — {len(names)} (в работу не входят, типы не занимают)", ""]
    for n in names[:12]:
        m = al.meta(n)
        g = al.hyp_gain(n)
        out.append(f"• {n}\n   ячейки: {', '.join(al.targets_of(n)) or '-'} · выигрыш: {('%+.1f' % g) if g is not None else '-'}"
                   + (f"\n   ⚠️ устарела: принята {m.get('stale_by')}, перепроверить с нуля" if m.get("stale") else "\n   ✅ актуальна"))
    if len(names) > 12:
        out.append(f"… и ещё {len(names) - 12}")
    out.append("\nКарточка: гипотеза <имя>. Из неё можно повторить (🔁) или отдать на ревью (🔍).")
    return reply("\n".join(out), [[{"label": f"📄 {n}"[:40], "data": al._cb("hy", n) or "al:cmd:list"}] for n in names[:8]])


def cmd_autopilot_ru() -> str:
    cfg = al.limits_cfg()
    on = al.autopilot_flag().exists()
    st = al._autopilot_state()
    q = {k: len(list((al.EXP / "queue" / k).glob("*.yaml"))) for k in ("pending", "running")}
    return (f"🤖 Автопилот: {'включён ✅' if on else 'выключен'}\n"
            f"Очередь: идёт {q['running']}, ждёт {q['pending']} (держу не меньше {cfg['autopilot_target_depth']})\n"
            f"Сегодня поставлено гипотез: {len(st['names_today'])}" + (f" из {cfg['autopilot_max_per_day']}" if cfg['autopilot_max_per_day'] else " (дневного лимита нет)") + "\n"
            f"Как работает: раз в 10 минут, если очередь короче {cfg['autopilot_target_depth']} и шлюз свободен, генератор придумывает до {cfg['autopilot_batch']} гипотез "
            "по кругу тем, они проходят проверку и симулятор и встают в очередь. Вердикты приходят в группу сами.\n"
            "Команды: автопилот вкл (спросит подтверждение), автопилот выкл")


def ask_autopilot_on(uid, args: list[str]) -> dict:
    cfg = al.limits_cfg()
    if al.STOP_FILE.exists():
        return reply("Автолаборатория остановлена (флаг AUTOLAB_STOP). Сначала `продолжить`.")
    be, model = al.backend_desc("propose")
    per_day = (cfg["autopilot_max_per_day"] or 8) * 300      # no daily limit: an estimate for eight
    code = issue(uid, "autopilot_on", {})
    return reply(f"🤖 Включить автопилот?\nГенератор {be}/{model}; до {cfg['autopilot_batch']} гипотез за раз, {('не больше ' + str(cfg['autopilot_max_per_day']) + ' в сутки') if cfg['autopilot_max_per_day'] else 'без дневного лимита'} "
                 f"(оценка: около {per_day} запросов к шлюзу в сутки, {per_day / 1e6:.2%} недельного лимита), очередь держу не короче {cfg['autopilot_target_depth']}; "
                 f"останавливается при остановке автолаборатории, нехватке места в шлюзе и по команде `автопилот выкл`. {_money_note(be, cfg)}.\nКод {code}, действует 10 минут.",
                 confirm_buttons(code, "Включить автопилот"))


def do_autopilot_on(uid, e: dict) -> str:
    al.CFG_DIR.mkdir(parents=True, exist_ok=True)
    al.autopilot_flag().write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" on by {uid}\n")
    return "✅ Автопилот включён. Первая проверка очереди в течение 10 минут."


def ask_chief_on(uid, args: list[str]) -> dict:
    import autolab_chief as ch
    cfg = al.limits_cfg()
    if al.STOP_FILE.exists():
        return reply("Автолаборатория остановлена (флаг AUTOLAB_STOP). Сначала `продолжить`.")
    be, model = al.backend_desc("chief")
    code = issue(uid, "chief_on", {})
    mode = ("dry-run: шеф только планирует и пишет в чат, ничего не запускает (`chief_dry_run: true`)" if ch.dry_run()
            else "БОЕВОЙ режим: шеф сам ставит гипотезы, проверки, ревью и принимает в ветку auto/<имя> (в main только после вашего `принять`)")
    return reply(f"🧭 Включить шефа?\nМодель {be}/{model}; тик раз в 30 минут (около 0,1-0,3 $ за тик), лимит {cfg['usd_per_day']} $ в сутки и {cfg['usd_per_week']} $ в неделю на всё Claude автолаборатории. "
                 f"{mode}.\nПока шеф включён, автопилот ничего не предлагает (один генератор). Выключатель: `шеф выкл` или флаг остановки.\nКод {code}, действует 10 минут.",
                 confirm_buttons(code, "Включить шефа"))


def do_chief_on(uid, e: dict) -> str:
    import autolab_chief as ch
    al.CFG_DIR.mkdir(parents=True, exist_ok=True)
    al.chief_flag().write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" on by {uid}\n")
    return "✅ Шеф включён" + (" в режиме dry-run (ничего не запускает)." if ch.dry_run() else " в боевом режиме.") + " Первый тик в течение 30 минут."


def _spawn_chief_ask(chat, question: str) -> str:
    log = al.DRAFTS / f"tg_chiefask_{time.strftime('%Y%m%d_%H%M%S')}.log"
    al.DRAFTS.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        p = subprocess.Popen([sys.executable, str(HERE / "autolab_chief.py"), "ask", f"--chat={chat or ''}", question], stdout=out, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(HERE))
    return f"pid {p.pid}"


def cmd_chief_ru(args: list[str], uid=None, chat=None) -> dict:
    import autolab_chief as ch
    sub = (args[0].lower() if args else "")
    if sub in ("идея", "idea"):
        text = " ".join(args[1:]).strip()
        if len(text) < 15:
            return reply("Формат: `шеф идея <текст>` (хотя бы пара предложений: что попробовать, где и чего ждёте).")
        n = ch.add_idea(text, uid)
        journal(uid, "chief_idea", id=n, text=text[:200])
        return reply(f"💡 Идея #{n} записана. Шеф увидит её на ближайшем тике (в течение ~5 минут, если он включён) и ответит: возьмёт в работу, отложит или откажется с причиной. Список: `шеф идеи`."
                     + ("" if ch.is_on() else "\n⚠️ Шеф сейчас выключен: идея ждёт его включения."))
    if sub in ("идеи", "ideas"):
        return reply(ch.ideas_view())
    if sub in ("вопрос", "ask", "q"):
        text = " ".join(args[1:]).strip()
        if len(text) < 5:
            return reply("Формат: `шеф вопрос <текст>`. Шеф ответит по тому, что знает сейчас (состояние лаборатории, свой план, заметки, ваши идеи), отдельным сообщением в течение минуты.")
        journal(uid, "chief_question", text=text[:200])
        _spawn_chief_ask(chat or uid, text)
        return reply("🧭 Вопрос передан шефу, ответ придёт отдельным сообщением (обычно до минуты; один вызов модели, ~0,1 $).")
    if sub in ("вкл", "on"):
        return None
    if sub in ("план", "plan"):
        return reply(ch.plan_text())
    if sub in ("дайджест", "digest"):
        return reply(ch.digest_text())
    return reply(ch.status_text(), [[{"label": "Выключить" if ch.is_on() else "Включить", "data": "al:cmd:chief:" + ("выкл" if ch.is_on() else "вкл")},
                                     {"label": "📋 План", "data": "al:cmd:chief:план"}]])


def cmd_localdata_ru() -> str:
    on = al.localdata_flag().exists()
    pids = al._local_stream_pids()
    t = al._gpu_temp()
    return (f"🖥 Сбор данных на локальной Qwen хоста: {'включён ✅' if on else 'выключен'}\n"
            f"Поток: {'идёт (pid ' + ', '.join(pids) + ')' if pids else 'не запущен'}\n"
            f"GPU: {str(t) + ' °C' if t is not None else 'нет данных'}\n"
            "Что это: обычный поток сбора `stream_run.py --backend local` на модели qwen3.6-35b-a3b (llama-server, GGUF Q4) без запросов к шлюзу; "
            "результаты копятся отдельно от шлюзовых. Автопилот-тик раз в 10 минут следит, чтобы поток работал.\n"
            "Команды: локально вкл (спросит подтверждение), локально выкл")


def ask_localdata_on(uid, args: list[str]) -> dict:
    cfg = al.limits_cfg()
    return _run_now(uid, "localdata_on", {}, do_localdata_on, f"🖥 Включить сбор данных на локальной Qwen хоста?\nЗагрузит видеокарту (llama-server держит 3 параллельных запроса), запросов к шлюзу и денег не тратит; "
                 f"не стартует при температуре выше {cfg['local_data_max_temp_c']} °C, при стоп-флаге автолаборатории и пока идёт локальный эксперимент. "
                 f"Выключается командой `локально выкл` (поток закончит текущую порцию).")


def do_localdata_on(uid, e: dict) -> str:
    al.CFG_DIR.mkdir(parents=True, exist_ok=True)
    al.localdata_flag().write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" on by {uid}\n")
    (al.LAB / "STREAM_STOP_local").unlink(missing_ok=True)   # the human has just asked for the local stream: the old stop file is superseded
    return "✅ Сбор на локальной Qwen включён. Поток стартует при ближайшей проверке (до 10 минут)."


# ------------------------------------------------------------------ scheme status (harness/status/): read-only views + binding a hypothesis to a cell
def _status_root() -> Path:
    return al.status_root()


def _parts_reply(parts: list[str], buttons: list | None = None) -> dict:
    """Views come as messages of at most 4096 characters; the first goes into `text`, the others into `more` (the transport sends them in order)."""
    return {"text": parts[0], "buttons": buttons or [], "document": "", "more": parts[1:]}


def _no_status() -> dict:
    return reply("Схема статусов не найдена на хосте (папка status/ с index.json). Её нужно выложить в ~/harness-lab/status (см. SCRIPTS.md §7.11).")


def status_blocks() -> dict:
    root = _status_root()
    idx = astat.index(root)
    if not idx:
        return _no_status()
    rows, row = [], []
    for b, cells in sorted(astat._by_block(idx).items(), key=lambda kv: -(astat.readiness(list(kv[1].values())) or 0)):
        row.append({"label": f"{b} {astat.readiness(list(cells.values()))}%", "data": f"al:cmd:block:{b}"})
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return reply("🧱 Блоки схемы (от самых готовых). Нажмите блок, чтобы увидеть его готовность по типам задач.\nПроцент = готовность блока; общая картина: схема.", rows)


def status_types() -> dict:
    root = _status_root()
    idx = astat.index(root)
    if not idx:
        return _no_status()
    btn = [{"label": f"{astat.TYPE_ICON.get(t, t)} {astat.readiness([e for e in idx['entries'] if e['type'] == t])}%", "data": f"al:cmd:type:{t}"} for t in idx["types"]]
    return reply("🗂 Типы задач. Нажмите тип, чтобы увидеть все блоки для него.", [btn[:2], btn[2:]])


def status_block(args: list[str]) -> dict:
    import status_views as sv
    root = _status_root()
    idx = astat.index(root)
    if not idx:
        return _no_status()
    if not args:
        return status_blocks()
    b = args[0].lower()
    row = lambda ts: [{"label": t, "data": f"al:cmd:cell:{b}/{t}"} for t in ts]
    ts = idx["types"]
    btn = [row(ts[:2]), row(ts[2:])] if b in idx["blocks"] and ts[2:] else ([row(ts)] if b in idx["blocks"] else [])
    return _parts_reply(astat.block_text(root, b), btn)


def status_type(args: list[str]) -> dict:
    import status_views as sv
    root = _status_root()
    idx = astat.index(root)
    if not idx:
        return _no_status()
    if not args:
        return status_types()
    return _parts_reply(astat.type_text(root, astat.TYPE_ALIASES.get(args[0].lower(), args[0].lower())))


def status_cell(args: list[str]) -> dict:
    import status_views as sv
    import status_check as sc
    root = _status_root()
    if not astat.index(root):
        return _no_status()
    r = astat.parse_ref(" ".join(args))
    if not r:
        return reply("Какая ячейка? Пример: ячейка batcher math (или batcher/math; для общих свойств блока: ячейка batcher общее)")
    f = sc.cell_path(root, *r)
    if not f.exists():
        return _parts_reply(sv.cell_view(r[0], r[1], root))
    nice, whole = astat.cell_text(root, *r)
    # the md file of the cell from the host always goes along (full text, exact copy), the readable text is the message
    out = al.CFG_DIR / "tg_docs"
    out.mkdir(parents=True, exist_ok=True)
    doc = out / f"{r[0]}_{r[1]}.md"
    doc.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    if whole and len(nice) <= 3800:
        return reply(nice, [[{"label": "📜 Журнал гипотез", "data": f"al:cmd:journal:{r[0]}/{r[1]}"}, {"label": "💡 Своя гипотеза", "data": f"al:cmd:mine:{r[0]}/{r[1]}"}]], document=doc)
    cut = nice[:900].rsplit("\n", 1)[0]
    return reply(cut + f"\n…\n📎 полный текст файлом {doc.name}", [[{"label": "📜 Журнал гипотез", "data": f"al:cmd:journal:{r[0]}/{r[1]}"}, {"label": "💡 Своя гипотеза", "data": f"al:cmd:mine:{r[0]}/{r[1]}"}]], document=doc)


def journal_recent(root, n: int = 12) -> dict:
    """`журнал`: the newest rows of the hypothesis journals of all cells: everything that is not active lives here."""
    rows = []
    for e in astat.index(root)["entries"]:
        ref = f"{e['block']}/{e['type']}"
        for r in astat.journal_rows(root, ref, 100000):
            rows.append((r.get("date") or "", ref, r))
    rows.sort(key=lambda x: x[0], reverse=True)
    if not rows:
        return reply("📜 Журнал гипотез пока пуст.")
    ic = {"accept": "✅", "clarify": "⚖️", "reject": "❌", "failed": "⛔"}
    out = [f"📜 Журнал гипотез: последние {min(n, len(rows))} из {len(rows)} записей (все ячейки)", ""]
    for d, ref, r in rows[:n]:
        out.append(f"{ic.get(r['verdict'], '•')} {d} · {ref} · {r['name']}\n   {r['verdict']}: {str(r['result'])[:110]}")
    out += ["", "По ячейке: журнал <блок> <тип> [N] · журнал 30 — больше записей"]
    refs = []
    for d, ref, r in rows[:n]:
        if ref not in refs:
            refs.append(ref)
    return reply("\n".join(out), [[{"label": f"📜 {ref}"[:40], "data": f"al:cmd:journal:{ref}"}] for ref in refs[:5]])


def status_journal(args: list[str]) -> dict:
    """`журнал <блок> <тип> [N]`: the last N rows of the hypothesis journal of a cell (what was tried, result, verdict); the whole journal as a file when longer."""
    import status_views as sv
    root = _status_root()
    if not astat.index(root):
        return _no_status()
    if not args or (len(args) == 1 and args[0].isdigit()):
        return journal_recent(root, int(args[0]) if args else 12)
    n = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 5
    r = astat.parse_ref(" ".join(a for a in args if not a.isdigit() or a is args[0]))
    if not r:
        return reply("Какая ячейка? Пример: журнал batcher math (или batcher/math, или журнал batcher общее; последним числом можно задать, сколько записей показать)")
    v = sv.journal_view(r[0], r[1], n, root)
    parts = v["messages"]
    doc = None
    if v.get("document"):
        out = al.CFG_DIR / "tg_docs"
        out.mkdir(parents=True, exist_ok=True)
        doc = out / v["filename"]
        doc.write_text(v["document"], encoding="utf-8")
    rep = _parts_reply(parts, [[{"label": "📄 Ячейка", "data": f"al:cmd:cell:{r[0]}/{r[1]}"}]])
    if doc:
        rep["text"], rep["document"] = parts[0] + "\n📎 весь журнал файлом " + doc.name, str(doc)
    return rep


def status_scheme() -> dict:
    import status_views as sv
    if not astat.index(_status_root()):
        return _no_status()
    return _parts_reply(astat.scheme_text(_status_root()))


def status_priorities(args: list[str] | None = None) -> dict:
    """`приоритеты [accuracy|economy]`: the common rating, or the rating of one goal mode (buttons switch between them)."""
    if not astat.index(_status_root()):
        return _no_status()
    mode = next((a.lower() for a in (args or []) if a.lower() in astat.MODES), None)
    btns = [[{"label": f"{astat.MODE_ICON[m]} {m}", "data": f"al:cmd:priorities:{m}"} for m in astat.MODES] + ([{"label": "📊 Общий", "data": "al:cmd:priorities"}] if mode else [])]
    return _parts_reply(astat.priority_text(_status_root(), mode=mode), btns)


def bind_cell(uid, args: list[str]) -> dict:
    """`привязать <имя> <блок>/<тип> [...]`: attach target cells to a hypothesis made before the scheme existed (accept needs them)."""
    name = _safe_name(args[0]) if args else ""
    if not name or len(args) < 2:
        return reply("usage: привязать <имя> <блок>/<тип> (до 3 ячеек), например: привязать ab_batch8_mixed_accuracy batcher/math")
    if not al.meta(name):
        return reply(f"Нет гипотезы {name}.")
    root = _status_root()
    refs = [r for r in re.split(r"[,\s]+", " ".join(args[1:])) if r]
    if not astat.index(root) or not refs or len(refs) > astat.MAX_TARGETS:
        return reply("Нужны 1-3 ячейки вида блок/тип; схема должна быть на хосте.")
    cells = []
    for r in refs:
        ok, msg = astat.check_target(root, r)
        if not ok:
            return reply("Не привязано: " + msg)
        cells.append(msg)
    al.set_meta(name, target_cells=cells, why_cell=f"привязано вручную пользователем {uid}")
    journal(uid, "decision", what="bind", name=name, cells=cells)
    done = al.record_journal(name)
    return reply(f"✅ {name} привязана к ячейкам: {', '.join(cells)}. Теперь `принять {name}` обновит их статус."
                 + (f"\n📜 В журнал ячеек записано текущее состояние: {al.journal_kind(name)}." if done else ""))


def ask_review(uid, args: list[str], chat=None) -> dict:
    name = _safe_name(args[0] if args else "")
    if not name:
        return reply("Формат: рецензия <имя>")
    if not al.meta(name).get("result"):
        return reply(f"У {name} ещё нет вердикта: рецензенту нужен отчёт")
    if al.STOP_FILE.exists():
        return reply("Автолаборатория остановлена (флаг AUTOLAB_STOP). Сначала `продолжить`.")
    cfg = al.limits_cfg()
    be, model = al.backend_desc("review")
    code = issue(uid, "review", {"name": name, "chat": str(chat or uid)})
    return reply(f"🧐 Рецензия {name}: один запрос к модели, рецензент {be}/{model} (только чтение, видит гипотезу, патч и отчёт), "
                 f"{_money_note(be, cfg)}.\nКод {code}, действует 10 минут.",
                 confirm_buttons(code, "Запустить рецензию"))


def _spawn(argv: list[str], tag: str) -> str:
    log = al.DRAFTS / f"tg_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    al.DRAFTS.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        p = subprocess.Popen([sys.executable, str(HERE / "autolab.py"), *argv], stdout=out, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(HERE))
    return f"started pid {p.pid}, log {log.name}"


def do_cycle(uid, e: dict, with_exp: bool) -> str:
    a = e["args"]
    argv = (["cycle", "--n", str(a["n"]), "--wait"] if with_exp else ["propose", "--n", str(a["n"])]) + (["--topic", a["topic"]] if a["topic"] else []) + (["--mode", a["mode"]] if a.get("mode") else [])
    return _spawn(argv, "cycle" if with_exp else "propose")


def do_review(uid, e: dict) -> str:
    return _spawn(["review", e["args"]["name"], f"--notify={e['args'].get('chat') or uid}"], "review")   # '=': a group id starts with '-'


def do_check(uid, e: dict) -> str:
    """The hypothesis as a new experiment `<name>_fresh[N]` on another set (old data are not mixed in)."""
    import yaml
    name, ds = e["args"]["name"], e["args"]["set"]
    src = al.DRAFTS / f"{name}.yaml"
    base = al.family_base(name)
    conf = {t: w for t, w in al.type_conflicts(sorted(al.targets_of(name)), al.active_types()).items() if al.family_base(w) != base}
    if conf:
        return "not started: типы задач заняты активной гипотезой " + ", ".join(sorted(set(conf.values()))) + " (сначала решите по ней)"
    if ds not in al.check_sets():
        return f"not started: нет набора {ds}"
    if al.decision_of(base):
        al.set_meta(base, decision=None, stale=False, decision_note="")
    new = f"{base}_fresh"
    i = 2
    while (al.DRAFTS / f"{new}.yaml").exists() or (al.EXP / "queue" / "done" / f"{new}.yaml").exists():
        new, i = f"{base}_fresh{i}", i + 1
    y = yaml.safe_load(src.read_text(encoding="utf-8").split("# ---- written by lab.py")[0])
    for k in ("tasks", "sources", "unanswerable", "rewards", "seed", "exclude", "exclude_stream", "adversarial"):
        y.pop(k, None)
    y["name"], y["set"] = new, ds
    if e["args"].get("budget"):          # the chief checks a second budget
        y["budget"] = int(e["args"]["budget"])
    y["hypothesis"] = f"(check of {name} on set {ds}) " + str(y.get("hypothesis", ""))
    (al.DRAFTS / f"{new}.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    m = dict(al.meta(name))
    for k in ("verdict", "result", "queued_at", "experiment_requests", "stage_note", "decision", "decided_at", "decision_note", "decided_by", "stale", "stale_by", "stale_at", "review_file", "review_recommendation", "review_data_set", "review_data_text"):
        m.pop(k, None)
    m.update(stage="applied" if m.get("branch") else "proposed", sim_verdict="pass", check_of=name, check_set=ds, **({"check_budget": int(e["args"]["budget"])} if e["args"].get("budget") else {}))
    (al.DRAFTS / f"{new}.meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    try:
        al.cmd_submit(new)
    except SystemExit as ex:
        return f"not submitted: {ex}"
    _spawn(["await", new], "await")
    return f"✅ {new} поставлен в очередь (набор {ds}). Когда закончится, пришлю вердикт и отчёт в группу сам."


def do_rerun(uid, e: dict) -> str:
    import yaml
    name, reps = e["args"]["name"], e["args"]["repeats"]
    src = al.DRAFTS / f"{name}.yaml"
    base = al.family_base(name)
    conf = {t: w for t, w in al.type_conflicts(sorted(al.targets_of(name)), al.active_types()).items() if al.family_base(w) != base}
    if conf:
        return "not started: типы задач заняты активной гипотезой " + ", ".join(sorted(set(conf.values()))) + " (сначала решите по ней)"
    if al.decision_of(base):
        al.set_meta(base, decision=None, stale=False, decision_note="")      # taken back into work
    new = f"{name}_r{reps}"
    i = 2
    while (al.DRAFTS / f"{new}.yaml").exists() or (al.EXP / "queue" / "done" / f"{new}.yaml").exists():
        new, i = f"{name}_r{reps}_{i}", i + 1
    y = yaml.safe_load(src.read_text(encoding="utf-8"))
    y["name"], y["repeats"] = new, reps
    y["hypothesis"] = f"(rerun of {name} with {reps} repeats) " + str(y.get("hypothesis", ""))
    (al.DRAFTS / f"{new}.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    m = dict(al.meta(name))
    for k in ("verdict", "result", "queued_at", "experiment_requests", "stage_note", "decision", "decided_at", "decision_note", "decided_by", "stale", "stale_by", "stale_at"):
        m.pop(k, None)
    m.update(stage="applied" if m.get("branch") else "proposed", sim_verdict="pass", rerun_of=name)   # the simulator already had its say on the original
    (al.DRAFTS / f"{new}.meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    try:
        al.cmd_submit(new)
    except SystemExit as ex:
        return f"not submitted: {ex}"
    _spawn(["await", new], "await")   # nobody else waits for a rerun: this process sends the verdict to the chat when the experiment is done
    return f"✅ {new} поставлен в очередь ({reps} повторов). Когда закончится, пришлю вердикт и отчёт в группу сам."


# ------------------------------------------------------------------ accept: merge in the host copy, then push master (one button)
def _git(repo: Path, *a, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=autolab", "-c", "user.email=noreply@anthropic.com", *a], capture_output=True, text=True, env=env)


def deploy_key() -> Path:
    return Path(os.path.expanduser(os.environ.get("AUTOLAB_DEPLOY_KEY") or al.limits_cfg()["deploy_key_file"]))


def accept_preflight(name: str, against: str = "HEAD") -> tuple[dict | None, str]:
    """`against`: the base the patch of the branch is measured from (the chief passes the stack branch for a hypothesis built on the stack)."""
    name = _safe_name(name)
    if not name:
        return None, "usage: accept <name> (принять <имя>)"
    m = al.meta(name)
    for k, word in (("running", "идёт"), ("pending", "ждёт в очереди")):
        if (al.EXP / "queue" / k / f"{name}.yaml").exists():
            return None, f"{name}: эксперимент ещё {word}, вердикта нет, принимать пока нечего. Дождитесь результата (`статус`, `очередь`)."
    root = al.status_root()
    if not astat.index(root):
        return None, "схема статусов (status/index.json) не найдена на хосте, обновлять нечего; см. SCRIPTS.md §7.11"
    cells = al.targets_of(name)
    if not cells:
        return None, (f"гипотеза {name} не привязана ни к одной ячейке схемы (блок × тип), а принятие обновляет статус ячейки. "
                      f"Привяжите: `привязать {name} <блок>/<тип>` (например batcher/math); список ячеек: `схема`, `приоритеты`.")
    for c in cells:
        ok, msg = astat.check_target(root, c)
        if not ok:
            return None, msg + f" (привязка: `привязать {name} <блок>/<тип>`)"
    plan = al.status_plan(name, accept=True)
    try:
        astat.dry_run(root, plan)
    except ValueError as ex:
        return None, f"статус ячеек не обновить: {ex}"
    if not m.get("branch"):
        if al.verdict_kind(name) != "accept":
            return None, f"{name}: кода для слияния нет, а вердикт лаборатории {al.verdict_kind(name)}, не accept: принимать нечего (результат уже записан в историю ячеек {', '.join(cells)})"
        return {"name": name, "no_code": True, "branch": "", "commit": "", "files": [], "stat": "кода нет: только запись результата в схему статусов", "verdict": m.get("verdict"),
                "review": m.get("review_recommendation"), "cells": cells, "plan": plan}, ""
    repo = al.repo_root()
    if m.get("stack_at_apply") and against == "HEAD":
        return None, f"{name}: ветка собрана поверх стека принятых гипотез ({al.STACK_BRANCH}, на момент сборки: {m['stack_at_apply']}), отдельно в main её не влить: принимается весь стек (`стек`, `принять стек`)"
    if repo.name != "autolab_repo":
        return None, "accept works only in the host copy ~/harness-lab/autolab_repo (this looks like a dev checkout)"
    if _git(repo, "status", "--porcelain").stdout.strip():
        return None, "the host copy has uncommitted changes"
    head = _git(repo, "rev-parse", "--short", m["branch"]).stdout.strip()
    if not head or head != m.get("commit"):
        return None, f"branch head {head or '?'} differs from the commit autolab recorded ({m.get('commit')}): refusing"
    base = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if base != "main":
        return None, f"the host copy is on '{base}', not main"
    names = _git(repo, "diff", "--name-only", f"{against}...{m['branch']}").stdout.split()
    diff = _git(repo, "diff", "-U0", f"{against}...{m['branch']}").stdout.splitlines()
    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    why = al.check_patch(names, added, removed)
    if why:
        return None, f"the patch breaks the autolab rules: {why}"
    stat = _git(repo, "diff", "--stat", f"{against}...{m['branch']}").stdout.strip()
    return {"name": name, "branch": m["branch"], "commit": head, "files": names, "stat": stat, "verdict": m.get("verdict"),
            "review": m.get("review_recommendation"), "cells": cells, "plan": plan}, ""


def _fresh_warning(name: str) -> str:
    """Review R-K5/R-S7: an accept on a single dev-set experiment is a winner's pick. Warn (not block) when there is no run of the same
    hypothesis on other data (`проверка <имя>` makes `<имя>_fresh`) with a verdict."""
    base = al.family_base(_safe_name(name) or name)
    for suffix in ("_fresh", "_fresh2", "_fresh3"):
        if al.meta(base + suffix).get("verdict"):
            return ""
    if al.meta(base).get("check_of"):   # this experiment is itself the check on other data
        return ""
    return "\n⚠️ Подтверждения на других данных нет: сначала `проверка <имя>` (свежий или узкий набор), иначе это отбор победителя по одному прогону."


def ask_accept_stack(uid) -> dict:
    import autolab_chief as ch
    info, why = ch.accept_stack_preflight()
    if not info:
        return reply("accept refused: " + why)
    cfg = al.limits_cfg()
    key = "есть" if deploy_key().exists() else "НЕТ (будет только слияние на хосте, push не выполнится)"
    code = issue(uid, "accept_stack", info)
    return reply(f"ПРИНЯТЬ СТЕК {al.STACK_BRANCH} ({len(info['names'])} гипотез: {', '.join(info['names'])}): слить в main хостовой копии и запушить {cfg['push_branch']} в {cfg['push_remote']} (без --force). Ключ: {key}.\n"
                 f"Общая проверка стека: {info['check']}.\nСхема статусов: ячейки {', '.join(info['cells']) or '-'}; решения по этим гипотезам будут записаны как принятые.\n{info['stat']}\nКод {code}, 10 минут.",
                 confirm_buttons(code, "Принять стек и запушить"))


def do_accept_stack(uid, e: dict) -> str:
    import autolab_chief as ch
    return ch.accept_stack_do(uid, e["args"])


def cmd_stack_ru(uid, args: list[str]) -> dict:
    import autolab_chief as ch
    if args and args[0].lower() in ("сбросить", "reset"):
        journal(uid, "decision", what="stack_reset")
        return reply(ch.stack_reset("вручную из чата"))
    return reply(ch.stack_text(), [[{"label": "✅ Принять стек", "data": "al:cmd:accept:стек"}, {"label": "🗑 Сбросить стек", "data": "al:cmd:stack:сбросить"}]])


def ask_accept(uid, args: list[str]) -> dict:
    nm = args[0] if args else ""
    if nm.lower() in ("стек", "stack"):
        return ask_accept_stack(uid)
    if _safe_name(nm) and al.meta(al.family_base(nm)).get("stale"):
        return reply(f"⚠️ {nm} устарела: после неё принята другая гипотеза того же типа, результат получен на старом коде. Сначала перепроверьте: 🔁 Повтор (с нуля).",
                     [[{"label": "🔁 Повтор", "data": al._cb("rerun", nm) or "al:cmd:list"}]])
    info, why = accept_preflight(args[0] if args else "")
    if not info:
        return reply("accept refused: " + why)
    if _fresh_warning(nm) and not any(x.lower() in ("force", "--force", "без-проверки") for x in args[1:]):   # R-L2/R-S7: no accept on a single dev-set run
        return reply("accept refused: нет подтверждения на других данных. Сначала `проверка " + nm + "` (свежий или узкий набор). "
                     "Осознанное исключение: `принять " + nm + " force`.")
    cfg = al.limits_cfg()
    key = "есть" if deploy_key().exists() else "НЕТ (будет только слияние на хосте, push не выполнится)"
    rev = f"рекомендация рецензента: {info['review']}" if info.get("review") else "рецензии нет (`рецензия <имя>`)"
    code = issue(uid, "accept", info)
    plan_txt = "; ".join(f"{it['cell']}: " + (f"статус → {it['status']}, потенциал {it['headroom']}" + (f", прирост {it['gain']}" if "gain" in it else "") if it.get("status") else "только история") for it in info["plan"])
    head = ("ЗАПИСАТЬ РЕЗУЛЬТАТ В СХЕМУ (кода для слияния нет)" if info.get("no_code") else f"ПРИНЯТЬ {info['branch']} @ {info['commit']}: слить в main хостовой копии ~/harness-lab/autolab_repo")
    return reply(f"{head} и запушить {cfg['push_branch']} в {cfg['push_remote']} (без --force). Ключ для push: {key}.\n"
                 f"Схема статусов: {plan_txt}.\n"
                 f"вердикт: {info.get('verdict') or 'нет'}; {rev} (она не блокирует).{_fresh_warning(nm)}\n{info['stat']}\nКод {code}, 10 минут.",
                 confirm_buttons(code, "Принять и запушить"))


def _status_into_clone(clone: Path, env: dict, info: dict) -> str:
    """Replay every result not yet pushed (this accept and earlier clarify/reject notes) through status_check.update on the clone's status/ and
    commit it separately. Returns an error text, or "" when done (also when there was nothing to write)."""
    plan, pos = astat.pending_plan(al.EXP)
    info["_event_pos"] = pos
    if not plan:
        return ""
    cs = clone / "status"
    if not (cs / "index.json").exists():      # the repository on GitHub has no scheme yet: import the host copy whole
        astat.sync_tree(al.status_root(), cs)
        msg = "status: схема статусов (импорт с хоста автолаборатории)"
    else:
        try:
            astat.apply(cs, plan)
        except ValueError as ex:
            _git(clone, "checkout", "--", "status", env=env)
            _git(clone, "clean", "-fdq", "status", env=env)
            return f"статус не записан в репозиторий ({ex}); код не запушен, чтобы не разводить историю: исправьте ячейку и повторите"
        msg = f"status: {info['name']} ({', '.join(info.get('cells') or [])}) вердикт и история"
    if _git(clone, "add", "status", env=env).returncode or _git(clone, "commit", "-m", msg, env=env).returncode:
        _git(clone, "reset", "--hard", "HEAD", env=env)
        return "не удалось создать коммит статуса"
    return ""


def push_master(info: dict, before: str) -> str:
    cfg = al.limits_cfg()
    key = deploy_key()
    if not key.exists():
        return "push не выполнен: нет ключа (~/.config/autolab/deploy_key)"
    remote = os.environ.get("AUTOLAB_PUSH_REMOTE") or cfg["push_remote"]
    br = cfg["push_branch"]
    clone = Path(os.environ.get("AUTOLAB_PUSH_CLONE") or al.LAB / "push_repo")
    env = dict(os.environ, GIT_SSH_COMMAND=f"ssh -i {key} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes", GIT_TERMINAL_PROMPT="0")
    if not (clone / ".git").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", remote, str(clone)], capture_output=True, text=True, env=env)
        if r.returncode:
            return "push не выполнен: клонирование не удалось: " + (r.stderr or r.stdout)[-300:]
    for step in (("fetch", "origin"), ("checkout", "-B", br, f"origin/{br}"), ("reset", "--hard", f"origin/{br}")):
        r = _git(clone, *step, env=env)
        if r.returncode:
            return f"push не выполнен: git {step[0]}: " + (r.stderr or r.stdout)[-300:]
    if not info.get("no_code"):
        patch = _git(al.repo_root(), "diff", f"{before}...{info['branch']}").stdout
        pf = clone.parent / f"{clone.name}.patch"
        pf.write_bytes(patch.replace("\r\n", "\n").encode("utf-8"))
        if _git(clone, "apply", "--check", str(pf), env=env).returncode:
            return f"push не выполнен: патч не применяется к {br} на GitHub (история разошлась с хостовой копией); слияние на хосте есть, перенесите вручную: `autolab.py export {info['name']}`"
        if _git(clone, "apply", "--index", str(pf), env=env).returncode or _git(clone, "commit", "-m", f"autolab accept {info['name']} ({info['commit']})", env=env).returncode:
            _git(clone, "reset", "--hard", f"origin/{br}", env=env)
            return "push не выполнен: не удалось создать коммит"
    st = _status_into_clone(clone, env, info)
    if st:
        return "push не выполнен: " + st
    r = _git(clone, "push", "origin", br, env=env)   # never --force
    if r.returncode:
        _git(clone, "reset", "--hard", f"origin/{br}", env=env)
        return "push не выполнен: " + al.scrub((r.stderr or r.stdout))[-300:]
    astat.events_mark_pushed(al.EXP, info.get("_event_pos") or [])
    try:
        astat.sync_tree(clone / "status", al.status_root())   # the host copy gets whatever the scheme session committed meanwhile
    except OSError as ex:
        return f"запушено, но копия схемы на хосте не обновлена: {ex}"
    return f"запушено: {_git(clone, 'rev-parse', '--short', 'HEAD').stdout.strip()} в {remote} {br}"


def do_accept(uid, e: dict) -> str:
    info = e["args"]
    info2, why = accept_preflight(info["name"])
    if not info2 or info2["commit"] != info["commit"]:
        return f"accept refused at the last check: {why or 'the branch moved since the button was shown'}"
    info = dict(info2)                      # the plan and the cells as they are now
    before, merged = None, "кода для слияния нет"
    if not info.get("no_code"):
        repo = al.repo_root()
        before = _git(repo, "rev-parse", "HEAD").stdout.strip()
        snap = al.snapshot_protected()
        r = _git(repo, "merge", "--no-ff", "-m", f"autolab accept {info['name']} ({info['commit']})", info["branch"])
        if r.returncode:
            _git(repo, "merge", "--abort")
            return "слияние не удалось и отменено:\n" + (r.stdout + r.stderr)[-500:]
        if al.snapshot_protected() != snap:
            _git(repo, "reset", "--hard", before)
            return "слияние изменило защищённые файлы: откат"
        merged = f"влито в main хостовой копии ({_git(repo, 'rev-parse', '--short', 'HEAD').stdout.strip()})"
    try:
        astat.apply(al.status_root(), info["plan"])
        for it in info["plan"]:
            astat.events_add(al.EXP, {"name": info["name"], "item": it})
        al.record_journal(info["name"], accepted=("да (переключатели, без кода)" if info.get("no_code") else f"да (код влит, ветка {info['branch']})"))
        stat_txt = "схема: " + ", ".join(info["cells"]) + " обновлены, журнал дополнен"
    except ValueError as ex:   # preflight made a dry run, so this is a race with another editor of status/
        stat_txt = f"схема НЕ обновлена: {ex}"
    al.set_meta(info["name"], stage="accepted", merged_by=str(uid), merged_at=time.strftime("%Y-%m-%d %H:%M:%S"), status_cells=info["cells"])
    stale = al.set_decision(info["name"], "accepted", "", str(uid))      # before the push: the decision row travels with it
    pushed = push_master(info, before)
    al.set_meta(info["name"], stage="accepted", merged_by=str(uid), merged_at=time.strftime("%Y-%m-%d %H:%M:%S"), push_result=pushed, status_cells=info["cells"])
    return f"{merged}; {stat_txt}; {pushed}" + (f"\n♻️ Устарели отложенные гипотезы тех же типов (перепроверьте с нуля): {', '.join(stale)}" if stale else "")


# ------------------------------------------------------------------ stop / resume
def do_stop() -> str:
    for p in _stop_files():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " stop from chat\n")
    return "🛑 Остановлено: новых вызовов LLM и постановок в очередь не будет, диспетчер уйдёт после текущего эксперимента. `продолжить` снимает флаги."


def do_resume() -> str:
    gone = [p.name for p in _stop_files() if p.exists()]
    for p in _stop_files():
        p.unlink(missing_ok=True)
    return "▶️ Флаги сняты: " + (", ".join(gone) or "их и не было")


HELP = """🧪 Автолаборатория. Команда пишется словом (можно с /autolab или @бот). <имя> — имя гипотезы ab_..., [..] — необязательно, режим: accuracy или economy. Тратящее сначала покажет оценку и кнопку «Запустить» (код 10 минут).

👀 Смотреть
• статус • очередь • расход • отложенные
• гипотезы [все] — ждущие решения, кнопка на каждую
• отчёт <имя> • вердикт <имя>
• журнал [<блок> <тип> [N]] — последние записи или история ячейки

✅ Решать (кнопки: принять, отклонить, ревью, повтор, другие данные, отложить)
• гипотеза <имя> — карточка • принять <имя> • отклонить <имя> • отложить <имя>
• рецензия <имя> — мнение модели и совет, где перепроверить
• повтор <имя> [n] — те же данные, n повторов (без n посчитаю)
• проверка <имя> [набор] — другие данные: набор по совету рецензии или ваш

🧪 Задания
• идеи [n] [режим] [тема] — только придумать и проверить в симуляторе
• цикл [n] [режим] [тема] — придумать, запустить, дождаться вердикта
• своя <блок>/<тип> <текст> — ваша гипотеза, первой в очереди
  пример: цикл 2 economy порядок задач при бюджете 30k

🗺 Схема (14 блоков × 4 типа: coding, math, search, data_analysis)
• схема • блоки • типы • приоритеты [режим]
• блок <имя> • тип <тип> • ячейка <блок> <тип>
• привязать <имя> <блок>/<тип>

🎛 Управлять
• стоп • продолжить • меню • /id
• автопилот [вкл|выкл] • локально [вкл|выкл]
• стек [сбросить] • принять стек — накопленные шефом гипотезы (ветка auto/stack): состояние, общий перепрогон, слияние в main
• шеф [вкл|выкл|план|дайджест|идеи] • шеф идея <текст> • шеф вопрос <текст> — агент-шеф: сам ведёт траекторию улучшений, принимает только в ветку auto/
• ключ (или проверка ключа) — провайдер, модели, баланс ключа Claude/OpenRouter без траты; реальный тест кнопкой
• команды (или помощь) — этот список • команды файл — подробно файлом для команды"""
ALIASES = {"принять": "accept", "merge": "accept", "отчёт": "report", "отчет": "report", "рецензия": "review", "статус": "status", "очередь": "queue",
           "стоп": "stop", "продолжить": "resume", "список": "list", "гипотезы": "list", "проверка": "check", "ключ": "key", "расход": "usage", "меню": "menu", "помощь": "help", "команды": "help", "вердикт": "verdict", "блок": "block", "тип": "type", "ячейка": "cell", "описание": "cell", "схема": "scheme", "блоки": "blocks", "типы": "types", "журнал": "journal", "история": "journal", "journal": "journal", "приоритеты": "priorities", "привязать": "bind", "автопилот": "autopilot", "шеф": "chief", "стек": "stack", "локально": "localdata", "идеи": "propose", "идея": "propose", "цикл": "cycle", "повтор": "rerun", "отклонить": "reject", "отложить": "defer", "отложенные": "postponed", "гипотеза": "card", "карточка": "card", "hy": "card", "своя": "mine", "rj": "reject"}


def menu() -> dict:
    row = lambda *items: [{"label": a, "data": f"al:cmd:{b}"} for a, b in items]
    return reply("Лаборатория: выберите действие. Действия за деньги (Claude) и принятие сначала покажут оценку и попросят подтверждение; повторы, проверки и сбор на Qwen запускаются сразу.",
                 [row(("Статус", "status"), ("Очередь", "queue"), ("Расход", "usage")),
                  row(("Гипотезы", "list"), ("Новый цикл ×2", "cycle:2")),
                  row(("Автопилот", "autopilot"), ("Локальный сбор", "localdata")),
                  row(("Схема", "scheme"), ("Блоки", "blocks"), ("Типы", "types")),
                  row(("Приоритеты", "priorities")),
                  row(("Стоп", "stop"), ("Продолжить", "resume"))])


def list_reply(show_all: bool = False) -> dict:
    """Buttons: the card (with the five decisions) of every hypothesis that waits for a decision, then the postponed ones."""
    act = sorted([f.stem for f in al.DRAFTS.glob("*.yaml") if not f.name.endswith(".queued.yaml") and _hyp_group(f.stem) in ("decide", "running")], key=_gain_key)[:8]
    btns = [[{"label": f"📄 {n}"[:40], "data": al._cb("hy", n) or "al:cmd:list"}] for n in act]
    npost = sum(1 for f in al.DRAFTS.glob("*.meta.json") if al.meta(f.name[:-10]).get("decision") == "postponed")
    btns.append([{"label": "📜 Журнал", "data": "al:cmd:journal"}, {"label": f"⏸ Отложенные ({npost})", "data": "al:cmd:postponed"}])
    if not show_all:
        btns.append([{"label": "📋 Все гипотезы", "data": "al:cmd:list:все"}])
    return reply(cmd_list(show_all), btns)


# ------------------------------------------------------------------ entry points
def handle(uid, text: str, chat=None, chat_type: str = "private") -> dict | None:
    """The entry point of the transport. Returns None for a rejected user or chat (the transport stays silent)."""
    if not allowed(uid, chat, chat_type):
        journal(uid, "denied", text=text[:80], chat=str(chat))
        return None
    parts = text.strip().split()
    if parts and parts[0].lstrip("/").split("@")[0].lower() in ("autolab", "al"):
        parts = parts[1:]
    cmd, args = (parts[0].lower() if parts else "menu"), parts[1:]
    cmd = ALIASES.get(cmd, cmd)
    if cmd != "mine":
        _drop_capture(uid, chat or uid)         # any other command ends the wait for the text of a hypothesis
    journal(uid, "command", cmd=cmd, args=" ".join(args)[:200])
    try:
        if cmd == "status":
            return reply(cmd_status())
        if cmd == "queue":
            return reply(cmd_queue())
        if cmd == "usage":
            return reply(cmd_usage())
        if cmd == "menu":
            return menu()
        if cmd in ("list", "branches"):
            return list_reply(bool(args) and args[0].lower() in ("все", "all"))
        if cmd == "report":
            return cmd_report(args[0] if args else "")
        if cmd == "verdict":
            return reply(cmd_verdict(args[0] if args else ""))
        if cmd == "propose":
            return ask_cycle(uid, args, False)
        if cmd == "cycle":
            return ask_cycle(uid, args, True)
        if cmd == "autopilot":
            if args and args[0].lower() in ("вкл", "on"):
                return ask_autopilot_on(uid, args)
            if args and args[0].lower() in ("выкл", "off"):
                journal(uid, "decision", what="autopilot_off")
                al.autopilot_flag().unlink(missing_ok=True)
                return reply("⏹ Автопилот выключен. Уже поставленное в очередь доработает.")
            return reply(cmd_autopilot_ru(), [[{"label": "Включить" if not al.autopilot_flag().exists() else "Выключить", "data": "al:cmd:autopilot:" + ("вкл" if not al.autopilot_flag().exists() else "выкл")}]])
        if cmd == "chief":
            if args and args[0].lower() in ("вкл", "on"):
                return ask_chief_on(uid, args)
            if args and args[0].lower() in ("выкл", "off"):
                journal(uid, "decision", what="chief_off")
                al.chief_flag().unlink(missing_ok=True)
                return reply("⏹ Шеф выключен. Уже поставленное в очередь доработает, автопилот снова может предлагать (если включён).")
            return cmd_chief_ru(args, uid, chat)
        if cmd == "localdata":
            if args and args[0].lower() in ("вкл", "on"):
                return ask_localdata_on(uid, args)
            if args and args[0].lower() in ("выкл", "off"):
                journal(uid, "decision", what="localdata_off")
                al.localdata_flag().unlink(missing_ok=True)
                (al.LAB / "STREAM_STOP_local").parent.mkdir(parents=True, exist_ok=True)
                (al.LAB / "STREAM_STOP_local").write_text("stop from chat\n")
                return reply("⏹ Локальный сбор выключен: поток закончит текущую порцию и остановится.")
            on = al.localdata_flag().exists()
            return reply(cmd_localdata_ru(), [[{"label": "Выключить" if on else "Включить", "data": "al:cmd:localdata:" + ("выкл" if on else "вкл")}]])
        if cmd == "block":
            return status_block(args)
        if cmd == "type":
            return status_type(args)
        if cmd == "cell":
            return status_cell(args)
        if cmd == "blocks":
            return status_blocks()
        if cmd == "types":
            return status_types()
        if cmd == "journal":
            return status_journal(args)
        if cmd == "scheme":
            return status_scheme()
        if cmd == "priorities":
            return status_priorities(args)
        if cmd == "bind":
            return bind_cell(uid, args)
        if cmd == "rerun":
            return ask_rerun(uid, args)
        if cmd == "check":
            return ask_check(uid, args)
        if cmd == "key":
            return ask_key(uid)
        if cmd == "card":
            return cmd_card(args)
        if cmd == "reject":
            return decide(uid, args, "rejected")
        if cmd == "defer":
            return decide(uid, args, "postponed")
        if cmd == "postponed":
            return cmd_postponed()
        if cmd == "mine":
            return ask_mine(uid, args, chat, chat_type)
        if cmd == "accept":
            return ask_accept(uid, args)
        if cmd == "stack":
            return cmd_stack_ru(uid, args)
        if cmd == "review":
            return ask_review(uid, args, chat)
        if cmd == "ok" and args:
            return handle_callback(uid, f"al:ok:{args[0]}")
        if cmd == "stop":
            journal(uid, "decision", what="stop")
            return reply(do_stop())
        if cmd == "resume":
            journal(uid, "decision", what="resume")
            return reply(do_resume())
        if args and args[0].lower() in ("файл", "file"):      # the long version for the team, as a document
            doc = next((c for c in (HERE / "COMMANDS.md", al.LAB / "harness" / "tools" / "COMMANDS.md") if c.exists()), None)
            return reply("Команды автолаборатории: подробно, с форматом ввода", document=doc) if doc else reply(HELP)
        return reply(HELP)
    except Exception as ex:  # noqa: BLE001 - a bug must not kill the bot's handler
        journal(uid, "error", cmd=cmd, error=f"{type(ex).__name__}: {str(ex)[:200]}")
        return reply(f"error: {type(ex).__name__}: {str(ex)[:200]}")


def handle_callback(uid, data: str, chat=None, chat_type: str = "private") -> dict | None:
    if not allowed(uid, chat, chat_type):
        journal(uid, "denied", text=data[:60])
        return None
    try:
        _, verb, code = data.split(":", 2)
    except ValueError:
        return reply("bad button")
    if verb == "cmd":   # a menu / report button: the same as typing the command (spending ones still ask for confirmation)
        return handle(uid, " ".join(code.split(":")), chat, chat_type)
    if verb == "cancel":
        p = _pending()
        if p.get(code, {}).get("user") == str(uid):
            del p[code]
            _save_pending(p)
        journal(uid, "decision", what="cancel", code=code)
        return reply("cancelled")
    e, why = consume(uid, code, ("cycle", "propose", "rerun", "check", "keyprobe", "review", "accept", "accept_stack", "autopilot_on", "chief_on", "localdata_on", "mine"))
    if not e:
        journal(uid, "refused", what="confirm", why=why)
        return reply("not done: " + why)
    journal(uid, "decision", what=e["action"], args=json.dumps(e["args"])[:200])
    out = {"cycle": lambda: do_cycle(uid, e, True), "propose": lambda: do_cycle(uid, e, False), "rerun": lambda: do_rerun(uid, e), "check": lambda: do_check(uid, e), "keyprobe": lambda: do_keyprobe(uid, e),
           "review": lambda: do_review(uid, e), "accept": lambda: do_accept(uid, e), "accept_stack": lambda: do_accept_stack(uid, e), "autopilot_on": lambda: do_autopilot_on(uid, e), "chief_on": lambda: do_chief_on(uid, e), "localdata_on": lambda: do_localdata_on(uid, e), "mine": lambda: do_mine(uid, e)}[e["action"]]()
    journal(uid, "result", what=e["action"], out=out[:200])
    return reply(out)


# ------------------------------------------------------------------ selftest (no Telegram, no Claude, no gateway)
def selftest() -> None:
    import tempfile
    global CFG_DIR
    tmp = Path(tempfile.mkdtemp())
    CFG_DIR = tmp / "cfg"
    al.CFG_DIR = CFG_DIR   # autolab.py shares the config dir (outbox, autopilot flag): never the real one in a selftest
    al.EXP = tmp / "exp"
    al.DRAFTS = al.EXP / "autolab"
    al.USAGE = al.DRAFTS / "usage.jsonl"
    al.STOP_FILE = tmp / "AUTOLAB_STOP"
    (al.EXP / "queue" / "pending").mkdir(parents=True)
    (al.EXP / "queue" / "done").mkdir(parents=True)
    (al.EXP / "queue" / "running").mkdir(parents=True)
    (al.EXP / "sets").mkdir()
    al.DRAFTS.mkdir(parents=True)
    # 1. whitelist: missing file, empty list, other user -> silent rejection and a journal line
    assert handle(1, "/autolab status") is None
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    _allow_file().write_text('{"user_ids": []}')
    assert handle(1, "/autolab status") is None
    _allow_file().write_text('{"user_ids": [42, "43"]}')
    assert handle(7, "/autolab stop") is None and not al.STOP_FILE.exists()
    assert "Шлюз" in handle(43, "/autolab status")["text"]
    assert "denied" in (CFG_DIR / "tg_journal.jsonl").read_text()
    # 2. stop / resume
    assert "Остановлено" in handle(42, "/autolab stop")["text"] and al.STOP_FILE.exists()
    assert "остановлена" in handle(42, "/autolab cycle 2 order")["text"].lower() and not _pending()   # no code is issued while stopped
    assert "Флаги сняты" in handle(42, "/autolab resume")["text"] and not al.STOP_FILE.exists()
    # 3. a spending command needs a code from the same user; one use only; cancel works; expiry works
    spawned = []
    globals()["_spawn"] = lambda argv, tag: spawned.append(argv) or "started (mock)"
    r = handle(42, "/autolab cycle 5 task order at 30k")
    assert "Код" in r["text"] and "примерно 900 запросов" in r["text"], r["text"]   # n is capped by max_hyp_per_cycle (3)
    code = r["buttons"][0][0]["data"].split(":")[2]
    assert not spawned
    assert "another user" in handle_callback(43, f"al:ok:{code}")["text"] and not spawned   # wrong user cannot use it
    assert "started" in handle_callback(42, f"al:ok:{code}")["text"] and spawned[0][:4] == ["cycle", "--n", "3", "--wait"] and "--topic" in spawned[0]
    assert "unknown or expired" in handle_callback(42, f"al:ok:{code}")["text"] and len(spawned) == 1   # one-time
    r = handle(42, "/autolab propose 1")
    c2 = r["buttons"][0][0]["data"].split(":")[2]
    assert handle_callback(42, f"al:cancel:{c2}")["text"] == "cancelled" and "unknown" in handle_callback(42, f"al:ok:{c2}")["text"]
    r = handle(42, "/autolab propose 1")
    c3 = r["buttons"][0][0]["data"].split(":")[2]
    p = _pending()
    p[c3]["expires"] = time.time() - 1
    (CFG_DIR / "tg_pending.json").write_text(json.dumps(p))
    assert "unknown or expired" in handle(42, f"/autolab ok {c3}")["text"]
    # 4. rerun makes a new draft with more repeats and submits it (mock the lab loader)
    import yaml
    (al.EXP / "sets" / "mixed100_dev.yaml").write_text("role: dev\ntasks: {math: 4}\nseed: 1\n")
    d = {"name": "ab_x_test", "hypothesis": "h", "set": "mixed100_dev", "model": "auto", "repeats": 1, "budget": 30000,
         "arms": {"a": {}, "b": {"env": {"HARNESS_SORTER": "fifo"}}}}
    al.write_draft(d)
    al.set_meta("ab_x_test", stage="verdict", sim_verdict="pass", verdict={"b": "clarify"})
    assert "Повторов должно быть" in handle(42, "/autolab rerun ab_x_test 9")["text"]
    import types
    fake_lab = types.ModuleType("lab")
    fake_lab.load_experiment = lambda p: {}
    sys.modules["lab"] = fake_lab
    r = handle(42, "/autolab rerun ab_x_test 3")          # gateway/Qwen tokens only: no confirmation, the plan and the result come in one message
    assert "Запуск: 3 повторов" in r["text"] and "задано вручную" in r["text"] and not r["buttons"] and "Код " not in r["text"]
    out = r["text"]
    assert "ab_x_test_r3 поставлен в очередь" in out and ["await", "ab_x_test_r3"] in spawned, out
    assert (al.EXP / "queue" / "pending" / "ab_x_test_r3.yaml").exists()
    assert yaml.safe_load((al.EXP / "queue" / "pending" / "ab_x_test_r3.yaml").read_text())["repeats"] == 3
    # 4b. decisions on a hypothesis: five buttons, a task type held by one active hypothesis, postponed ones, a repeat planned from the data
    import status_check as _sc0
    dec_status = tmp / "dec_status"
    astat.mini(dec_status)
    os.environ["AUTOLAB_STATUS_ROOT"] = str(dec_status)
    al.set_meta("ab_x_test", target_cells=["blk/a"], threshold_used=7.69, queued_at=time.time() + 1,
                verdict={"a": None, "b": "clarify"}, result={"a": {"points_per_run": 100, "tokens_per_run": 20000, "accuracy": 0.5, "points_diff_vs_first": None},
                                                       "b": {"points_per_run": 104, "tokens_per_run": 21000, "accuracy": 0.52, "points_diff_vs_first": {"mean": 4.0, "ci95": [-2.0, 10.0]}}})
    card = handle(42, "гипотеза ab_x_test")
    labels = [b["label"] for row in card["buttons"][:3] for b in row]
    assert labels == ["✅ Принять", "❌ Отклонить", "🔍 Ревью", "🔁 Повтор", "🧪 Другие данные", "⏸ Отложить"], labels
    assert "🔒 Держит типы задач: a" in card["text"] and "blk/a" in card["text"] and "+4.0" in card["text"], card["text"]
    assert all(len(b["data"].encode()) <= 64 for row in card["buttons"] for b in row)
    # goal modes: the card and the list show the mode; priorities per mode; cycle takes a mode; a hypothesis for both modes and a cross-check flag are visible
    assert "Режим: 🎯 accuracy" in card["text"]
    al.set_meta("ab_x_test", target_mode="both", cross_check=True)
    c2 = handle(42, "гипотеза ab_x_test")["text"]
    assert "Режим: ⚖️ оба режима" in c2 and "нужна проверка на другом режиме" in c2, c2
    al.set_meta("ab_x_test", target_mode="economy", cross_check=False)
    assert "Режим: 💸 economy" in handle(42, "гипотеза ab_x_test")["text"] and "режим: 💸 economy" in cmd_list(True)
    pe = handle(42, "приоритеты economy")
    assert "Приоритеты режима 💸 economy" in pe["text"] and {x["data"] for row in pe["buttons"] for x in row} >= {"al:cmd:priorities:accuracy", "al:cmd:priorities:economy", "al:cmd:priorities"}, pe
    assert "Приоритеты режима 🎯 accuracy" in handle(42, "приоритеты accuracy")["text"] and "Что делать в первую очередь" in handle(42, "приоритеты")["text"]
    cy = handle(42, "цикл 2 economy порядок задач")
    assert "Режим: 💸 economy" in cy["text"] and "Тема: порядок задач" in cy["text"], cy["text"]
    n_sp0 = len(spawned)
    handle_callback(42, cy["buttons"][0][0]["data"])
    assert spawned[n_sp0][:4] == ["cycle", "--n", "2", "--wait"] and spawned[n_sp0][spawned[n_sp0].index("--mode") + 1] == "economy" and "--topic" in spawned[n_sp0]
    assert "поровну" in handle(42, "цикл 2")["text"]
    al.set_meta("ab_x_test", target_mode="accuracy")
    plan = rerun_plan("ab_x_test")
    assert plan["n"] == 3 and "Нужно примерно 3" in plan["text"] and plan["est"] > 0, plan             # width 6 -> target 3.48: 1*(6/3.48)^2 = 2.97 -> 3 repeats
    # another hypothesis holds type a: the repeat is refused with the active one's five buttons
    d2 = dict(d, name="ab_other")
    al.write_draft(d2)
    al.set_meta("ab_other", target_cells=["blk/a"], queued_at=time.time() + 1)
    (al.EXP / "queue" / "pending" / "ab_other.yaml").write_text("name: ab_other\n")
    assert al.active_types()["a"] == "ab_other"
    blocked = handle(42, "повтор ab_x_test")
    assert "⛔" in blocked["text"] and "a: ab_other" in blocked["text"] and "Сначала решите по активной" in blocked["text"], blocked["text"]
    assert any(b["data"] == "al:cmd:defer:ab_other" for row in blocked["buttons"] for b in row)
    out = handle(42, "отложить ab_x_test")
    assert "отложена" in out["text"] and "ab_x_test_r3" in out["text"] and al.decision_of("ab_x_test_r3") == "postponed"      # its waiting repeat leaves the queue
    assert not (al.EXP / "queue" / "pending" / "ab_x_test_r3.yaml").exists() and (al.EXP / "queue" / "cancelled" / "ab_x_test_r3.yaml").exists()
    pp = handle(42, "отложенные")
    assert "ab_x_test" in pp["text"] and "актуальна" in pp["text"] and al.active_types()["a"] == "ab_other"
    rj = handle(42, "отклонить ab_other")
    assert "отклонена" in rj["text"] and al.active_types() == {} and (al.EXP / "queue" / "cancelled" / "ab_other.yaml").exists()
    assert "уже отклонена" in handle(42, "отклонить ab_other")["text"]
    # the first accepted hypothesis of the type makes the postponed ones stale; a stale one cannot be accepted without a fresh run
    al.set_meta("ab_win", target_cells=["blk/a"], queued_at=time.time() + 1)
    assert al.set_decision("ab_win", "accepted", "", "42") == ["ab_x_test"] and al.meta("ab_x_test")["stale"] is True
    assert "устарела" in handle(42, "принять ab_x_test")["text"] and "устарела" in handle(42, "отложенные")["text"]
    fresh = handle(42, "повтор ab_x_test")
    assert "пересчитает всё с нуля" in fresh["text"] and "около" in fresh["text"] and "поставлен в очередь" in fresh["text"], fresh["text"]
    assert al.decision_of("ab_x_test") is None and al.active_types()["a"] in ("ab_x_test", "ab_x_test_r3_r3", "ab_x_test_r3")
    # the command list: "команды" and "помощь" give the same text, "команды файл" the long version; every word of the list is a command the dispatcher knows
    assert handle(42, "команды")["text"] == handle(42, "помощь")["text"] and len(handle(42, "команды")["text"]) < 3500
    cf = handle(42, "команды файл")
    assert cf.get("document") and str(cf["document"]).endswith("COMMANDS.md") and "проверка <имя> [набор]" in Path(cf["document"]).read_text(encoding="utf-8")
    for w in re.findall(r"^[•] ([а-яёa-z/]+)", HELP.replace(" • ", "\n• "), re.M):
        assert ALIASES.get(w, w) in ("accept", "report", "review", "status", "queue", "stop", "resume", "list", "check", "usage", "menu", "help", "verdict", "block", "type", "cell", "scheme", "blocks", "types", "journal", "priorities", "bind", "autopilot", "chief", "stack", "localdata", "propose", "cycle", "rerun", "reject", "defer", "postponed", "card", "mine", "id", "/id", "key") or w in ("/id",), w
    # the key check: free status first, the paid probe only after the button; no key in the text
    import tempfile as _tf
    kf2 = Path(_tf.mkdtemp()) / "k"
    kf2.write_text("sk-or-v1-FAKE-CTL-SELFTEST-0000" + chr(10))
    os.environ["AUTOLAB_KEY_FILE"] = str(kf2)
    calls_k = []
    al_status, al_llm_get = al.key_status, None
    al.key_status = lambda probe=False: calls_k.append(probe) or ("PROBED" if probe else "STATUS sk-or-v1-no")
    kr = handle(42, "ключ")
    assert calls_k == [False] and "STATUS" in kr["text"] and "Проверить запросом" in kr["buttons"][0][0]["label"] and "FAKE-CTL" not in kr["text"], kr
    assert handle(42, "проверка ключа")["text"].startswith("STATUS") and calls_k == [False, False]
    assert "PROBED" in handle_callback(42, kr["buttons"][0][0]["data"])["text"] and calls_k[-1] is True and handle_callback(42, kr["buttons"][0][0]["data"])["text"].count("PROBED") == 0   # one-time code
    al.key_status = al_status
    os.environ.pop("AUTOLAB_KEY_FILE", None)
    # check on other data: the reviewer's advice (a set of the lab) or your own set; holdout and unknown sets are refused; the run is a new experiment <base>_fresh on that set
    (al.EXP / "sets" / "dev_blk.yaml").write_text("role: dev\ntasks: {math: 8}\nseed: 5\n")
    (al.EXP / "sets" / "hold_blk.yaml").write_text("role: holdout\ntasks: {math: 8}\nseed: 6\n")
    assert "не назвала набор" in handle(42, "проверка ab_x_test")["text"] and "Нет набора hold_blk" in handle(42, "проверка ab_x_test hold_blk")["text"]
    al.set_meta("ab_x_test", review_data_set="dev_blk")
    ck = handle(42, "проверка ab_x_test")
    assert "Набор: dev_blk" in ck["text"] and "ab_x_test_fresh поставлен в очередь" in ck["text"] and not ck["buttons"] and "Код " not in ck["text"], ck["text"]      # no confirmation: gateway tokens only
    assert yaml.safe_load((al.EXP / "queue" / "pending" / "ab_x_test_fresh.yaml").read_text().split("# ---- written by lab.py")[0])["set"] == "dev_blk" and ["await", "ab_x_test_fresh"] in spawned
    assert al.meta("ab_x_test_fresh")["check_of"] == "ab_x_test" and "review_data_set" not in al.meta("ab_x_test_fresh")
    # the verdict message carries the five buttons while the hypothesis is active, none once it is closed
    al.set_meta("ab_x_test", decision=None)
    al.notify_verdict("ab_x_test", ["42"])
    ob = sorted((CFG_DIR / "tg_outbox").glob("*.json"))[-1]
    msg = json.loads(ob.read_text(encoding="utf-8"))
    assert [b["label"] for row in msg["buttons"][:2] for b in row][:2] == ["✅ Принять", "❌ Отклонить"] and "заняты" in msg["text"], msg
    ob.unlink()
    # 4c. a hypothesis of a human: the button under the cell; a held type is said plainly with the active hypothesis' buttons; then the text, the confirmation, the spawn
    cellrep = handle(42, "ячейка blk a")
    assert any(b["data"] == "al:cmd:mine:blk/a" for row in cellrep["buttons"] for b in row)
    held = handle(42, "своя blk/a попробовать порядок по предсказанной стоимости")
    assert "⛔" in held["text"] and "a: ab_x_test" in held["text"] and "Сначала решите по активной" in held["text"], held["text"]
    assert any(b["data"] == "al:cmd:reject:ab_x_test" for row in held["buttons"] for b in row)
    assert not has_capture(42, 42)
    handle(42, "отклонить ab_x_test")
    asked = handle_callback(42, "al:cmd:mine:blk/a")
    assert has_capture(42, 42) and "Напишите ОДНИМ" in asked["text"] and "blk/a" in asked["text"], asked["text"]
    assert handle_capture(99, 99, "текст чужого") is None and handle_capture(42, 777, "другой чат, не ждём") is None
    assert "Слишком коротко" in handle_capture(42, 42, "ага")["text"] and has_capture(42, 42)
    conf = handle_capture(42, 42, "Попробовать сортировку по предсказанной стоимости вместо fifo, она должна ставить дешёвые задачи вперёд")
    assert not has_capture(42, 42) and "Оформить и запустить" in conf["buttons"][0][0]["label"] and "ПЕРВЫМ" in conf["text"] and "типы задач: a" in conf["text"], conf["text"]
    n_sp = len(spawned)
    assert "started" in handle_callback(42, conf["buttons"][0][0]["data"])["text"] and spawned[n_sp][0] == "mine" and spawned[n_sp][spawned[n_sp].index("--cell") + 1] == "blk/a"
    handle_callback(42, "al:cmd:mine:blk/a")
    assert "отменена" in handle_capture(42, 42, "отмена")["text"] and not has_capture(42, 42)
    handle_callback(42, "al:cmd:mine:blk/a")
    handle(42, "статус")
    assert not has_capture(42, 42)                       # any other command ends the wait
    assert "Формат" in handle(42, "своя")["text"] and "не применима" in handle(42, "своя blk/b что-нибудь")["text"]
    for f_ in (al.EXP / "queue" / "pending").glob("ab_x_test*"):
        f_.unlink()
    astat.events_rewrite(al.EXP, [])            # the journal rows of this block are not part of the accept test below
    os.environ.pop("AUTOLAB_STATUS_ROOT", None)
    # 5. accept = merge in the host copy + push master + the scheme status of the target cells: one button, user-bound one-time code, commit pinned, rules re-checked, never --force
    import shutil
    import status_check as _sc
    host_status = tmp / "host_status"
    astat.mini(host_status)
    os.environ["AUTOLAB_STATUS_ROOT"] = str(host_status)
    repo = tmp / "autolab_repo"
    repo.mkdir()
    (repo / "harness").mkdir()
    g = lambda *a: _git(repo, *a)
    g("init", "-b", "main")
    (repo / "harness" / "a.py").write_text("x = 1\n")
    g("add", "-A")
    g("commit", "-m", "base")
    # the "GitHub": a bare repo with the same files on master plus an unrelated later commit (history differs from the host copy)
    bare = tmp / "github.git"
    subprocess.run(["git", "init", "--bare", "-b", "master", str(bare)], capture_output=True)
    seed = tmp / "seed"
    subprocess.run(["git", "clone", str(bare), str(seed)], capture_output=True)
    (seed / "harness").mkdir()
    (seed / "harness" / "a.py").write_text("x = 1\n")
    (seed / "README.md").write_text("later work\n")
    shutil.copytree(host_status, seed / "status")
    for a_ in (("checkout", "-B", "master"), ("add", "-A"), ("commit", "-m", "github base"), ("push", "origin", "master")):
        _git(seed, *a_)
    g("checkout", "-b", "autolab/ab_m")
    (repo / "harness" / "a.py").write_text("x = 1\ny = 2\n")
    g("commit", "-am", "change")
    sha = g("rev-parse", "--short", "HEAD").stdout.strip()
    g("checkout", "main")
    al.repo_root = lambda: repo
    al.set_meta("ab_m", stage="applied", branch="autolab/ab_m", commit=sha, verdict={"b": "accept"}, review_recommendation="повторить")
    os.environ.update(AUTOLAB_PUSH_REMOTE=str(bare), AUTOLAB_PUSH_CLONE=str(tmp / "push_repo"), AUTOLAB_DEPLOY_KEY=str(tmp / "no_such_key"))
    (al.EXP / "queue" / "running").mkdir(parents=True, exist_ok=True)
    (al.EXP / "queue" / "running" / "ab_m.yaml").write_text("name: ab_m" + chr(10))
    assert "ещё идёт" in handle(42, "принять ab_m")["text"] and "принимать пока нечего" in handle(42, "принять ab_m")["text"]      # running: not a fake "reject"
    (al.EXP / "queue" / "running" / "ab_m.yaml").unlink()
    r = handle(42, "принять ab_m")      # not bound to a cell: refused with a way out, nothing is asked
    assert "не привязана" in r["text"] and "привязать ab_m" in r["text"] and not r["buttons"], r
    assert "Нет гипотезы" in handle(42, "привязать ab_nope blk/a")["text"] and "usage" in handle(42, "привязать ab_m")["text"]
    assert "Не привязано" in handle(42, "привязать ab_m blk/b")["text"] and "Не привязано" in handle(42, "привязать ab_m zzz/a")["text"]   # n/a and unknown cells
    assert "привязана" in handle(42, "привязать ab_m blk/a")["text"] and al.meta("ab_m")["target_cells"] == ["blk/a"]
    assert handle(7, "привязать ab_m blk/a") is None
    assert "нет подтверждения на других данных" in handle(42, "принять ab_m")["text"]      # no check on other data: refused, `force` is the conscious exception
    r = handle(42, "принять ab_m force")      # Russian alias, no slash
    assert "Схема статусов: blk/a: статус → explored-headroom" in r["text"], r["text"]
    assert "ПРИНЯТЬ" in r["text"] and "a.py" in r["text"] and "рекомендация рецензента: повторить" in r["text"] and "НЕТ" in r["text"], r["text"]
    code = r["buttons"][0][0]["data"].split(":")[2]
    assert "another user" in handle_callback(43, f"al:ok:{code}")["text"]
    out = handle_callback(42, f"al:ok:{code}")["text"]
    assert "влито" in out and "push не выполнен: нет ключа" in out and "blk/a обновлены" in out, out      # no deploy key: merged on the host, honest about the push
    assert _sc.load_index(host_status)["entries"][1]["status"] == "explored-headroom" and any("ab_m" in x and "accept" in x for x in astat.cell_history(host_status, "blk/a", 3))
    assert len(astat.pending_plan(al.EXP)[0]) >= 2                 # the journal row and the status update wait for the next push
    jr, _ = _sc.read_journal(host_status, "blk", "a")
    assert any(x["name"] == "ab_m" and x["verdict"] == "accept" and x["accepted"].startswith("да (код влит, ветка autolab/ab_m") for x in jr), jr
    assert "y = 2" in (repo / "harness" / "a.py").read_text() and al.meta("ab_m")["stage"] == "accepted"
    assert "unknown or expired" in handle_callback(42, f"al:ok:{code}")["text"]            # one-time
    # with a key: the same patch is applied on the clone of the GitHub master and pushed
    (tmp / "key").write_text("dummy")
    os.environ["AUTOLAB_DEPLOY_KEY"] = str(tmp / "key")
    g("checkout", "-b", "autolab/ab_m3")
    (repo / "harness" / "b.py").write_text("z = 3\n")
    g("add", "-A")
    g("commit", "-m", "second")
    sha3 = g("rev-parse", "--short", "HEAD").stdout.strip()
    g("checkout", "main")
    al.set_meta("ab_m3", stage="applied", branch="autolab/ab_m3", commit=sha3, target_cells=["blk/a", "blk/_all"], verdict={"b": "accept"})
    r = handle(42, "/autolab accept ab_m3 force")
    assert "рецензии нет" in r["text"] and "есть" in r["text"]
    out = handle_callback(42, r["buttons"][0][0]["data"])["text"]
    assert "запушено" in out, out
    chk = subprocess.run(["git", "-C", str(bare), "log", "--oneline", "master"], capture_output=True, text=True).stdout
    assert "autolab accept ab_m3" in chk and "github base" in chk
    assert "z = 3" in subprocess.run(["git", "-C", str(bare), "show", "master:harness/b.py"], capture_output=True, text=True).stdout
    assert "status: ab_m3" in chk                                   # the status goes in its own commit with the merge
    cellmd = subprocess.run(["git", "-C", str(bare), "show", "master:status/blocks/blk/a.md"], capture_output=True, text=True, encoding="utf-8").stdout
    assert "ab_m3" in cellmd and "ab_m:" in cellmd and "автор: autolab" in cellmd      # this accept and the earlier one that could not be pushed
    assert "| ab_m3 |" in cellmd and "| accept |" in cellmd and "Журнал гипотез" in cellmd      # the journal rows travel with the status commit
    assert "ab_m3" in subprocess.run(["git", "-C", str(bare), "show", "master:status/blocks/blk/_all.md"], capture_output=True, text=True, encoding="utf-8").stdout
    assert not astat.pending_plan(al.EXP)[0] and "запушено" in out and "ab_m3" in (host_status / "blocks" / "blk" / "a.md").read_text(encoding="utf-8")
    # a patch that no longer applies to GitHub master is reported, not forced
    g("checkout", "-b", "autolab/ab_m4")
    (repo / "harness" / "a.py").write_text("x = 1\ny = 2\nw = 4\n")
    g("commit", "-am", "third")
    sha4 = g("rev-parse", "--short", "HEAD").stdout.strip()
    g("checkout", "main")
    seed_a = seed / "harness" / "a.py"
    _git(seed, "pull", "origin", "master")
    seed_a.write_text("x = 99\n")
    _git(seed, "commit", "-am", "diverge")
    _git(seed, "push", "origin", "master")
    al.set_meta("ab_m4", stage="applied", branch="autolab/ab_m4", commit=sha4, target_cells=["blk/a"], verdict={"b": "accept"})
    r = handle(42, "принять ab_m4 force")
    out = handle_callback(42, r["buttons"][0][0]["data"])["text"]
    assert "влито" in out and "не применяется" in out, out
    # a branch touching the lab is refused before any button; a moved branch is refused; a dev checkout is refused; a switch-only hypothesis has nothing to merge
    g("checkout", "-b", "autolab/ab_bad")
    (repo / "tools").mkdir()
    (repo / "tools" / "lab.py").write_text("NOISE = 0\n")
    g("add", "-A")
    g("commit", "-m", "bad")
    badsha = g("rev-parse", "--short", "HEAD").stdout.strip()
    g("checkout", "main")
    al.set_meta("ab_bad", stage="applied", branch="autolab/ab_bad", commit=badsha, target_cells=["blk/a"])
    assert "rules" in handle(42, "/autolab accept ab_bad force")["text"]
    al.set_meta("ab_m5", stage="applied", branch="autolab/ab_m", commit="deadbee", target_cells=["blk/a"])
    assert "differs" in handle(42, "/autolab accept ab_m5 force")["text"]
    al.set_meta("ab_sw", stage="verdict", target_cells=["blk/a"], verdict={"b": "clarify"})
    assert "кода для слияния нет" in handle(42, "/autolab accept ab_sw force")["text"]            # a switch-only hypothesis that was not accepted by the lab: nothing to accept
    # a switch-only hypothesis the lab accepted: no merge, only the scheme status is written and pushed
    al.set_meta("ab_sw2", stage="verdict", target_cells=["blk/a"], verdict={"b": "accept"}, threshold_used=7.69,
                result={"a": {"points_diff_vs_first": None}, "b": {"points_diff_vs_first": {"mean": 12.5, "ci95": [8.0, 17.0]}}})
    r = handle(42, "принять ab_sw2 force")
    assert "ЗАПИСАТЬ РЕЗУЛЬТАТ В СХЕМУ" in r["text"] and r["buttons"], r["text"]
    out = handle_callback(42, r["buttons"][0][0]["data"])["text"]
    assert "кода для слияния нет" in out and "запушено" in out and "blk/a обновлены" in out and al.meta("ab_sw2")["stage"] == "accepted", out
    assert "ab_sw2" in subprocess.run(["git", "-C", str(bare), "show", "master:status/blocks/blk/a.md"], capture_output=True, text=True, encoding="utf-8").stdout
    # a repository without a scheme yet gets the host copy whole
    bare2 = tmp / "github2.git"
    subprocess.run(["git", "init", "--bare", "-b", "master", str(bare2)], capture_output=True)
    seed2 = tmp / "seed2"
    subprocess.run(["git", "clone", str(bare2), str(seed2)], capture_output=True)
    (seed2 / "README.md").write_text("x\n")
    for a_ in (("checkout", "-B", "master"), ("add", "-A"), ("commit", "-m", "base2"), ("push", "origin", "master")):
        _git(seed2, *a_)
    os.environ.update(AUTOLAB_PUSH_REMOTE=str(bare2), AUTOLAB_PUSH_CLONE=str(tmp / "push_repo2"))
    astat.events_add(al.EXP, {"name": "ab_sw2", "item": {"cell": "blk/a", "evidence": "ab_sw3: reject", "note": "x"}})
    assert "запушено" in push_master({"name": "ab_sw3", "no_code": True, "cells": ["blk/a"]}, None)
    assert "status/index.json" in subprocess.run(["git", "-C", str(bare2), "ls-tree", "-r", "--name-only", "master"], capture_output=True, text=True).stdout
    os.environ.update(AUTOLAB_PUSH_REMOTE=str(bare), AUTOLAB_PUSH_CLONE=str(tmp / "push_repo"))
    # views of the scheme: block, type, cell (long ones as a document), scheme, priorities; the buttons come back as the same commands
    r = handle(42, "блок blk")
    assert "Блок blk — готовность" in r["text"] and "🔵" in r["text"] and r["buttons"][0][0]["data"] == "al:cmd:cell:blk/a", r
    assert "Нет блока" in handle(42, "блок zzz")["text"]
    assert "готовность" in handle(42, "тип a")["text"] and "🔵" in handle(42, "тип a")["text"] and "Нет типа" in handle(42, "тип zzz")["text"]
    r = handle_callback(42, "al:cmd:cell:blk/a")
    assert "blk × a" in r["text"] and "🔵" in r["text"] and "🕓 История" in r["text"] and r["document"].endswith("blk_a.md") and "ab_m3" in Path(r["document"]).read_text(encoding="utf-8"), r["text"][:80]
    (host_status / "blocks" / "blk" / "_all.md").write_text("# blk × _all\n\n- Статус: works-unexplored\n- Применимость: да\n\n## Измерено\n" + "длинная строка\n" * 400, encoding="utf-8")
    r = handle(42, "описание blk общее")
    assert r["document"].endswith("blk__all.md") and Path(r["document"]).exists() and "полный текст" in r["text"] and len(r["text"]) < 1200, r["text"][-120:]
    assert "Какая ячейка" in handle(42, "ячейка blk")["text"] and "Нет ячейки" in handle(42, "ячейка blk zzz")["text"]
    r = handle(42, "блоки")
    assert "Блоки схемы" in r["text"] and r["buttons"][0][0]["data"].startswith("al:cmd:block:") and "%" in r["buttons"][0][0]["label"], r
    assert "Блоки схемы" in handle(42, "блок")["text"]
    r = handle(42, "типы")
    assert r["buttons"][0][0]["data"] == "al:cmd:type:a" and "Типы задач" in r["text"], r
    assert "готовность" in handle_callback(42, r["buttons"][0][0]["data"])["text"] and "Типы задач" in handle(42, "тип")["text"]
    assert handle(7, "блоки") is None
    r = handle(42, "журнал blk a 1")
    assert "Журнал blk/a" in r["text"] and "показаны последние 1" in r["text"] and r["document"].endswith("journal_blk_a.md") and "ab_m3" in Path(r["document"]).read_text(encoding="utf-8"), r["text"]
    assert r["buttons"][0][0]["data"] == "al:cmd:cell:blk/a" and "ab_m" in handle(42, "история blk/a 50")["text"] and not handle(42, "история blk/a 50")["document"]
    assert "Какая ячейка" in handle(42, "журнал blk")["text"] and "Нет ячейки" in handle(42, "журнал blk zzz")["text"]
    assert "Журнал blk" in handle_callback(42, "al:cmd:journal:blk/a")["text"] and handle(7, "журнал blk a") is None
    assert handle(42, "ячейка blk a")["buttons"][0][0]["data"] == "al:cmd:journal:blk/a"
    r = handle(42, "схема")
    assert "blk" in r["text"] and "Кружки по порядку" in r["text"] and "🔵" in r["text"] and "%" in r["text"], r["text"]
    assert "Что делать в первую очередь" in handle(42, "приоритеты")["text"]
    assert handle(7, "схема") is None and handle(7, "блок blk") is None
    al.repo_root = lambda: tmp / "not_autolab_repo"
    (tmp / "not_autolab_repo").mkdir()
    assert "host copy" in handle(42, "/autolab accept ab_m force")["text"]
    # report and review through the chat (the review is a spending command: estimate and button first; the work itself is a subprocess, mocked)
    al.set_meta("ab_x_test", threshold_used=7.69, verdict={"b": "clarify"}, result={"a": {"points_per_run": 100.0, "tokens_per_run": 20000, "accuracy": 0.7, "points_diff_vs_first": None},
                "b": {"points_per_run": 104.0, "tokens_per_run": 19000, "accuracy": 0.7, "points_diff_vs_first": {"mean": 4.0, "ci95": [-5.0, 13.0], "threshold": 7.69}}})
    rr = handle(42, "отчёт ab_x_test")
    assert rr["document"].endswith("ab_x_test.report.md") and Path(rr["document"]).exists() and "Вердикт" in rr["text"]
    assert "нет вердикта" in handle(42, "рецензия ab_nope")["text"]
    r = handle(42, "рецензия ab_x_test")
    assert "один запрос к модели" in r["text"] and not [a for a in spawned if a and a[0] == "review"]
    c_rev = r["buttons"][0][0]["data"].split(":")[2]
    spawned.clear()
    globals()["_spawn"] = lambda argv, tag: spawned.append(argv) or "started (mock)"
    assert "started" in handle_callback(42, f"al:ok:{c_rev}")["text"] and spawned[0] == ["review", "ab_x_test", "--notify=42"]
    for k in ("AUTOLAB_PUSH_REMOTE", "AUTOLAB_PUSH_CLONE", "AUTOLAB_DEPLOY_KEY"):
        os.environ.pop(k, None)
    # 5b. group chat: only listed groups, only whitelisted users; the menu, whoami, buttons that act like commands
    _allow_file().write_text('{"user_ids": [42], "chat_ids": [-100500], "report_chats": [-100500]}')
    assert allowed(42, -100500, "supergroup") and not allowed(42, -999, "supergroup") and not allowed(7, -100500, "supergroup") and allowed(42, 42, "private")
    assert handle(42, "/autolab status", -999, "supergroup") is None and handle(7, "статус", -100500, "supergroup") is None
    m = handle(42, "/autolab", -100500, "supergroup")
    assert any(b["data"] == "al:cmd:status" for row in m["buttons"] for b in row), m
    assert handle(42, "меню", -100500, "supergroup")["buttons"]
    assert "Очередь" in str(handle_callback(42, "al:cmd:queue", -100500, "supergroup")) or handle_callback(42, "al:cmd:queue", -100500, "supergroup")["text"]
    assert handle_callback(7, "al:cmd:status", -100500, "supergroup") is None and handle_callback(42, "al:cmd:status", -999, "supergroup") is None
    c = handle_callback(42, "al:cmd:cycle:2", -100500, "supergroup")   # a spending button still asks first
    assert "Код" in c["text"] and c["buttons"][0][0]["data"].startswith("al:ok:")
    r = handle(42, "рецензия ab_x_test", -100500, "supergroup")
    spawned.clear()
    assert handle_callback(42, r["buttons"][0][0]["data"], -100500, "supergroup")["text"].startswith("started") and spawned[0][-1] == "--notify=-100500", spawned
    assert "42" in whoami(7, -100500, "supergroup")["text"] or "7" in whoami(7, -100500, "supergroup")["text"]
    _allow_file().write_text('{"user_ids": [42, "43"]}')
    # 5c. autopilot from the chat: status, one-button switch on, off at once
    r = handle(42, "автопилот")
    assert "выключен" in r["text"] and r["buttons"][0][0]["data"] == "al:cmd:autopilot:вкл", r
    r = handle(42, "автопилот вкл")
    assert "Включить автопилот" in r["text"] and not al.autopilot_flag().exists()
    assert "another user" in handle_callback(43, r["buttons"][0][0]["data"])["text"] and not al.autopilot_flag().exists()
    assert "включён" in handle_callback(42, r["buttons"][0][0]["data"])["text"] and al.autopilot_flag().exists()
    assert "включён" in handle(42, "автопилот")["text"]
    # the chief: a human switches it on with a confirmation; plan, digest, status are views; off is immediate
    r = handle(42, "шеф")
    assert "Шеф: выключен" in r["text"] and ("dry-run" in r["text"] or "боевой" in r["text"]) and r["buttons"][0][0]["data"] == "al:cmd:chief:вкл", r
    r = handle(42, "шеф вкл")
    assert "Включить шефа" in r["text"] and ("dry-run" in r["text"] or "БОЕВОЙ" in r["text"]) and "Код" in r["text"] and not al.chief_flag().exists()
    assert "another user" in handle_callback(43, r["buttons"][0][0]["data"])["text"] and not al.chief_flag().exists()
    assert "Шеф включён" in handle_callback(42, r["buttons"][0][0]["data"])["text"] and al.chief_flag().exists()
    assert "включён" in handle(42, "шеф статус")["text"] and "ещё не написан" in handle(42, "шеф план")["text"] and "Дайджест" in handle(42, "шеф дайджест")["text"]
    assert "chief is on" in al.cmd_autopilot()
    assert "Шеф выключен" in handle(42, "шеф выкл")["text"] and not al.chief_flag().exists() and handle(7, "шеф вкл") is None
    # ideas and questions for the chief
    assert "Формат" in handle(42, "шеф идея")["text"] and "Формат" in handle(42, "шеф вопрос")["text"] and "Идей пока нет" in handle(42, "шеф идеи")["text"]
    r = handle(42, "шеф идея Попробовать ранний выход для задач search при уверенном первом ответе")
    assert "Идея #1" in r["text"] and "выключен" in r["text"] and "ранний выход" in handle(42, "шеф идеи")["text"] and handle(7, "шеф идея Попробовать что-нибудь такое") is None
    assert "выключен" in handle(42, "автопилот выкл")["text"] and not al.autopilot_flag().exists()
    assert handle(7, "автопилот вкл") is None
    # 5d. local data collection switch (a flag + the old stop file), confirmation needed to turn it on
    al.LAB = tmp / "lab"
    r = handle(42, "локально")
    assert "выключен" in r["text"] and r["buttons"][0][0]["data"] == "al:cmd:localdata:вкл", r
    (al.LAB).mkdir(parents=True, exist_ok=True)
    (al.LAB / "STREAM_STOP_local").write_text("old")
    r = handle(42, "локально вкл")      # free (local Qwen): switched on at once
    assert "локальной Qwen" in r["text"] and "включён" in r["text"] and al.localdata_flag().exists()
    assert al.localdata_flag().exists() and not (al.LAB / "STREAM_STOP_local").exists()
    assert "выключен" in handle(42, "локально выкл")["text"] and not al.localdata_flag().exists() and (al.LAB / "STREAM_STOP_local").exists()
    assert handle(7, "локально вкл") is None
    # 5e. the hypotheses tab: grouped by outcome, a cell and a result per line
    NL = chr(10)
    for nm, extra in (("ab_l_acc", {"verdict": {"b": "accept"}, "result": {"a": {"points_diff_vs_first": None}, "b": {"points_diff_vs_first": {"mean": 20.0, "ci95": [9.0, 31.0]}}}, "threshold_used": 7.69}),
                      ("ab_l_cl", {"verdict": {"b": "clarify"}, "result": {"a": {"points_diff_vs_first": None}, "b": {"points_diff_vs_first": {"mean": 2.0, "ci95": [-5.0, 9.0]}}}, "threshold_used": 7.69}),
                      ("ab_l_acc2", {"verdict": {"b": "accept"}, "result": {"a": {"points_diff_vs_first": None}, "b": {"points_diff_vs_first": {"mean": 30.0, "ci95": [12.0, 48.0]}}}, "threshold_used": 7.69}),
                      ("ab_l_dec", {"verdict": {"b": "clarify"}, "queued_at": time.time() + 1, "result": {"a": {"points_diff_vs_first": None}, "b": {"points_diff_vs_first": {"mean": 5.0, "ci95": [-1.0, 11.0]}}}, "threshold_used": 7.69}),
                      ("ab_l_post", {"decision": "postponed"}), ("ab_l_fail", {}), ("ab_l_run", {}), ("ab_l_draft", {"stage": "proposed"}), ("ab_l_skip", {"stage": "filtered_by_sim", "sim_reason": "симулятор: выигрыша нет"})):
        (al.DRAFTS / f"{nm}.yaml").write_text(yaml.safe_dump({"name": nm, "hypothesis": "Гипотеза про " + nm + " [ячейка схемы: blk/a; почему: x]", "arms": {}}, allow_unicode=True), encoding="utf-8")
        al.set_meta(nm, target_cells=["blk/a"], **extra)
    (al.EXP / "queue" / "done").mkdir(parents=True, exist_ok=True)
    (al.EXP / "queue" / "done" / "ab_l_fail.yaml").write_text("name: ab_l_fail" + NL + "# ---- written by lab.py ----" + NL + "status: aborted" + NL + "reason: шлюз остановлен" + NL, encoding="utf-8")
    (al.EXP / "queue" / "pending").mkdir(parents=True, exist_ok=True)
    (al.EXP / "queue" / "pending" / "ab_l_run.yaml").write_text("name: ab_l_run" + NL, encoding="utf-8")
    act = handle(42, "гипотезы")
    assert "Активные гипотезы: 2" in act["text"] and "ab_l_dec" in act["text"] and "ab_l_run" in act["text"] and "ab_l_acc" not in act["text"] and "ab_l_post" not in act["text"], act["text"]
    assert [x["data"] for row in act["buttons"] for x in row][:2] == ["al:cmd:hy:ab_l_dec", "al:cmd:hy:ab_l_run"] or {"al:cmd:hy:ab_l_dec", "al:cmd:hy:ab_l_run"} <= {x["data"] for row in act["buttons"] for x in row}
    assert {"al:cmd:journal", "al:cmd:postponed", "al:cmd:list:все"} <= {x["data"] for row in act["buttons"] for x in row}
    jr = handle(42, "журнал")["text"]
    assert "Журнал гипотез" in jr
    t = handle(42, "гипотезы все")["text"]
    for want in ("🔄 Идут или ждут очереди (держат типы) — ", "🔔 Ждут вашего решения", "🔒 Заняты: a (ab_l_dec)", "⏸ Отложены", "ожидаемый выигрыш", "✅ Лучше эталона (accept) — ", "⚖️ Неясно", "⛔ Без результата", "📝 Придуманы", "🚫 Отсеяны", "ячейка: blk/a", "b: +20.0 очков [9.0; 31.0], порог 7.69",
                 "в очереди", "шлюз остановлен", "Гипотеза про ab_l_draft", "симулятор: выигрыша нет"):
        assert want in t, (want, t)
    assert "[ячейка схемы" not in t and t.index("🔔") < t.index("🔄") < t.index("✅") < t.index("⚖️") < t.index("⛔") < t.index("📝") < t.index("🚫") < t.index("⏸ Отложены")
    assert t.index("ab_l_acc2") < t.index("ab_l_acc" + NL), "the biggest gain first inside a group"
    lr = handle(42, "список")
    flat = [x for row in lr["buttons"] for x in row]
    assert any(x["data"] == "al:cmd:hy:ab_l_dec" for x in flat) and any(x["data"] == "al:cmd:postponed" and "(" in x["label"] for x in flat)
    assert "ab_l_post" in handle(42, "отложенные")["text"] and "ab_l_post" in handle(42, "список все")["text"]
    for nm in ("ab_l_acc", "ab_l_acc2", "ab_l_dec", "ab_l_post", "ab_l_cl", "ab_l_fail", "ab_l_run", "ab_l_draft", "ab_l_skip"):
        (al.DRAFTS / f"{nm}.yaml").unlink()
        (al.DRAFTS / f"{nm}.meta.json").unlink()
    (al.EXP / "queue" / "done" / "ab_l_fail.yaml").unlink()
    (al.EXP / "queue" / "pending" / "ab_l_run.yaml").unlink()
    # 6. odd input never raises and never reaches a shell
    for t in ("", "/autolab", "/autolab report ../../etc/passwd", "/autolab verdict x;rm -rf", "/autolab rerun a b", "/autolab merge", "/autolab ok", "/autolab zzz"):
        assert handle(42, t) is not None
    assert "Формат" in handle(42, "/autolab report ../../etc/passwd")["text"]
    # 7. the journal has every decision and no secret-looking material
    j = (CFG_DIR / "tg_journal.jsonl").read_text()
    assert '"event": "decision"' in j and '"event": "denied"' in j and "sk-ant" not in j
    print("autolab_ctl selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        print(__doc__)
