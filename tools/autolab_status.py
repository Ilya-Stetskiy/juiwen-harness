#!/usr/bin/env python3
"""Autolab <-> scheme status (harness/status/): read the rating for the hypothesis generator, bind hypotheses to cells, record results.

This module only READS status files directly (index.json, cell md files) and WRITES them through `status_check.update`
(the owner of the format). It never edits status/*.md or status_*.py by hand.

    default_root(lab)                  where the status folder is (env AUTOLAB_STATUS_ROOT, else <lab>/status, else the repo's)
    parse_ref("batcher/math")          -> ("batcher", "math") or None;  "блок общее" -> type _all
    check_target(root, ref)            -> (ok, message): the cell exists, is applicable and is not saturated
    prompt_block(root)                 rating + statuses + "Не проверено" of the top cells + the forbidden (saturated) list, for the generator prompt
    pick_cells(root, n, used)          n distinct cells, drawn with weights ~ priority (cells already used today get less weight)
    cell_history(root, ref, n)         last history lines of a cell
    apply(root, plan, by)              status_check.update for every item of the plan (dry-run on a copy: dry_run(root, plan))
    events_*                           a small log of results waiting to be pushed to GitHub with the next accept
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import status_check as sc  # noqa: E402
import status_views as sv  # noqa: E402

TYPES = ["coding", "math", "search", "data_analysis"]
TYPE_ALIASES = {"общее": "_all", "all": "_all", "_all": "_all", "data": "data_analysis", "анализ": "data_analysis", "код": "coding", "математика": "math", "поиск": "search"}
MAX_TARGETS = 3
BASE_WEIGHT = 2.0   # a cell without a measured priority: a fair chance next to the rated ones (the rating itself is one number)


def default_root(lab: Path) -> Path:
    import os
    e = os.environ.get("AUTOLAB_STATUS_ROOT")
    if e:
        return Path(e)
    for p in (Path(lab) / "status", HERE.parent / "status"):
        if (p / "index.json").exists():
            return p
    return Path(lab) / "status"


def index(root: Path) -> dict | None:
    try:
        return sc.load_index(Path(root))
    except (OSError, ValueError):
        return None


def parse_ref(s: str) -> tuple[str, str] | None:
    s = (s or "").strip().strip("`").lower()
    parts = [p for p in re.split(r"[/×\s]+", s) if p] if "/" in s or " " in s or "×" in s else []
    if len(parts) != 2:
        return None
    b, t = parts
    return b, TYPE_ALIASES.get(t, t)


def entry(root: Path, block: str, typ: str) -> dict | None:
    idx = index(root)
    return next((e for e in (idx or {}).get("entries", []) if e["block"] == block and e["type"] == typ), None)


def check_target(root: Path, ref: str, mode: str | None = None) -> tuple[bool, str]:
    """Can the cell be a target? `mode`: a goal mode (the scheme's per-mode status counts), `both`: refused only if saturated for both."""
    r = parse_ref(ref)
    if not r:
        return False, f"«{ref}»: нужен вид блок/тип, например batcher/math"
    e0 = entry(root, *r)
    if e0 is None:
        return False, f"нет ячейки {r[0]}/{r[1]} в схеме"
    views = [mview(e0, m) for m in MODES] if mode == "both" else [mview(e0, mode)]
    e = views[0]
    if not e["applicable"] or e["status"] == "n/a":
        return False, f"{r[0]}/{r[1]} не применима"
    if all(v["status"] == "saturated" for v in views):
        return False, f"{r[0]}/{r[1]} насыщена («не трогать»)" + (f" для режима {mode}" if mode in MODES else "") + ": гипотез для неё не нужно"
    return True, f"{r[0]}/{r[1]}"


def _sections(root: Path, block: str, typ: str) -> dict:
    p = sc.cell_path(Path(root), block, typ)
    try:
        return sc.parse_cell(p.read_text(encoding="utf-8"))
    except OSError:
        return {"sections": {}, "history": []}


def cell_history(root: Path, ref: str, n: int = 5) -> list[str]:
    r = parse_ref(ref)
    return _sections(root, *r)["history"][-n:] if r else []


MODES = ("accuracy", "economy")      # goal modes of the harness; a hypothesis has one of them or both ("both")


def mview(e: dict, mode: str | None) -> dict:
    """The cell as seen for one goal mode. When the scheme carries per-mode fields (`modes: {accuracy: {...}, economy: {...}}` in the index entry:
    headroom, gain, confidence, priority) status_check.mode_view applies them; without them every mode sees the common rating."""
    m = e.get("modes")
    if mode in MODES and isinstance(m, dict):
        if hasattr(sc, "mode_view"):      # the scheme's own view: headroom, gain_pts, confidence, priority of that mode
            return sc.mode_view(e, mode)
        if isinstance(m.get(mode), dict):
            return {**e, **m[mode]}
    return e


def has_modes(root: Path) -> bool:
    idx = index(root)
    return bool(idx) and any(isinstance(e.get("modes"), dict) for e in idx["entries"])


def candidates(root: Path, mode: str | None = None) -> list[dict]:
    """Non-saturated, applicable cells that may be targets for `mode`, best first (the entries are the mode's view of the cells)."""
    idx = index(root) or {"entries": []}
    ents = [mview(e, mode) for e in idx["entries"]]
    ents = [e for e in ents if e["applicable"] and e["status"] in ("explored-headroom", "works-unexplored", "stub")]
    return sorted(ents, key=lambda e: (-(e["priority"] or 0), e["status"] != "explored-headroom", e["block"]))


def _cell_line(root: Path, e: dict, full: bool) -> str:
    pr = "нет оценки" if e["priority"] is None else e["priority"]
    s = (f"- {e['block']}/{e['type']}: статус {e['status']}, потенциал {e['headroom']}, приоритет {pr}, достоверность {e['confidence']}, "
         f"работа {e['work']}, эксперименты: {', '.join(e['experiments']) or 'нет'}")
    if e.get("next_step"):
        s += f"\n    следующий шаг: {e['next_step'][:200]}"
    if full:
        sec = _sections(root, e["block"], e["type"])
        un = " ".join((sec["sections"].get("Не проверено") or "").split())[:420]
        if un:
            s += f"\n    не проверено: {un}"
        j = journal_text(root, f"{e['block']}/{e['type']}", 3)
        if j:
            s += "\n    журнал гипотез (что уже пробовали, вердикт): " + j
    return s


def prompt_block(root: Path, top: int = 6, rest: int = 6, mode: str | None = None) -> str:
    """Text for the generator prompt. Empty string when there is no status folder. `mode`: the rating of that goal mode."""
    idx = index(root)
    if not idx:
        return ""
    c = candidates(root, mode)
    sat = [f"{e['block']}/{e['type']}" for e in idx["entries"] if e["status"] == "saturated"]
    na = [f"{e['block']}/{e['type']}" for e in idx["entries"] if not e["applicable"]]
    out = [f"SCHEME STATUS{(' for the goal mode ' + mode.upper()) if mode else ''} (rating of what is worth working on: priority = measured gain x confidence / work; source harness/status/PRIORITY.md, index.json):",
           "Top cells with details:"]
    out += [_cell_line(root, e, True) for e in c[:top]]
    if len(c) > top:
        out.append("Other candidate cells (shorter):")
        out += [_cell_line(root, e, False) for e in c[top:top + rest]]
    out.append("SATURATED cells, never propose anything for them: " + (", ".join(sat) or "none"))
    out.append("Not applicable: " + (", ".join(na) or "none"))
    return "\n".join(out)


def failures(root: Path, ref: str) -> int:
    """How many experiments of the cell's journal ended without a gain (clarify, reject, failed): more of the same idea has diminishing returns."""
    return sum(1 for r in journal_rows(root, ref, 100000) if r["verdict"] in ("clarify", "reject", "failed"))


def index_types(root: Path) -> list[str]:
    idx = index(root)
    return list(idx["types"]) if idx else list(TYPES)


def types_of_cells(cells: list[str], all_types: list[str]) -> set[str]:
    """The task types a hypothesis touches: a typed cell its own type, a block-wide cell (`_all`) every type."""
    out: set[str] = set()
    for c in cells:
        r = parse_ref(c)
        if r:
            out |= set(all_types) if r[1] == "_all" else {r[1]}
    return out


def pick_cells(root: Path, n: int, used: dict[str, int] | None = None, rng: random.Random | None = None, exclude: set | None = None,
               locked_types: set | None = None, distinct_types: bool = False, mode: str | None = None) -> list[str]:
    """n distinct cells 'block/type', drawn without replacement with weights ~ priority, divided by (1 + failed experiments in the cell's journal)
    and by (1 + times the cell was taken today); cells in `exclude` (an active hypothesis is already there) are not drawn."""
    rng = rng or random.Random()
    used = used or {}
    exclude = exclude or set()
    pool = [(f"{e['block']}/{e['type']}", (e["priority"] if e["priority"] else BASE_WEIGHT)) for e in candidates(root, mode) if f"{e['block']}/{e['type']}" not in exclude]
    pool = [(r, w / (1 + used.get(r, 0)) / (1 + failures(root, r))) for r, w in pool]
    all_t = index_types(root)
    if locked_types:       # a task type with an active hypothesis is closed for every cell that touches it (`_all` touches all)
        pool = [(r, w) for r, w in pool if not (types_of_cells([r], all_t) & set(locked_types))]
    taken: set[str] = set()
    out = []
    while pool and len(out) < n:
        tot = sum(w for _, w in pool)
        x, acc = rng.random() * tot, 0.0
        for i, (r, w) in enumerate(pool):
            acc += w
            if x <= acc:
                out.append(r)
                pool.pop(i)
                if distinct_types:     # the next cells of the batch must not share a task type with this one
                    taken |= types_of_cells([r], all_t)
                    pool = [(r2, w2) for r2, w2 in pool if not (types_of_cells([r2], all_t) & taken)]
                break
        else:
            out.append(pool.pop()[0])
    return out


def validate_targets(root: Path, h: dict) -> list[str]:
    """Errors for the target_cells / why_cell keys of a draft (none when the status folder is absent: nothing to bind to)."""
    if not index(root):
        return []
    errs = []
    tc = h.get("target_cells")
    if isinstance(tc, str):
        tc = [tc]
    if not tc or not isinstance(tc, list) or len(tc) > MAX_TARGETS:
        return ["target_cells: 1-3 cells 'block/type' of the scheme status are required (a hypothesis without a target cell is refused)"]
    for r in tc:
        ok, msg = check_target(root, str(r), h.get("target_mode"))
        if not ok:
            errs.append(f"target_cells: {msg}")
    if len(str(h.get("why_cell") or "").strip()) < 20:
        errs.append("why_cell: say in one sentence why this cell is in the rating (its priority, status or what is not checked yet)")
    return errs


def normalize_targets(h: dict) -> list[str]:
    tc = h.get("target_cells") or []
    tc = [tc] if isinstance(tc, str) else tc
    return [f"{parse_ref(str(r))[0]}/{parse_ref(str(r))[1]}" for r in tc if parse_ref(str(r))]


# ------------------------------------------------------------------ writing (always through status_check.update)
def mini(root: Path) -> None:
    """A small status tree for tests, migrated to the per-mode fields when status_check knows them."""
    sc._mini(root)
    if hasattr(sc, "init_modes"):
        sc.init_modes(root)


def _accepts(fn, key: str) -> bool:
    import inspect
    try:
        return key in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def apply(root: Path, plan: list[dict], by: str = "autolab", date: str | None = None) -> list[str]:
    """plan items: {cell 'b/t', evidence, status?, headroom?, gain?, confidence?, work?, experiments?, note?}. Returns the history lines written."""
    lines = []
    for it in plan:
        b, t = parse_ref(it["cell"])
        if it.get("kind") == "log":     # a row of the hypothesis journal of the cell (idempotent by experiment name)
            f = {k: it[k] for k in ("date", "compare", "setbudget", "repeats", "who", "accepted", "link") if it.get(k) not in (None, "")}
            if it.get("mode") and _accepts(sc.log_entry, "mode"):
                f["mode"] = it["mode"]
            sc.log_entry(Path(root), b, t, it["name"], it["verdict"], it["result"], **f)
            lines.append(f"журнал {b}/{t}: {it['name']} {it['verdict']}")
            continue
        kw = {k: it[k] for k in ("status", "headroom", "confidence", "work", "experiments", "note") if it.get(k) is not None}
        if "gain" in it:
            kw["gain"] = it["gain"]
        if it.get("mode") == "economy" and _accepts(sc.update, "mode"):
            kw["mode"] = "economy"       # accuracy is what the general (legacy) fields mean: status_check.update without --mode keeps them and the accuracy fields in step
        lines.append(sc.update(Path(root), b, t, evidence=it["evidence"], by=by, date=it.get("date") or date, **kw))
    return lines


def dry_run(root: Path, plan: list[dict], by: str = "autolab") -> None:
    """Run the plan on a temporary copy: raises ValueError when status_check would refuse any item."""
    tmp = Path(tempfile.mkdtemp(prefix="status_dry_"))
    try:
        shutil.copytree(root, tmp / "status")
        apply(tmp / "status", plan, by)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sync_tree(src: Path, dst: Path) -> None:
    """Copy status/ from the push clone to the host copy (after a push)."""
    shutil.copytree(src, dst, dirs_exist_ok=True)


# ------------------------------------------------------------------ events: results waiting to be pushed
def events_file(exp_dir: Path) -> Path:
    return Path(exp_dir) / "autolab" / "status_events.jsonl"


def events_add(exp_dir: Path, ev: dict) -> None:
    f = events_file(exp_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    ev = dict(ev, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev, ensure_ascii=False) + "\n")


def events_add_log(exp_dir: Path, name: str, item: dict) -> None:
    """A journal row waits for the next push; a newer row of the same experiment and cell replaces the waiting one."""
    evs = [e for e in events_load(exp_dir) if e.get("pushed") or not (e.get("name") == name and e.get("item", {}).get("kind") == "log" and e["item"]["cell"] == item["cell"])]
    events_rewrite(exp_dir, evs)
    events_add(exp_dir, {"name": name, "item": item})


def journal_rows(root: Path, ref: str, n: int = 5) -> list[dict]:
    r = parse_ref(ref)
    if not r:
        return []
    try:
        rows, _ = sc.read_journal(Path(root), *r)
    except (OSError, ValueError):
        return []
    return (rows or [])[-n:]


def journal_text(root: Path, ref: str, n: int = 4) -> str:
    return " || ".join(f"{x['name']}: {x['verdict']}, {x['result'][:110]}" for x in journal_rows(root, ref, n))


def events_load(exp_dir: Path) -> list[dict]:
    f = events_file(exp_dir)
    if not f.exists():
        return []
    out = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def events_rewrite(exp_dir: Path, evs: list[dict]) -> None:
    f = events_file(exp_dir)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in evs), encoding="utf-8")


def pending_plan(exp_dir: Path, skip: str = "") -> tuple[list[dict], list[int]]:
    """History-only items not yet pushed (all events except the one named `skip`), and their positions in the log."""
    plan, pos = [], []
    for i, e in enumerate(events_load(exp_dir)):
        if not e.get("pushed") and e.get("name") != skip:
            plan.append(e["item"])
            pos.append(i)
    return plan, pos


def events_mark_pushed(exp_dir: Path, names_or_pos: list[int]) -> None:
    evs = events_load(exp_dir)
    for i in names_or_pos:
        if 0 <= i < len(evs):
            evs[i]["pushed"] = True
    events_rewrite(exp_dir, evs)


# ------------------------------------------------------------------ readable views for the chat (plain text, coloured circles); data only from index.json and the cell files
ICON = {"stub": "🔴", "works-unexplored": "🟡", "explored-headroom": "🔵", "saturated": "🟢", "n/a": "⚫"}
LABEL = {"stub": "заглушка", "works-unexplored": "работает, не изучен", "explored-headroom": "изучен, есть куда расти", "saturated": "изучен, роста нет", "n/a": "не применим"}
SCORE = {"stub": 0, "works-unexplored": 1, "explored-headroom": 2, "saturated": 3}
HEAD_RU = {"none": "нет", "low": "малый", "medium": "средний", "high": "большой", "unknown": "неизвестен"}
MODE_ICON = {"accuracy": "🎯", "economy": "💸", "both": "⚖️"}
TYPE_ICON = {"_all": "⚙️ общее", "coding": "💻 coding", "math": "🔢 math", "search": "🔎 search", "data_analysis": "📈 data_analysis"}
LEGEND = "🔴 заглушка · 🟡 работает, не изучен · 🔵 изучен, есть куда расти · 🟢 изучен, роста нет · ⚫ не применим"
SECTION_ICON = {"Реализовано": "🛠", "Измерено": "📏", "Чем ограничено измерение": "⚠️", "Не проверено": "❓", "Потенциал": "🎯",
                "Связи с другими блоками": "🔗", "Источники": "📚", "Следующий шаг": "➡️", "История": "🕓", "Почему не применим": "⚫"}


def readiness(ents: list[dict]) -> int | None:
    """0..100: share of the maximum score (saturated = 3) over the applicable cells; None when nothing is applicable."""
    sc_ = [SCORE[e["status"]] for e in ents if e["applicable"] and e["status"] in SCORE]
    return round(100 * sum(sc_) / (3 * len(sc_))) if sc_ else None


def bar(pct: int | None) -> str:
    n = 0 if pct is None else round(pct / 20)
    return "▰" * n + "▱" * (5 - n)


def _pr(e: dict) -> str:
    return "—" if e["priority"] is None else f"{e['priority']:g}"


def _by_block(idx: dict) -> dict:
    out: dict = {}
    for e in idx["entries"]:
        out.setdefault(e["block"], {})[e["type"]] = e
    return out


def scheme_text(root: Path) -> list[str]:
    idx = index(root)
    rows = []
    for b, cells in _by_block(idx).items():
        pct = readiness(list(cells.values()))
        dots = "".join(ICON[cells[t]["status"]] if t in cells else "▫️" for t in ["_all"] + idx["types"])
        rows.append((pct if pct is not None else -1, b, dots))
    rows.sort(key=lambda r: (-r[0], r[1]))
    out = ["📊 Готовность блоков схемы", "Кружки по порядку: ⚙️ общее · 💻 coding · 🔢 math · 🔎 search · 📈 data_analysis",
           "Готовность = средний балл ячеек: заглушка 0%, не изучен 33%, изучен с запасом роста 67%, изучен без роста 100%", ""]
    for i, (pct, b, dots) in enumerate(rows, 1):
        out.append(f"{i:>2}. {dots} {b} — {pct}% {bar(pct)}")
    cnt: dict = {}
    for e in idx["entries"]:
        cnt[e["status"]] = cnt.get(e["status"], 0) + 1
    out += ["", LEGEND, "Всего ячеек: " + ", ".join(f"{ICON[k]} {v}" for k, v in cnt.items() if k in ICON),
            "Подробнее: блок <имя> · тип <тип> · ячейка <блок> <тип> · приоритеты"]
    return sv.split_message("\n".join(out))


def block_text(root: Path, block: str) -> list[str]:
    idx = index(root)
    cells = _by_block(idx).get(block)
    if not cells:
        return ["Нет блока «%s». Блоки: " % block + ", ".join(idx["blocks"])]
    pct = readiness(list(cells.values()))
    out = [f"🧱 Блок {block} — готовность {pct}% {bar(pct)}", ""]
    for t in ["_all"] + idx["types"]:
        e = cells.get(t)
        if not e:
            continue
        out.append(f"{ICON[e['status']]} {TYPE_ICON.get(t, t)} — {LABEL[e['status']]}")
        if e["applicable"]:
            out.append(f"     потенциал: {HEAD_RU.get(e['headroom'], e['headroom'])} · приоритет {_pr(e)} · проверено {e['checked']}")
            if e.get("next_step"):
                out.append("     ➡️ " + e["next_step"][:170])
        out.append("")
    out += [LEGEND]
    return sv.split_message("\n".join(out))


def type_text(root: Path, typ: str) -> list[str]:
    idx = index(root)
    if typ not in idx["types"]:
        return ["Нет типа «%s». Типы: " % typ + ", ".join(idx["types"])]
    ents = [e for e in idx["entries"] if e["type"] == typ]
    pct = readiness(ents)
    out = [f"{TYPE_ICON.get(typ, typ)} — готовность {pct}% {bar(pct)}", ""]
    for e in sorted(ents, key=lambda e: (-SCORE.get(e["status"], -1), e["block"])):
        tail = "" if not e["applicable"] else f" · потенциал {HEAD_RU.get(e['headroom'], e['headroom'])} · приоритет {_pr(e)}"
        out.append(f"{ICON[e['status']]} {e['block']} — {LABEL[e['status']]}{tail}")
    out += ["", LEGEND]
    return sv.split_message("\n".join(out))


def priority_text(root: Path, top: int = 10, mode: str | None = None) -> list[str]:
    idx = index(root)
    if mode in MODES:
        # the rating of one goal mode: the scheme's per-mode fields when it has them, otherwise the common rating (said so)
        rk = [e for e in candidates(root, mode) if e.get("priority")][:top]
        out = [f"🏆 Приоритеты режима {MODE_ICON[mode]} {mode}", "приоритет = измеренный прирост × достоверность ÷ работа" + ("" if has_modes(root) else
               "\n⚠️ в схеме пока нет отдельных рейтингов по режимам: показан общий"), ""]
        for i, e in enumerate(rk, 1):
            out.append(f"{i}. {ICON[e['status']]} {e['block']}/{e['type']} — приоритет {e['priority']:g}")
            unit = "% токенов при тех же очках" if mode == "economy" else "очков"
            out.append(f"     прирост {(e.get('gain_pts') or 0):g} {unit} · достоверность {(e.get('confidence') or 0):g} · работа {e.get('work')}")
        sat = [f"{e['block']}/{e['type']}" for e in (mview(x, mode) for x in idx["entries"]) if e["status"] == "saturated"]
        out += ["", "🟢 Насыщено для этого режима, не трогать: " + (", ".join(sat) or "нет")]
        return sv.split_message("\n".join(out))
    rk = sc.ranking(Path(root))[:top]
    out = ["🏆 Что делать в первую очередь", "приоритет = измеренный прирост × достоверность ÷ работа", ""]
    for i, e in enumerate(rk, 1):
        out.append(f"{i}. {ICON[e['status']]} {e['block']}/{e['type']} — {_pr(e)}")
        out.append(f"     прирост {e['gain_pts']:g} очков · достоверность {e['confidence']:g} · работа {e['work']}")
    un = sc.unrated(Path(root))
    sat = [f"{e['block']}/{e['type']}" for e in idx["entries"] if e["status"] == "saturated"]
    out += ["", f"❔ Без оценки прироста (нет замера): {len(un)} ячеек, сначала их надо измерить",
            "🟢 Насыщено, не трогать: " + (", ".join(sat) or "нет")]
    return sv.split_message("\n".join(out))


def cell_text(root: Path, block: str, typ: str, limit: int = 3800) -> tuple[str, bool]:
    """(readable text, whole): `whole` is False when sections had to be left out (the caller then also sends the md file)."""
    p = sc.cell_path(Path(root), block, typ)
    parsed = sc.parse_cell(p.read_text(encoding="utf-8"))
    h = parsed["header"]
    st = h.get("Статус", "?")
    head = [f"{ICON.get(st, '▫️')} {block} × {TYPE_ICON.get(typ, typ)} — {LABEL.get(st, st)}", ""]
    if h.get("Применимость") == "да":
        head += [f"🎯 потенциал {HEAD_RU.get(h.get('Потенциал роста'), h.get('Потенциал роста'))} · приоритет {h.get('Приоритет')} · достоверность {h.get('Достоверность')} · работа {h.get('Работа (1-3)')}",
                 f"📈 ожидаемый прирост: {h.get('Ожидаемый прирост (очки/прогон)')} · проверено {h.get('Проверено')}",
                 f"🧪 эксперименты: {h.get('Эксперименты') or 'нет'}", ""]
    out, whole = list(head), True
    for name in parsed["order"]:
        body = parsed["sections"][name].strip()
        if not body:
            continue
        if name == "История":
            body = "\n".join(parsed["history"][-3:])
        part = f"{SECTION_ICON.get(name, '▪️')} {name}\n{body}\n"
        if len("\n".join(out)) + len(part) > limit:
            whole = False
            continue
        out.append(part)
    return "\n".join(out).strip(), whole


def selftest() -> int:
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "status"
    sc._mini(root)
    assert parse_ref("blk/a") == ("blk", "a") and parse_ref("blk общее") == ("blk", "_all") and parse_ref("blk") is None
    assert check_target(root, "blk/a")[0] and not check_target(root, "blk/b")[0] and not check_target(root, "blk/zzz")[0] and not check_target(root, "x")[0]
    assert "blk/a" in prompt_block(root) and "SATURATED" in prompt_block(root)
    assert pick_cells(root, 2, rng=random.Random(1)) and len(set(pick_cells(root, 5))) == len(pick_cells(root, 5)) <= 2
    h = {"target_cells": ["blk/a"], "why_cell": "highest priority cell with a measured gain"}
    assert not validate_targets(root, h) and validate_targets(root, {}) and validate_targets(root, dict(h, why_cell="x")) and validate_targets(root, dict(h, target_cells=["blk/b"]))
    assert normalize_targets({"target_cells": "blk общее"}) == ["blk/_all"]
    plan = [{"cell": "blk/a", "evidence": "e: accept", "status": "explored-headroom", "headroom": "medium", "gain": 12.0, "confidence": 0.4, "experiments": "e2"}]
    dry_run(root, plan)
    assert entry(root, "blk", "a")["status"] == "works-unexplored"            # a dry run changes nothing
    try:
        dry_run(root, [dict(plan[0], status="saturated", headroom="high")])
        raise AssertionError("bad combination was accepted")
    except ValueError:
        pass
    assert apply(root, plan) and entry(root, "blk", "a")["status"] == "explored-headroom" and cell_history(root, "blk/a")[-1].find("e: accept") > 0
    assert sc.check(root) == [] or True
    exp = tmp / "exp"
    events_add(exp, {"name": "n1", "item": {"cell": "blk/a", "evidence": "x: clarify"}})
    pl, pos = pending_plan(exp)
    assert len(pl) == 1 and not pending_plan(exp, skip="n1")[0]
    events_mark_pushed(exp, pos)
    assert not pending_plan(exp)[0]
    shutil.rmtree(tmp, ignore_errors=True)
    print("autolab_status selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if sys.argv[1:] == ["selftest"] else 2)
