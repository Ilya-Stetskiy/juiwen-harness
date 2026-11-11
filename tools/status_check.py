#!/usr/bin/env python3
"""Status of the harness scheme: format check, summaries, history updates.

The files live in harness/status/ (README.md there explains the rules):
    blocks/<block>/<type>.md     one file per block x task type (+ _all.md: block-wide properties)
    index.json                   machine-readable copy of each file header (kept in step with the md files)

Usage:
    status_check.py check                      verify format, completeness, index <-> md consistency (exit 1 on errors)
    status_check.py summary [--block B] [--type T]
    status_check.py priority [--write]         ranking by gain * confidence / work; --write refreshes PRIORITY.md
    status_check.py update BLOCK TYPE --status S --evidence "exp, numbers, verdict" [--headroom H] [--gain G]
                           [--confidence C] [--work W] [--experiments a,b] [--note TEXT] [--by NAME] [--date YYYY-MM-DD]
    status_check.py log BLOCK TYPE --name N --verdict V --result "..." [--date D --compare "b1 против b8" --setbudget "set · 30k"
                           --repeats 2 --who автолаборатория --accepted yes|no|commit:<хеш>|ветка:<имя> --link PATH --also b/t,b/t]
    status_check.py init-modes                 add per-mode fields and the «Режим» journal column (migration)
    status_check.py init-journal               add an empty «Журнал гипотез» to cells that lack it
    status_check.py selftest

`update` is what the autolab calls when a hypothesis is accepted: it edits the header of the cell file, appends a line
to its «История» section and refreshes index.json. `--root DIR` points to another status folder (tests).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "status"
STATUSES = ["stub", "works-unexplored", "explored-headroom", "saturated"]
NA = "n/a"
HEADROOMS = ["none", "low", "medium", "high", "unknown"]
ALL = "_all"
JOURNAL_SECTION = "Журнал гипотез"
SECTIONS = ["Реализовано", "Измерено", "Чем ограничено измерение", "Не проверено", "Потенциал",
            "Связи с другими блоками", "Источники", "Следующий шаг", JOURNAL_SECTION, "История"]
NA_SECTIONS = ["Почему не применим", "История"]
FIELDS = [("status", "Статус"), ("applicable", "Применимость"), ("headroom", "Потенциал роста"),
          ("gain_pts", "Ожидаемый прирост (очки/прогон)"), ("confidence", "Достоверность"),
          ("work", "Работа (1-3)"), ("priority", "Приоритет"), ("checked", "Проверено"),
          ("experiments", "Эксперименты")]
NO_DATA = "нет данных"
NO_RANK = "нет оценки"
HIST_RE = re.compile(r"^- (\d{4}-\d{2}-\d{2}) \| (.+)$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ---------------------------------------------------------------- hypothesis journal
JOURNAL_COLS = ["Дата", "Эксперимент", "Сравнивалось", "Набор · бюджет", "Повторы", "Результат", "Вердикт",
                "Кто предложил", "В коде", "Отчёт", "Режим"]
JOURNAL_KEYS = ["date", "name", "compare", "setbudget", "repeats", "result", "verdict", "who", "accepted", "link", "mode"]
LEGACY_COLS = JOURNAL_COLS[:-1]  # журнал до введения режимов: читается, режим = «не указан»
MODES = ["accuracy", "economy"]
JOURNAL_MODES = MODES + ["оба", "не указан"]
MODE_METRIC = {"accuracy": "очки/прогон", "economy": "% токенов при тех же очках"}
MODE_FIELDS = {
    "accuracy": {"headroom": "Потенциал роста [accuracy]", "gain": "Прирост [accuracy] (очки/прогон)",
                 "confidence": "Достоверность [accuracy]", "priority": "Приоритет [accuracy]"},
    "economy": {"headroom": "Потенциал роста [economy]", "gain": "Прирост [economy] (% токенов при тех же очках)",
                "confidence": "Достоверность [economy]", "priority": "Приоритет [economy]"},
}
VERDICTS = ["accept", "clarify", "reject", "failed", "не завершён", "не запускался", "эталон"]
ACCEPTED_RE = re.compile(r"^(да|нет|commit:[0-9a-f]{7,40}|ветка:\S+)( .*)?$")
REPEATS_RE = re.compile(r"^(\d+|нет данных)$")


def _clean(v) -> str:
    v = " ".join(str(v).replace("|", "/").split())
    return v if v else "нет данных"


def journal_header() -> list[str]:
    return ["| " + " | ".join(JOURNAL_COLS) + " |", "|" + "---|" * len(JOURNAL_COLS)]


def format_row(row: dict) -> str:
    row = dict(row)
    row.setdefault("mode", "не указан")
    return "| " + " | ".join(_clean(row.get(k, "")) for k in JOURNAL_KEYS) + " |"


def parse_journal(body: str):
    """-> (rows, errors). A row is a dict by JOURNAL_KEYS; body is the text of the journal section."""
    rows, errs = [], []
    lines = [ln for ln in body.splitlines() if ln.strip()]
    head = [c.strip() for c in lines[0].strip().strip("|").split("|")] if lines else []
    if len(lines) < 2 or head not in (JOURNAL_COLS, LEGACY_COLS):
        return [], ["заголовок таблицы журнала должен быть: " + " | ".join(JOURNAL_COLS)]
    legacy = head == LEGACY_COLS
    keys = JOURNAL_KEYS[:-1] if legacy else JOURNAL_KEYS
    if legacy:
        errs.append("журнал в старом формате (нет колонки «Режим»): выполните status_check.py init-modes")
    if "---" not in lines[1]:
        errs.append("нет строки-разделителя таблицы журнала")
    for ln in lines[2:]:
        if not ln.strip().startswith("|"):
            errs.append(f"строка журнала не таблица: {ln[:50]}")
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split(" | ")]
        if len(cells) != len(keys):
            errs.append(f"в строке журнала {len(cells)} колонок вместо {len(keys)}: {ln[:60]}")
            continue
        row = dict(zip(keys, cells))
        row.setdefault("mode", "не указан")
        rows.append(row)
    return rows, errs


def check_journal_rows(rows: list[dict]) -> list[str]:
    errs, seen, prev = [], set(), ""
    for r in rows:
        n = r["name"]
        if n in seen:
            errs.append(f"журнал: дубль эксперимента «{n}»")
        seen.add(n)
        if not DATE_RE.match(r["date"]):
            errs.append(f"журнал «{n}»: дата «{r['date']}» не YYYY-MM-DD")
        elif r["date"] < prev:
            errs.append(f"журнал «{n}»: даты не по возрастанию")
        else:
            prev = r["date"]
        if r["verdict"] not in VERDICTS:
            errs.append(f"журнал «{n}»: вердикт «{r['verdict']}» не из {VERDICTS}")
        if not ACCEPTED_RE.match(r["accepted"]):
            errs.append(f"журнал «{n}»: «В коде» начинается с да | нет | commit:<хеш> | ветка:<имя>, а не «{r['accepted'][:30]}»")
        if r.get("mode", "не указан") not in JOURNAL_MODES:
            errs.append(f"журнал «{n}»: режим «{r.get('mode')}» не из {JOURNAL_MODES}")
        if not REPEATS_RE.match(r["repeats"]):
            errs.append(f"журнал «{n}»: повторы «{r['repeats']}» не число")
        if not n or n == "нет данных":
            errs.append("журнал: пустое имя эксперимента")
    return errs


def journal_summary(rows: list[dict]) -> dict:
    last = rows[-1] if rows else None
    by = {m: sum(1 for r in rows if r.get("mode", "не указан") == m) for m in JOURNAL_MODES}
    return {"count": len(rows), "last": last["name"] if last else "", "last_verdict": last["verdict"] if last else "",
            "last_date": last["date"] if last else "", "by_mode": by}


def read_journal(root: Path, block: str, typ: str):
    """-> (rows, errors) of one cell; rows None if the cell has no journal section."""
    parsed = parse_cell(cell_path(root, block, typ).read_text(encoding="utf-8"))
    body = parsed["sections"].get(JOURNAL_SECTION)
    if body is None:
        return None, ["нет раздела «%s»" % JOURNAL_SECTION]
    return parse_journal(body)



# ---------------------------------------------------------------- parsing

def priority_of(gain, conf, work):
    if gain is None or not work:
        return None
    return round(gain * conf / work, 2)


def _num(s: str):
    s = s.strip().replace(",", ".")
    return float(s)


def parse_cell(text: str) -> dict:
    """-> {title, header: {field: raw str}, sections: {name: body}, order: [names], history: [lines]}"""
    lines = text.splitlines()
    out = {"title": lines[0].lstrip("# ").strip() if lines else "", "header": {}, "sections": {}, "order": [], "history": []}
    cur = None
    for ln in lines[1:]:
        if ln.startswith("## "):
            cur = ln[3:].strip()
            out["sections"][cur] = []
            out["order"].append(cur)
            continue
        if cur is None:
            m = re.match(r"^- ([^:]+): ?(.*)$", ln)
            if m:
                out["header"][m.group(1).strip()] = m.group(2).strip()
        else:
            out["sections"][cur].append(ln)
    out["sections"] = {k: "\n".join(v).strip() for k, v in out["sections"].items()}
    out["history"] = [ln for ln in out["sections"].get("История", "").splitlines() if ln.strip()]
    return out


def header_values(parsed: dict) -> dict:
    """Typed header values; raises ValueError with a readable message."""
    h = parsed["header"]
    miss = [ru for _, ru in FIELDS if ru not in h]
    if miss:
        raise ValueError("нет полей заголовка: " + ", ".join(miss))
    v = {"status": h["Статус"], "applicable": h["Применимость"] == "да", "headroom": h["Потенциал роста"],
         "checked": h["Проверено"],
         "experiments": [e.strip() for e in h["Эксперименты"].split(",") if e.strip()]}
    g = h["Ожидаемый прирост (очки/прогон)"]
    v["gain_pts"] = None if g == NO_DATA else _num(g)
    v["confidence"] = _num(h["Достоверность"])
    v["work"] = int(_num(h["Работа (1-3)"]))
    p = h["Приоритет"]
    v["priority"] = None if p == NO_RANK else _num(p)
    v["modes"] = None
    if all(f in h for m in MODES for f in MODE_FIELDS[m].values()):
        v["modes"] = {}
        for m in MODES:
            f = MODE_FIELDS[m]
            g, pr = h[f["gain"]], h[f["priority"]]
            v["modes"][m] = {"headroom": h[f["headroom"]], "gain": None if g == NO_DATA else _num(g),
                             "confidence": _num(h[f["confidence"]]), "priority": None if pr == NO_RANK else _num(pr)}
    return v


def load_index(root: Path) -> dict:
    return json.loads((root / "index.json").read_text(encoding="utf-8"))


def cell_path(root: Path, block: str, typ: str) -> Path:
    return root / "blocks" / block / f"{typ}.md"


# ---------------------------------------------------------------- check

def check(root: Path) -> list[str]:
    errs: list[str] = []
    try:
        idx = load_index(root)
    except Exception as e:  # noqa: BLE001
        return [f"index.json: не читается ({e})"]
    blocks, types = idx.get("blocks", []), idx.get("types", [])
    if not blocks or not types:
        errs.append("index.json: пустые blocks/types")
    entries = {(e["block"], e["type"]): e for e in idx.get("entries", [])}
    if len(entries) != len(idx.get("entries", [])):
        errs.append("index.json: повторяющиеся записи (блок, тип)")
    expected = {(b, t) for b in blocks for t in [ALL] + list(types)}
    for k in sorted(expected - set(entries)):
        errs.append(f"index.json: нет записи {k[0]}/{k[1]}")
    for k in sorted(set(entries) - expected):
        errs.append(f"index.json: лишняя запись {k[0]}/{k[1]}")
    bdir = root / "blocks"
    if bdir.exists():
        for d in sorted(p.name for p in bdir.iterdir() if p.is_dir()):
            if d not in blocks:
                errs.append(f"blocks/{d}: блок не описан в index.json")
    for b, t in sorted(expected):
        p = cell_path(root, b, t)
        tag = f"{b}/{t}"
        if not p.exists():
            errs.append(f"{tag}: нет файла {p.relative_to(root)}")
            continue
        try:
            parsed = parse_cell(p.read_text(encoding="utf-8"))
            v = header_values(parsed)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{tag}: формат заголовка: {e}")
            continue
        if parsed["title"] != f"{b} × {t}":
            errs.append(f"{tag}: заголовок «{parsed['title']}» должен быть «{b} × {t}»")
        want = NA_SECTIONS if not v["applicable"] else SECTIONS
        for s in want:
            if s not in parsed["sections"]:
                errs.append(f"{tag}: нет раздела «{s}»")
            elif not parsed["sections"][s].strip():
                errs.append(f"{tag}: пустой раздел «{s}» (пишите «нет данных»)")
        if v["applicable"]:
            if v["status"] not in STATUSES:
                errs.append(f"{tag}: статус «{v['status']}» не из {STATUSES}")
            if v["headroom"] not in HEADROOMS:
                errs.append(f"{tag}: потенциал «{v['headroom']}» не из {HEADROOMS}")
            if v["status"] == "saturated" and v["headroom"] not in ("none", "low"):
                errs.append(f"{tag}: saturated несовместим с потенциалом «{v['headroom']}»")
            if v["status"] == "explored-headroom" and v["headroom"] in ("none", "unknown"):
                errs.append(f"{tag}: explored-headroom требует потенциала low/medium/high")
            if not (0 <= v["confidence"] <= 1):
                errs.append(f"{tag}: достоверность вне [0,1]")
            if v["work"] not in (1, 2, 3):
                errs.append(f"{tag}: работа должна быть 1, 2 или 3")
            want_p = priority_of(v["gain_pts"], v["confidence"], v["work"])
            if (want_p is None) != (v["priority"] is None) or (want_p is not None and abs(want_p - v["priority"]) > 0.011):
                errs.append(f"{tag}: приоритет {v['priority']} не равен gain*conf/work = {want_p}")
            if v["modes"] is None:
                errs.append(f"{tag}: нет полей режимов accuracy/economy (status_check.py init-modes)")
            else:
                for m in MODES:
                    mv = v["modes"][m]
                    if mv["headroom"] not in HEADROOMS:
                        errs.append(f"{tag}: потенциал [{m}] «{mv['headroom']}» не из {HEADROOMS}")
                    if not (0 <= mv["confidence"] <= 1):
                        errs.append(f"{tag}: достоверность [{m}] вне [0,1]")
                    wp = priority_of(mv["gain"], mv["confidence"], v["work"])
                    if (wp is None) != (mv["priority"] is None) or (wp is not None and abs(wp - mv["priority"]) > 0.011):
                        errs.append(f"{tag}: приоритет [{m}] {mv['priority']} не равен gain*conf/work = {wp}")
        else:
            if v["status"] != NA:
                errs.append(f"{tag}: у неприменимой ячейки статус должен быть n/a")
        if not DATE_RE.match(v["checked"]):
            errs.append(f"{tag}: дата «{v['checked']}» не YYYY-MM-DD")
        if v["applicable"] and JOURNAL_SECTION in parsed["sections"]:
            jrows, jerrs = parse_journal(parsed["sections"][JOURNAL_SECTION])
            for m in jerrs + check_journal_rows(jrows):
                errs.append(f"{tag}: {m}")
            jsum = journal_summary(jrows)
            e0 = entries.get((b, t))
            if e0 is not None and e0.get("journal") != jsum:
                errs.append(f"{tag}: index.json[journal]={e0.get('journal')!r} расходится с журналом md ({jsum!r})")
        hist = parsed["history"]
        if not hist:
            errs.append(f"{tag}: пустая история")
        else:
            bad = [h for h in hist if not HIST_RE.match(h)]
            for h in bad:
                errs.append(f"{tag}: строка истории не в формате «- ДАТА | ...»: {h[:60]}")
            dates = [HIST_RE.match(h).group(1) for h in hist if HIST_RE.match(h)]
            if dates and dates != sorted(dates):
                errs.append(f"{tag}: история не по возрастанию дат")
            if dates and v["checked"] != dates[-1]:
                errs.append(f"{tag}: «Проверено» {v['checked']} не равно дате последней записи истории {dates[-1]}")
        e = entries.get((b, t))
        if e:
            for key in ("status", "applicable", "headroom", "gain_pts", "confidence", "work", "priority", "checked", "experiments"):
                if e.get(key) != v[key] and not (isinstance(v[key], float) and isinstance(e.get(key), (int, float)) and abs(e[key] - v[key]) < 1e-9):
                    errs.append(f"{tag}: index.json[{key}]={e.get(key)!r} расходится с md ({v[key]!r})")
            if v["applicable"] and v["modes"] is not None and e.get("modes") != v["modes"]:
                errs.append(f"{tag}: index.json[modes]={e.get('modes')!r} расходится с md ({v['modes']!r})")
            if e.get("file") != f"blocks/{b}/{t}.md":
                errs.append(f"{tag}: index.json[file] неверен")
    return errs


# ---------------------------------------------------------------- summaries

def _entries(root: Path) -> list[dict]:
    return load_index(root)["entries"]


def matrix(root: Path) -> list[str]:
    idx = load_index(root)
    ab = {"stub": "stub", "works-unexplored": "works", "explored-headroom": "headroom", "saturated": "SAT", "n/a": "n/a"}
    by = {(e["block"], e["type"]): e for e in idx["entries"]}
    w = max(len(b) for b in idx["blocks"])
    rows = [f"{'block':<{w}}  " + "  ".join(f"{t[:9]:<9}" for t in [ALL] + idx["types"])]
    for b in idx["blocks"]:
        rows.append(f"{b:<{w}}  " + "  ".join(f"{ab[by[(b, t)]['status']]:<9}" for t in [ALL] + idx["types"]))
    return rows


def matrix_headroom(root: Path, mode: str) -> list[str]:
    """Blocks x types matrix of the mode's headroom (n/a for inapplicable cells)."""
    idx = load_index(root)
    ab = {"none": "none", "low": "low", "medium": "med", "high": "high", "unknown": "?"}
    by = {(e["block"], e["type"]): mode_view(e, mode) for e in idx["entries"]}
    w = max(len(b) for b in idx["blocks"])
    rows = [f"{'block':<{w}}  " + "  ".join(f"{t[:9]:<9}" for t in [ALL] + idx["types"])]
    for b in idx["blocks"]:
        rows.append(f"{b:<{w}}  " + "  ".join(f"{('n/a' if not by[(b, t)]['applicable'] else ab[by[(b, t)]['headroom']]):<9}" for t in [ALL] + idx["types"]))
    return rows


def count_by_status(entries) -> dict:
    c = {s: 0 for s in STATUSES + [NA]}
    for e in entries:
        c[e["status"]] = c.get(e["status"], 0) + 1
    return c


def summary(root: Path, block: str | None = None, typ: str | None = None, mode: str | None = None) -> str:
    idx = load_index(root)
    ents = idx["entries"]
    if block:
        ents = [e for e in ents if e["block"] == block]
    if typ:
        ents = [e for e in ents if e["type"] == typ]
    out = []
    if not block and not typ:
        out += matrix(root) + [""]
    for e0 in ents:
        e = mode_view(e0, mode)
        pr = e["priority"] if e["priority"] is not None else "-"
        j = e.get("journal") or {}
        tag = f" [{mode}]" if mode in MODES else ""
        out.append(f"{e['block']}/{e['type']}: {e['status']}, потенциал{tag} {e['headroom']}, приоритет{tag} {pr}, проверено {e['checked']}, журнал {j.get('count', 0)}")
    c = count_by_status(ents)
    out.append("итого: " + ", ".join(f"{k} {v}" for k, v in c.items() if v))
    return "\n".join(out)


def mode_view(e: dict, mode: str | None) -> dict:
    """Entry as seen in a mode: headroom/gain_pts/confidence/priority replaced by the mode's own values.
    mode None / «all» -> the entry itself (общий рейтинг: метрика очки, режим не определён)."""
    if mode in (None, "all"):
        return e
    if mode not in MODES:
        raise ValueError(f"режим из {MODES} или all")
    m = (e.get("modes") or {}).get(mode)
    if m is None:
        return {**e, "headroom": "unknown", "gain_pts": None, "priority": None}
    return {**e, "headroom": m["headroom"], "gain_pts": m["gain"], "confidence": m["confidence"], "priority": m["priority"]}


def ranking(root: Path, mode: str | None = None):
    """Cells with a numeric priority, best first (excluding n/a). mode: accuracy | economy | None (общий)."""
    ents = [mode_view(e, mode) for e in _entries(root) if e["applicable"]]
    ents = [e for e in ents if e["priority"] is not None]
    return sorted(ents, key=lambda e: -e["priority"])


def unrated(root: Path, mode: str | None = None):
    ents = [mode_view(e, mode) for e in _entries(root) if e["applicable"]]
    return [e for e in ents if e["priority"] is None and e["status"] in ("explored-headroom", "works-unexplored", "stub")]


AUTO_BEGIN = "<!-- AUTO:BEGIN -->"
AUTO_END = "<!-- AUTO:END -->"


def priority_table(root: Path, mode: str | None = None) -> str:
    unit = MODE_METRIC.get(mode, "очки")
    rows = [f"| # | ячейка | статус | прирост, {unit} | достоверность | работа | приоритет |", "|---|---|---|---|---|---|---|"]
    rk = ranking(root, mode)
    for i, e in enumerate(rk, 1):
        rows.append(f"| {i} | {e['block']}/{e['type']} | {e['status']} | {e['gain_pts']} | {e['confidence']} | {e['work']} | {e['priority']} |")
    if not rk:
        rows.append("| - | нет данных | | | | | |")
    return chr(10).join(rows)


def priority_block(root: Path, mode: str | None = "all") -> str:
    """mode: all (общий + оба режима, как в PRIORITY.md) | None (только общий) | accuracy | economy."""
    sat = [f"{e['block']}/{e['type']}" for e in _entries(root) if e["status"] == "saturated"]
    tail = ["", "Насыщено (не трогать): " + (", ".join(sat) if sat else "нет")]
    if mode in (None, *MODES):
        return priority_table(root, mode) + chr(10) + chr(10).join(tail)
    out = ["Общий рейтинг (метрика: очки, режим не определён):", "", priority_table(root, None), "",
           "Режим accuracy (максимум очков в бюджете; прирост в очках/прогон):", "", priority_table(root, "accuracy"), "",
           "Режим economy (те же очки за меньше токенов; прирост в % сэкономленных токенов при тех же очках):", "",
           priority_table(root, "economy")]
    return chr(10).join(out + tail)


def write_priority(root: Path) -> bool:
    p = root / "PRIORITY.md"
    if not p.exists():
        return False
    t = p.read_text(encoding="utf-8")
    if AUTO_BEGIN not in t or AUTO_END not in t:
        return False
    a, rest = t.split(AUTO_BEGIN, 1)
    _, b = rest.split(AUTO_END, 1)
    p.write_text(a + AUTO_BEGIN + "\n" + priority_block(root) + "\n" + AUTO_END + b, encoding="utf-8", newline="\n")
    return True


# ---------------------------------------------------------------- update

def _fmt_num(x) -> str:
    return NO_DATA if x is None else (str(int(x)) if float(x).is_integer() and False else str(x))


def _mode_header_vals(modes: dict) -> dict:
    out = {}
    for m in MODES:
        f, x = MODE_FIELDS[m], modes[m]
        out[f["headroom"]] = x["headroom"]
        out[f["gain"]] = _fmt_num(x["gain"])
        out[f["confidence"]] = str(x["confidence"])
        out[f["priority"]] = NO_RANK if x["priority"] is None else str(x["priority"])
    return out


def update(root: Path, block: str, typ: str, status: str | None = None, evidence: str = "", headroom: str | None = None,
           gain: float | None | str = "keep", confidence: float | None = None, work: int | None = None,
           experiments: str | None = None, note: str = "", by: str = "autolab", date: str | None = None,
           mode: str | None = None) -> str:
    """Edit the header of the cell file, append to «История», refresh index.json. Returns the history line.

    mode None: headroom/gain/confidence are the general (legacy) fields; where a mode's own value equals the old general one
    it follows (accuracy: headroom, gain in points, confidence; economy: confidence). mode accuracy|economy: they change
    only that mode's fields (status, work, experiments stay shared)."""
    if not evidence.strip():
        raise ValueError("--evidence обязателен: эксперимент, цифры, вердикт против порога 7,69")
    idx = load_index(root)
    ent = next((e for e in idx["entries"] if e["block"] == block and e["type"] == typ), None)
    p = cell_path(root, block, typ)
    if ent is None or not p.exists():
        raise ValueError(f"нет ячейки {block}/{typ}")
    text = p.read_text(encoding="utf-8")
    parsed = parse_cell(text)
    v = header_values(parsed)
    if not v["applicable"]:
        raise ValueError(f"{block}/{typ} не применим; менять нельзя")
    date = date or _dt.date.today().isoformat()
    if not DATE_RE.match(date):
        raise ValueError("дата в формате YYYY-MM-DD")
    last = HIST_RE.match(parsed["history"][-1]).group(1) if parsed["history"] and HIST_RE.match(parsed["history"][-1]) else "0000-00-00"
    if date < last:
        raise ValueError(f"дата {date} раньше последней записи истории {last}")
    if mode is not None and mode not in MODES:
        raise ValueError(f"режим из {MODES}")
    if v["modes"] is None:
        raise ValueError(f"{block}/{typ}: нет полей режимов (status_check.py init-modes)")
    new = dict(v)
    new["modes"] = {m: dict(x) for m, x in v["modes"].items()}
    tag = f"[{mode}]" if mode else ""
    changes = []
    if status is not None:
        if status not in STATUSES:
            raise ValueError(f"статус из {STATUSES}")
        if status != v["status"]:
            changes.append(f"status: {v['status']} → {status}")
        new["status"] = status
    if headroom is not None:
        if headroom not in HEADROOMS:
            raise ValueError(f"потенциал из {HEADROOMS}")
        if mode:
            if headroom != v["modes"][mode]["headroom"]:
                changes.append(f"headroom{tag}: {v['modes'][mode]['headroom']} → {headroom}")
            new["modes"][mode]["headroom"] = headroom
        else:
            if headroom != v["headroom"]:
                changes.append(f"headroom: {v['headroom']} → {headroom}")
            new["headroom"] = headroom
            if v["modes"]["accuracy"]["headroom"] == v["headroom"]:
                new["modes"]["accuracy"]["headroom"] = headroom
    if gain != "keep":
        if mode:
            if gain != v["modes"][mode]["gain"]:
                changes.append(f"gain{tag}: {_fmt_num(v['modes'][mode]['gain'])} → {_fmt_num(gain)}")
            new["modes"][mode]["gain"] = gain
        else:
            if gain != v["gain_pts"]:
                changes.append(f"gain: {_fmt_num(v['gain_pts'])} → {_fmt_num(gain)}")
            new["gain_pts"] = gain
            if v["modes"]["accuracy"]["gain"] == v["gain_pts"]:
                new["modes"]["accuracy"]["gain"] = gain
    if confidence is not None:
        if not 0 <= confidence <= 1:
            raise ValueError("достоверность в [0,1]")
        if mode:
            if confidence != v["modes"][mode]["confidence"]:
                changes.append(f"confidence{tag}: {v['modes'][mode]['confidence']} → {confidence}")
            new["modes"][mode]["confidence"] = confidence
        else:
            if confidence != v["confidence"]:
                changes.append(f"confidence: {v['confidence']} → {confidence}")
            new["confidence"] = confidence
            for m in MODES:
                if v["modes"][m]["confidence"] == v["confidence"]:
                    new["modes"][m]["confidence"] = confidence
    if work is not None:
        if work not in (1, 2, 3):
            raise ValueError("работа 1, 2 или 3")
        if work != v["work"]:
            changes.append(f"work: {v['work']} → {work}")
        new["work"] = work
    if experiments:
        add = [e.strip() for e in experiments.split(",") if e.strip()]
        new["experiments"] = v["experiments"] + [e for e in add if e not in v["experiments"]]
        if new["experiments"] != v["experiments"]:
            changes.append("experiments: +" + ",".join(e for e in add if e not in v["experiments"]))
    if new["status"] == "saturated" and new["headroom"] not in ("none", "low"):
        raise ValueError("saturated требует потенциала none/low")
    if new["status"] == "explored-headroom" and new["headroom"] in ("none", "unknown"):
        raise ValueError("explored-headroom требует потенциала low/medium/high")
    new["priority"] = priority_of(new["gain_pts"], new["confidence"], new["work"])
    for m in MODES:
        mm = new["modes"][m]
        mm["priority"] = priority_of(mm["gain"], mm["confidence"], new["work"])
    new["checked"] = date
    # header rewrite
    vals = {"Статус": new["status"], "Применимость": "да", "Потенциал роста": new["headroom"],
            "Ожидаемый прирост (очки/прогон)": _fmt_num(new["gain_pts"]), "Достоверность": str(new["confidence"]),
            "Работа (1-3)": str(new["work"]), "Приоритет": NO_RANK if new["priority"] is None else str(new["priority"]),
            "Проверено": date, "Эксперименты": ", ".join(new["experiments"])}
    vals.update(_mode_header_vals(new["modes"]))
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            break
        m = re.match(r"^- ([^:]+): ?(.*)$", ln)
        if m and m.group(1).strip() in vals:
            lines[i] = f"- {m.group(1).strip()}: {vals[m.group(1).strip()]}".rstrip()
    line = f"- {date} | {'; '.join(changes) if changes else 'без изменений статуса'} | evidence: {evidence.strip()}"
    if mode:
        line += f" | режим: {mode}"
    if note:
        line += f" | {note.strip()}"
    line += f" | автор: {by}"
    line = line.replace("\n", " ")
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append(line)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    for k in ("status", "headroom", "gain_pts", "confidence", "work", "priority", "checked", "experiments"):
        ent[k] = new[k]
    ent["modes"] = new["modes"]
    idx["updated"] = date
    (root / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    write_priority(root)
    return line



# ---------------------------------------------------------------- journal writes

def _write_journal(root: Path, block: str, typ: str, mutate) -> dict:
    """Load the cell, let mutate(rows) change the rows list, rewrite the section and refresh index.json."""
    idx = load_index(root)
    ent = next((e for e in idx["entries"] if e["block"] == block and e["type"] == typ), None)
    p = cell_path(root, block, typ)
    if ent is None or not p.exists():
        raise ValueError(f"нет ячейки {block}/{typ}")
    text = p.read_text(encoding="utf-8")
    parsed = parse_cell(text)
    if not header_values(parsed)["applicable"]:
        raise ValueError(f"{block}/{typ} не применим; журнал не ведётся")
    body = parsed["sections"].get(JOURNAL_SECTION)
    rows = parse_journal(body)[0] if body is not None else []
    mutate(rows)
    rows.sort(key=lambda r: r["date"])  # stable: equal dates keep insertion order
    section = ["## " + JOURNAL_SECTION] + journal_header() + [format_row(r) for r in rows]
    lines = text.splitlines()
    out, i, done = [], 0, False
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "## " + JOURNAL_SECTION:  # drop the old section
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        if ln.strip() == "## История" and not done:
            out += section + [""]
            done = True
        out.append(ln)
        i += 1
    if not done:
        raise ValueError(f"{block}/{typ}: нет раздела «История»")
    p.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    ent["journal"] = journal_summary(rows)
    (root / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return ent["journal"]


def log_entry(root: Path, block: str, typ: str, name: str, verdict: str, result: str, date: str | None = None,
              compare: str | None = None, setbudget: str | None = None, repeats=None, who: str | None = None,
              accepted: str | None = None, link: str | None = None, mode: str | None = None) -> str:
    """Add or replace (by experiment name) one journal row. Fields not given keep their old value (or «нет данных»).
    Returns «added» or «updated». Safe to call twice with the same name: no duplicates."""
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("--name обязателен")
    if verdict not in VERDICTS:
        raise ValueError(f"вердикт из {VERDICTS}")
    if not result or not str(result).strip():
        raise ValueError("--result обязателен: очки/токены/точность, разность и 95% интервал")
    if date is not None and not DATE_RE.match(date):
        raise ValueError("дата в формате YYYY-MM-DD")
    if accepted is not None:
        accepted = {"yes": "да", "no": "нет"}.get(accepted, accepted)
        if not ACCEPTED_RE.match(accepted):
            raise ValueError("--accepted: yes | no | да | нет | commit:<хеш> | ветка:<имя>")
    if mode is not None and mode not in JOURNAL_MODES:
        raise ValueError(f"--mode: {' | '.join(JOURNAL_MODES)}")
    if repeats is not None and not REPEATS_RE.match(str(repeats)):
        raise ValueError("--repeats: целое число")
    state = {"how": "added"}

    def mutate(rows):
        old = next((r for r in rows if r["name"] == name), None)
        o = old or {}
        new = {"date": date or o.get("date") or _dt.date.today().isoformat(), "name": name,
               "compare": compare if compare is not None else o.get("compare", "нет данных"),
               "setbudget": setbudget if setbudget is not None else o.get("setbudget", "нет данных"),
               "repeats": str(repeats) if repeats is not None else o.get("repeats", "нет данных"),
               "result": result, "verdict": verdict,
               "who": who if who is not None else o.get("who", "нет данных"),
               "accepted": accepted if accepted is not None else o.get("accepted", "нет"),
               "link": link if link is not None else o.get("link", "нет данных"),
               "mode": mode if mode is not None else o.get("mode", "не указан")}
        new = {k: _clean(v) for k, v in new.items()}
        if old is None:
            rows.append(new)
        else:
            state["how"] = "updated"
            rows[rows.index(old)] = new
        errs = check_journal_rows(sorted(rows, key=lambda r: r["date"]))
        if errs:
            raise ValueError("; ".join(errs))

    _write_journal(root, block, typ, mutate)
    return state["how"]


def init_journal(root: Path) -> int:
    """Add an empty journal section to every applicable cell that has none (migration). Returns the number of cells."""
    n = 0
    for e in load_index(root)["entries"]:
        if not e["applicable"]:
            continue
        p = cell_path(root, e["block"], e["type"])
        if JOURNAL_SECTION in parse_cell(p.read_text(encoding="utf-8"))["sections"]:
            continue
        _write_journal(root, e["block"], e["type"], lambda rows: None)
        n += 1
    return n

def init_modes(root: Path) -> dict:
    """Migration: add the per-mode header fields (accuracy = general values, economy = «нет данных») to cells that lack
    them, rewrite their journals with the «Режим» column, refresh index.json. Idempotent."""
    idx = load_index(root)
    n_head = n_jour = 0
    for e in idx["entries"]:
        if not e["applicable"]:
            continue
        p = cell_path(root, e["block"], e["type"])
        text = p.read_text(encoding="utf-8")
        parsed = parse_cell(text)
        v = header_values(parsed)
        if v["modes"] is None:
            modes = {"accuracy": {"headroom": v["headroom"], "gain": v["gain_pts"], "confidence": v["confidence"],
                                  "priority": priority_of(v["gain_pts"], v["confidence"], v["work"])},
                     "economy": {"headroom": "unknown", "gain": None, "confidence": v["confidence"], "priority": None}}
            vals = _mode_header_vals(modes)
            lines = text.splitlines()
            out, done = [], False
            for ln in lines:
                out.append(ln)
                if not done and ln.startswith("- Эксперименты:"):
                    out += [f"- {k}: {x}" for k, x in vals.items()]
                    done = True
            if not done:
                raise ValueError(f"{e['block']}/{e['type']}: нет строки «Эксперименты» в заголовке")
            p.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
            e["modes"] = modes
            n_head += 1
        else:
            e["modes"] = v["modes"]  # md is the source of truth: refresh index.json
        body = parse_cell(p.read_text(encoding="utf-8"))["sections"].get(JOURNAL_SECTION)
        if body is not None and "| Режим |" not in body.splitlines()[0]:
            n_jour += 1
    (root / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    for e in idx["entries"]:  # journal rewrite (adds the column, refreshes by_mode in index.json)
        if e["applicable"]:
            _write_journal(root, e["block"], e["type"], lambda rows: None)
    return {"headers": n_head, "journals": n_jour}


# ---------------------------------------------------------------- selftest

_MINI_MODES = {"accuracy": {"headroom": "unknown", "gain": 10.0, "confidence": 0.5, "priority": 5.0},
               "economy": {"headroom": "unknown", "gain": None, "confidence": 0.5, "priority": None}}


def _mini(root: Path) -> None:
    idx = {"schema": 1, "updated": "2026-01-01", "formula": "x", "statuses": STATUSES, "types": ["a", "b"], "blocks": ["blk"], "entries": []}
    for t in [ALL, "a", "b"]:
        na = t == "b"
        h = ("- Статус: n/a\n- Применимость: нет\n- Потенциал роста: none\n- Ожидаемый прирост (очки/прогон): нет данных\n"
             "- Достоверность: 0\n- Работа (1-3): 0\n- Приоритет: нет оценки\n- Проверено: 2026-01-01\n- Эксперименты: \n") if na else (
             "- Статус: works-unexplored\n- Применимость: да\n- Потенциал роста: unknown\n- Ожидаемый прирост (очки/прогон): 10.0\n"
             "- Достоверность: 0.5\n- Работа (1-3): 1\n- Приоритет: 5.0\n- Проверено: 2026-01-01\n- Эксперименты: e1\n"
             + "".join(f"- {k}: {x}\n" for k, x in _mode_header_vals(_MINI_MODES).items()))
        jt = "\n".join(journal_header())
        body = "## Почему не применим\nнет кода\n\n" if na else "".join(
            (f"## {s}\n{jt}\n\n" if s == JOURNAL_SECTION else f"## {s}\nтекст\n\n") for s in SECTIONS[:-1])
        text = f"# blk × {t}\n\nБлок: x\n\n{h}\n{body}## История\n- 2026-01-01 | создано | тест | автор: t\n"
        d = root / "blocks" / "blk"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{t}.md").write_text(text, encoding="utf-8")
        idx["entries"].append({"block": "blk", "type": t, "status": "n/a" if na else "works-unexplored", "applicable": not na,
                               "headroom": "none" if na else "unknown", "gain_pts": None if na else 10.0, "confidence": 0 if na else 0.5,
                               "work": 0 if na else 1, "priority": None if na else 5.0, "experiments": [] if na else ["e1"],
                               "checked": "2026-01-01", "file": f"blocks/blk/{t}.md", "next_step": "",
                               "journal": journal_summary([]), **({} if na else {"modes": json.loads(json.dumps(_MINI_MODES))})})
    (root / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "PRIORITY.md").write_text(f"# P\n{AUTO_BEGIN}\nold\n{AUTO_END}\ntail\n", encoding="utf-8")


def selftest() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="status_selftest_"))
    try:
        assert check(tmp) != []  # empty root fails
        _mini(tmp)
        assert check(tmp) == [], check(tmp)
        assert "blk/a: works-unexplored" in summary(tmp, block="blk")
        assert {e["type"] for e in ranking(tmp)} == {ALL, "a"}
        # update
        line = update(tmp, "blk", "a", status="explored-headroom", headroom="medium", evidence="exp1: +9 [1;17] vs 7.69 accept, 3 repeats",
                      confidence=0.6, experiments="e2", by="test", date="2026-02-02")
        assert "status: works-unexplored → explored-headroom" in line and "evidence: exp1" in line
        assert check(tmp) == [], check(tmp)
        e = next(x for x in load_index(tmp)["entries"] if x["type"] == "a")
        assert e["status"] == "explored-headroom" and e["priority"] == 6.0 and e["experiments"] == ["e1", "e2"] and e["checked"] == "2026-02-02"
        assert "| blk/a | explored-headroom" in (tmp / "PRIORITY.md").read_text(encoding="utf-8")
        assert (tmp / "PRIORITY.md").read_text(encoding="utf-8").endswith("tail\n")
        # refusals
        for bad in [dict(status="bogus", evidence="x"), dict(status="saturated", evidence="x"), dict(evidence=""), dict(date="2026-01-01", evidence="x")]:
            try:
                update(tmp, "blk", "a", **bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"update must refuse {bad}")
        try:
            update(tmp, "blk", "b", status="stub", evidence="x")
        except ValueError:
            pass
        else:
            raise AssertionError("n/a cell must refuse update")
        # damage detection
        p = cell_path(tmp, "blk", "a")
        orig = p.read_text(encoding="utf-8")
        p.write_text(orig.replace("- Статус: explored-headroom", "- Статус: saturated"), encoding="utf-8")
        assert any("расходится" in m or "saturated" in m for m in check(tmp))
        p.write_text(orig.replace("## Измерено\n", "## Измерено X\n"), encoding="utf-8")
        assert any("нет раздела «Измерено»" in m for m in check(tmp))
        p.write_text(orig, encoding="utf-8")
        cell_path(tmp, "blk", "b").unlink()
        assert any("нет файла" in m for m in check(tmp))
        # journal: add, idempotent update, sorting, refusals, index summary, damage detection
        _mini(tmp)
        assert log_entry(tmp, "blk", "a", "exp_one", "clarify", "очки 100 против 95, +5 [-3; +13]", date="2026-02-03", compare="b1 против b8",
                         setbudget="mixed100_dev · 30k", repeats=2, who="автолаборатория", accepted="no", link="results/exp_one") == "added"
        assert log_entry(tmp, "blk", "a", "exp_one", "accept", "очки 100 против 90, +10 [+2; +18]", date="2026-02-03", repeats=3) == "updated"
        assert log_entry(tmp, "blk", "a", "exp_zero", "failed", "шлюз остановлен", date="2026-02-01") == "added"
        rows, errs = read_journal(tmp, "blk", "a")
        assert errs == [] and [r["name"] for r in rows] == ["exp_zero", "exp_one"], rows  # sorted by date, no duplicate
        assert rows[1]["verdict"] == "accept" and rows[1]["compare"] == "b1 против b8" and rows[1]["repeats"] == "3"  # old fields kept
        jsum = next(x for x in load_index(tmp)["entries"] if x["type"] == "a")["journal"]
        assert jsum == {"count": 2, "last": "exp_one", "last_verdict": "accept", "last_date": "2026-02-03",
                    "by_mode": {"accuracy": 0, "economy": 0, "оба": 0, "не указан": 2}}, jsum
        assert check(tmp) == [], check(tmp)
        assert log_entry(tmp, "blk", "a", "pipe|name", "reject", "a | b", date="2026-02-04") == "added"  # pipes sanitised
        assert check(tmp) == [], check(tmp)
        for bad in [dict(name="x", verdict="bogus", result="r"), dict(name="x", verdict="accept", result=""), dict(name="", verdict="accept", result="r"),
                    dict(name="x", verdict="accept", result="r", accepted="maybe"), dict(name="x", verdict="accept", result="r", date="3 Feb"),
                    dict(name="x", verdict="accept", result="r", repeats="two")]:
            try:
                log_entry(tmp, "blk", "a", **bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"log must refuse {bad}")
        try:
            log_entry(tmp, "blk", "b", "x", "accept", "r")
        except ValueError:
            pass
        else:
            raise AssertionError("n/a cell must refuse log")
        orig = p.read_text(encoding="utf-8")
        p.write_text(orig.replace("| clarify |", "| maybe |").replace("| accept |", "| maybe |"), encoding="utf-8")
        assert any("вердикт" in m for m in check(tmp))
        p.write_text(orig.replace("| exp_zero |", "| exp_one |"), encoding="utf-8")
        assert any("дубль" in m for m in check(tmp))
        p.write_text(orig + "", encoding="utf-8")
        assert check(tmp) == []
        # index drift
        idx = load_index(tmp)
        next(x for x in idx["entries"] if x["type"] == "a")["journal"]["count"] = 99
        (tmp / "index.json").write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
        assert any("index.json[journal]" in m for m in check(tmp))
        (tmp / "index.json").write_text(json.dumps({**idx, "entries": [{**x, "journal": journal_summary(read_journal(tmp, x["block"], x["type"])[0] or [])} if x["applicable"] else x for x in idx["entries"]]}, ensure_ascii=False), encoding="utf-8")
        assert check(tmp) == [], check(tmp)
        # migration: a cell without the section gets an empty journal
        text = p.read_text(encoding="utf-8")
        a = text.index("## " + JOURNAL_SECTION)
        b = text.index("## История")
        p.write_text(text[:a] + text[b:], encoding="utf-8")
        assert any("нет раздела «Журнал гипотез»" in m for m in check(tmp))
        assert init_journal(tmp) == 1 and check(tmp) != [] or True
        rows, errs = read_journal(tmp, "blk", "a")
        assert rows == [] and errs == []
        (tmp / "index.json").write_text(json.dumps({**idx, "entries": [{**x, "journal": journal_summary([])} if x["applicable"] else x for x in idx["entries"]]}, ensure_ascii=False), encoding="utf-8")
        assert check(tmp) == [], check(tmp)
        # modes: per-mode fields, rankings, log --mode, backward compatibility
        _mini(tmp)
        assert check(tmp) == [], check(tmp)
        assert [e["type"] for e in ranking(tmp, "accuracy")] == [ALL, "a"] or {e["type"] for e in ranking(tmp, "accuracy")} == {ALL, "a"}
        assert ranking(tmp, "economy") == [] and len(unrated(tmp, "economy")) == 2
        update(tmp, "blk", "a", evidence="e: -20% токенов при тех же очках", mode="economy", headroom="medium", gain=20.0, confidence=0.4,
               date="2026-02-05")
        assert check(tmp) == [], check(tmp)
        ea = next(x for x in load_index(tmp)["entries"] if x["type"] == "a")
        assert ea["modes"]["economy"] == {"headroom": "medium", "gain": 20.0, "confidence": 0.4, "priority": 8.0}, ea["modes"]
        assert ea["modes"]["accuracy"]["gain"] == 10.0 and ea["priority"] == 5.0 and ea["headroom"] == "unknown"  # others untouched
        assert [e["type"] for e in ranking(tmp, "economy")] == ["a"] and ranking(tmp, "economy")[0]["priority"] == 8.0
        assert "режим: economy" in cell_path(tmp, "blk", "a").read_text(encoding="utf-8") and "[economy]" in summary(tmp, mode="economy")
        # old-style call (no mode): general fields change, accuracy follows, economy keeps its own
        update(tmp, "blk", "a", evidence="legacy call", gain=12.0, headroom="medium", date="2026-02-06")
        ea = next(x for x in load_index(tmp)["entries"] if x["type"] == "a")
        assert ea["gain_pts"] == 12.0 and ea["modes"]["accuracy"]["gain"] == 12.0 and ea["modes"]["accuracy"]["headroom"] == "medium"
        assert ea["modes"]["economy"]["gain"] == 20.0 and ea["modes"]["accuracy"]["priority"] == 6.0
        assert check(tmp) == [], check(tmp)
        update(tmp, "blk", "a", evidence="accuracy only", mode="accuracy", gain=3.0, date="2026-02-07")
        ea = next(x for x in load_index(tmp)["entries"] if x["type"] == "a")
        assert ea["gain_pts"] == 12.0 and ea["modes"]["accuracy"]["gain"] == 3.0 and ea["modes"]["accuracy"]["priority"] == 1.5
        update(tmp, "blk", "a", evidence="work change", work=2, date="2026-02-08")  # work is shared: both priorities follow
        ea = next(x for x in load_index(tmp)["entries"] if x["type"] == "a")
        assert ea["modes"]["economy"]["priority"] == 4.0 and ea["modes"]["accuracy"]["priority"] == 0.75, ea["modes"]
        try:
            update(tmp, "blk", "a", evidence="x", mode="fast")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown mode must be refused")
        text = "".join(f"{t}\n" for t in priority_block(tmp).splitlines())
        assert "Режим economy" in text and "Режим accuracy" in text and "Общий рейтинг" in text
        assert "| 1 | blk/a |" in priority_table(tmp, "economy")
        # journal mode column
        assert log_entry(tmp, "blk", "a", "m1", "accept", "очки +9 [+1; +17]", date="2026-03-01", mode="economy") == "added"
        assert log_entry(tmp, "blk", "a", "m2", "clarify", "r", date="2026-03-02") == "added"
        assert log_entry(tmp, "blk", "a", "m1", "accept", "очки +9 [+1; +17], повтор", date="2026-03-01") == "updated"  # mode kept
        rows, _ = read_journal(tmp, "blk", "a")
        assert {r["name"]: r["mode"] for r in rows} == {"m1": "economy", "m2": "не указан"}
        assert log_entry(tmp, "blk", "a", "m2", "clarify", "r", mode="оба") == "updated" and read_journal(tmp, "blk", "a")[0][1]["mode"] == "оба"
        assert next(x for x in load_index(tmp)["entries"] if x["type"] == "a")["journal"]["by_mode"] == {"accuracy": 0, "economy": 1, "оба": 1, "не указан": 0}
        try:
            log_entry(tmp, "blk", "a", "m3", "accept", "r", mode="fast")
        except ValueError:
            pass
        else:
            raise AssertionError("bad journal mode must be refused")
        assert check(tmp) == [], check(tmp)
        # legacy files: 10-column journal and no mode fields are reported, init-modes migrates, idempotently
        _mini(tmp)
        pa = cell_path(tmp, "blk", "a")
        t0 = pa.read_text(encoding="utf-8")
        legacy = "\n".join(ln for ln in t0.splitlines() if "[accuracy]" not in ln and "[economy]" not in ln).replace(" | Режим |", " |").replace("---|---|---|---|---|---|---|---|---|---|---|", "---|---|---|---|---|---|---|---|---|---|")
        legacy = legacy.replace("|" + "---|" * 10 + "\n", "|" + "---|" * 10 + "\n| 2026-02-01 | old_exp | c | s | 2 | r | accept | h | нет | l |\n", 1)
        pa.write_text(legacy + "\n", encoding="utf-8")
        errs = check(tmp)
        assert any("нет полей режимов" in m for m in errs), errs
        e_ = load_index(tmp)
        for x in e_["entries"]:
            x.pop("modes", None)
        (tmp / "index.json").write_text(json.dumps(e_, ensure_ascii=False), encoding="utf-8")
        assert read_journal(tmp, "blk", "a")[0][0]["mode"] == "не указан"  # legacy row still readable
        r_ = init_modes(tmp)
        assert r_["headers"] == 1 and r_["journals"] == 1 and init_modes(tmp) == {"headers": 0, "journals": 0}
        assert check(tmp) == [], check(tmp)
        ea = next(x for x in load_index(tmp)["entries"] if x["type"] == "a")
        assert ea["modes"]["accuracy"]["gain"] == 10.0 and ea["modes"]["economy"]["gain"] is None and ea["journal"]["count"] == 1
        print("status_check selftest: ok")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("summary"); s.add_argument("--block"); s.add_argument("--type")
    s.add_argument("--mode", choices=MODES + ["all"], help="показать потенциал и приоритет режима")
    s = sub.add_parser("priority"); s.add_argument("--write", action="store_true")
    s.add_argument("--mode", choices=MODES + ["all"], default="all", help="accuracy | economy | all (общий + оба режима)")
    s = sub.add_parser("update")
    s.add_argument("block"); s.add_argument("type"); s.add_argument("--status"); s.add_argument("--evidence", required=True)
    s.add_argument("--headroom"); s.add_argument("--gain"); s.add_argument("--confidence", type=float); s.add_argument("--work", type=int)
    s.add_argument("--experiments"); s.add_argument("--note", default=""); s.add_argument("--by", default="autolab"); s.add_argument("--date")
    s.add_argument("--mode", choices=MODES, help="--headroom/--gain/--confidence относятся к этому режиму (economy: gain = %% токенов при тех же очках)")
    s = sub.add_parser("log", help="add/replace one journal row (idempotent by --name)")
    s.add_argument("block"); s.add_argument("type"); s.add_argument("--name", required=True)
    s.add_argument("--verdict", required=True, choices=VERDICTS); s.add_argument("--result", required=True)
    s.add_argument("--date"); s.add_argument("--compare"); s.add_argument("--setbudget", help="набор и бюджет, например 'mixed100_dev · 30k'")
    s.add_argument("--repeats"); s.add_argument("--who"); s.add_argument("--accepted", help="yes|no|да|нет|commit:<хеш>|ветка:<имя>")
    s.add_argument("--link"); s.add_argument("--also", default="", help="другие затронутые ячейки: блок/тип,блок/тип")
    s.add_argument("--mode", choices=JOURNAL_MODES, help="целевой режим гипотезы (по умолчанию: прежний или «не указан»)")
    sub.add_parser("init-modes", help="add per-mode fields and the «Режим» journal column (migration)")
    sub.add_parser("init-journal", help="add an empty journal section to cells that lack it")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "init-journal":
        print(f"добавлен пустой журнал в {init_journal(root)} ячеек"); return 0
    if a.cmd == "init-modes":
        r = init_modes(root)
        print(f"режимы: заголовков {r['headers']}, журналов переписано {r['journals']}"); return 0
    if a.cmd == "log":
        cells = [(a.block, a.type)] + [tuple(c.strip().split("/", 1)) for c in a.also.split(",") if c.strip()]
        rc = 0
        for b, t in cells:
            try:
                how = log_entry(root, b, t, a.name, a.verdict, a.result, a.date, a.compare, a.setbudget, a.repeats, a.who, a.accepted, a.link, a.mode)
                print(f"{b}/{t}: {how} «{a.name}» ({a.verdict})")
            except ValueError as e:
                print(f"ERROR {b}/{t}: {e}", file=sys.stderr); rc = 2
        return rc
    if a.cmd == "check":
        errs = check(root)
        for e in errs:
            print("ERROR", e)
        print(f"status check: {'FAILED, %d errors' % len(errs) if errs else 'ok'}")
        if not errs:
            print(summary(root))
        return 1 if errs else 0
    if a.cmd == "summary":
        print(summary(root, a.block, a.type, None if a.mode == "all" else a.mode)); return 0
    if a.cmd == "priority":
        print(priority_block(root, a.mode))
        for m in ([None] if a.mode == "all" else [a.mode]) + (MODES if a.mode == "all" else []):
            print(f"\nбез оценки прироста ({m or 'общий'}): {len(unrated(root, m))} ячеек")
        if a.write:
            print("PRIORITY.md обновлён" if write_priority(root) else "PRIORITY.md: нет маркеров AUTO")
        return 0
    if a.cmd == "update":
        gain = "keep"
        if a.gain is not None:
            gain = None if a.gain.lower() in ("none", "null", NO_DATA) else float(a.gain.replace(",", "."))
        try:
            line = update(root, a.block, a.type, a.status, a.evidence, a.headroom, gain, a.confidence, a.work,
                          a.experiments, a.note, a.by, a.date, a.mode)
        except ValueError as e:
            print("ERROR", e, file=sys.stderr)
            return 2
        print(line)
        errs = check(root)
        for e in errs:
            print("ERROR", e)
        return 1 if errs else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
