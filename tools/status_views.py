#!/usr/bin/env python3
"""Compact text views of harness/status/ for a chat (Telegram: one message <= 4096 characters).

Functions return a list of message strings, each within `limit` characters (split on line boundaries; a part counter
"(i/n)" is appended when there is more than one part). Plain text, no markdown. Nothing here talks to Telegram:
the bot session calls these functions.

    block_view(block, mode=None)   the four task types (+ block-wide row) for one block
    type_view(typ, mode=None)      all blocks for one task type
    cell_view(block, typ)    full text of one cell file
    matrix_view(mode=None)   blocks x types matrix with statuses (+ the mode's headroom matrix when mode is given)
    priority_view(top=10, mode=None)  ranking by gain * confidence / work; mode accuracy|economy, None = общий
    journal_view(block, typ, n=5)  last n hypothesis-journal rows compactly (+ the whole journal as a document if longer)
    handle_command(text)     «журнал <блок> <тип> [N]», «схема [режим]», «приоритеты [режим]» (+ /journal /scheme /priorities)
    parse_mode(word)         accuracy | economy | None (accepts also точность, экономия, all, общий)
    split_message(text, limit)

CLI: status_views.py block B | type T | cell B T | matrix | priority | journal B T [N] | selftest   (--root DIR; --mode accuracy|economy)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_check as sc  # noqa: E402

TG_LIMIT = 4096
ROOT = sc.DEFAULT_ROOT
_COUNTER_ROOM = 12  # "\n(12/12)"


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Split on line boundaries; a line longer than the limit is cut hard. Adds "(i/n)" when there are several parts."""
    text = text.strip("\n")
    if len(text) <= limit:
        return [text]
    budget = limit - _COUNTER_ROOM
    parts: list[str] = []
    cur = ""
    for line in text.split("\n"):
        while len(line) > budget:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:budget])
            line = line[budget:]
        add = line if not cur else cur + "\n" + line
        if len(add) > budget:
            parts.append(cur)
            cur = line
        else:
            cur = add
    if cur:
        parts.append(cur)
    n = len(parts)
    return [f"{p}\n({i}/{n})" for i, p in enumerate(parts, 1)]


def _idx(root):
    return sc.load_index(Path(root))


_MODE_WORDS = {"accuracy": "accuracy", "точность": "accuracy", "economy": "economy", "экономия": "economy", "эконом": "economy",
               "all": None, "все": None, "общий": None, "": None}


def parse_mode(word: str | None):
    """-> "accuracy" | "economy" | None; raises ValueError for an unknown word."""
    w = (word or "").strip().lower()
    if w not in _MODE_WORDS:
        raise ValueError("режим: accuracy | economy | all")
    return _MODE_WORDS[w]


def _line(e: dict, mode=None) -> str:
    e = sc.mode_view(e, mode)
    pr = "-" if e["priority"] is None else e["priority"]
    t = "общее" if e["type"] == sc.ALL else e["type"]
    return f"{t}: {e['status']} | потенциал {e['headroom']} | приоритет {pr} | проверено {e['checked']} | журнал {(e.get('journal') or {}).get('count', 0)}"


def block_view(block: str, root=ROOT, limit: int = TG_LIMIT, mode=None) -> list[str]:
    idx = _idx(root)
    if block not in idx["blocks"]:
        return ["Нет блока «%s». Блоки: " % block + ", ".join(idx["blocks"])]
    by = {e["type"]: e for e in idx["entries"] if e["block"] == block}
    out = [f"Блок {block}" + (f" (режим {mode})" if mode else "")]
    for t in idx["types"] + [sc.ALL]:
        e = by[t]
        out.append("• " + _line(e, mode))
        if e["next_step"]:
            out.append("   дальше: " + e["next_step"][:160])
    return split_message("\n".join(out), limit)


def type_view(typ: str, root=ROOT, limit: int = TG_LIMIT, mode=None) -> list[str]:
    idx = _idx(root)
    if typ not in idx["types"]:
        return ["Нет типа «%s». Типы: " % typ + ", ".join(idx["types"])]
    out = [f"Тип {typ}: все блоки" + (f" (режим {mode})" if mode else "")]
    ents = [e for e in idx["entries"] if e["type"] == typ]
    for e in ents:
        e = sc.mode_view(e, mode)
        pr = "-" if e["priority"] is None else e["priority"]
        out.append(f"• {e['block']}: {e['status']} | {e['headroom']} | приоритет {pr}")
    c = sc.count_by_status(ents)
    out.append("итого: " + ", ".join(f"{k} {v}" for k, v in c.items() if v))
    return split_message("\n".join(out), limit)


def cell_view(block: str, typ: str, root=ROOT, limit: int = TG_LIMIT) -> list[str]:
    p = sc.cell_path(Path(root), block, typ)
    if not p.exists():
        return [f"Нет ячейки {block}/{typ}."]
    return split_message(p.read_text(encoding="utf-8"), limit)


def matrix_view(root=ROOT, limit: int = TG_LIMIT, mode=None) -> list[str]:
    rows = sc.matrix(Path(root))
    c = sc.count_by_status(_idx(root)["entries"])
    rows += ["", "легенда: works=works-unexplored, headroom=explored-headroom, SAT=saturated (статус общий для режимов)",
             "итого: " + ", ".join(f"{k} {v}" for k, v in c.items() if v)]
    if mode:
        rows += ["", f"потенциал роста, режим {mode} (none/low/med/high/?=нет данных):"] + sc.matrix_headroom(Path(root), mode)
    return split_message("\n".join(rows), limit)


def priority_view(top: int = 10, root=ROOT, limit: int = TG_LIMIT, mode=None) -> list[str]:
    root = Path(root)
    unit = sc.MODE_METRIC.get(mode, "очки")
    out = [f"Приоритеты{' режима ' + mode if mode else ' (общий рейтинг)'}: прирост ({unit}) × достоверность / работа"]
    for i, e in enumerate(sc.ranking(root, mode)[:top], 1):
        out.append(f"{i}. {e['block']}/{e['type']}: {e['priority']} (прирост {e['gain_pts']}, достоверность {e['confidence']}, "
                   f"работа {e['work']}) — {e['status']}")
    un = sc.unrated(root, mode)
    sat = ", ".join(f"{e['block']}/{e['type']}" for e in _idx(root)["entries"] if e["status"] == "saturated")
    out.append(f"Без оценки прироста (нет данных): {len(un)} ячеек. Насыщено (не трогать): {sat or 'нет'}")
    return split_message("\n".join(out), limit)


def _journal_line(r: dict) -> str:
    return (f"{r['date']} {r['name']} [{r['verdict']}] {r['setbudget']}, повт. {r['repeats']}: {r['result']} "
            f"(режим: {r.get('mode', 'не указан')}; в коде: {r['accepted']}; {r['who']})")


def journal_view(block: str, typ: str, n: int = 5, root=ROOT, limit: int = TG_LIMIT) -> dict:
    """-> {"messages": [...], "document": str | None, "filename": str}. The last n rows (newest first) go into messages;
    if the journal has more, `document` holds the complete journal as markdown (to send as a file)."""
    root = Path(root)
    fn = f"journal_{block}_{typ}.md"
    p = sc.cell_path(root, block, typ)
    if not p.exists():
        return {"messages": [f"Нет ячейки {block}/{typ}."], "document": None, "filename": fn}
    rows, errs = sc.read_journal(root, block, typ)
    if rows is None:
        return {"messages": [f"В ячейке {block}/{typ} нет журнала (неприменима или не мигрирована)."], "document": None, "filename": fn}
    if not rows:
        return {"messages": [f"Журнал {block}/{typ}: записей нет."], "document": None, "filename": fn}
    n = max(1, int(n))
    last = rows[::-1][:n]
    head = f"Журнал {block}/{typ}: записей {len(rows)}" + (f", показаны последние {len(last)}" if len(rows) > n else "")
    msgs = split_message(chr(10).join([head] + ["• " + _journal_line(r) for r in last]), limit)
    doc = None
    if len(rows) > n:
        doc = chr(10).join([f"# Журнал гипотез {block} × {typ}", ""] + sc.journal_header() + [sc.format_row(r) for r in rows]) + chr(10)
    return {"messages": msgs, "document": doc, "filename": fn}


def handle_command(text: str, root=ROOT) -> dict | None:
    """Parse «журнал <блок> <тип> [N]», «схема [режим]», «приоритеты [режим]» (also /journal /scheme /priorities);
    None if the text is none of them. Always {"messages", "document", "filename"}."""
    parts = text.strip().split()
    cmd = parts[0].lower().lstrip("/") if parts else ""
    if cmd in ("схема", "scheme", "приоритеты", "priorities"):
        try:
            mode = parse_mode(parts[1] if len(parts) > 1 else None)
        except ValueError as e:
            return {"messages": [str(e)], "document": None, "filename": ""}
        msgs = matrix_view(root, mode=mode) if cmd in ("схема", "scheme") else priority_view(10, root, mode=mode)
        return {"messages": msgs, "document": None, "filename": ""}
    if cmd not in ("журнал", "journal"):
        return None
    if len(parts) < 3:
        return {"messages": ["Формат: журнал <блок> <тип> [N]. Тип: coding, math, search, data_analysis или _all."], "document": None, "filename": ""}
    n = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 5
    return journal_view(parts[1], parts[2], n, root)


def selftest() -> int:
    long = "\n".join(f"строка {i} " + "я" * 100 for i in range(200))
    parts = split_message(long, 4096)
    assert len(parts) > 1 and all(len(p) <= 4096 for p in parts), [len(p) for p in parts]
    assert f"(1/{len(parts)})" in parts[0]
    body = "\n".join(p.rsplit("\n(", 1)[0] for p in parts)
    assert body == long
    assert split_message("короткий") == ["короткий"]
    assert all(len(p) <= 100 for p in split_message("x" * 1000, 100))
    tmp = Path(tempfile.mkdtemp(prefix="status_views_selftest_"))
    try:
        sc._mini(tmp)
        v = block_view("blk", tmp)
        assert len(v) == 1 and "a: works-unexplored" in v[0] and "b: n/a" in v[0] and "общее:" in v[0]
        assert "blk: works-unexplored" in type_view("a", tmp)[0]
        assert "Нет блока" in block_view("zzz", tmp)[0] and "Нет типа" in type_view("zzz", tmp)[0]
        assert "# blk × a" in cell_view("blk", "a", tmp)[0] and "Нет ячейки" in cell_view("blk", "zzz", tmp)[0]
        assert "blk" in matrix_view(tmp)[0] and "blk/a" in priority_view(5, tmp)[0]
        for i in range(8):
            sc.log_entry(tmp, "blk", "a", f"exp{i}", "clarify", "очки 1 против 2, -1 [-2; 0]", date=f"2026-03-0{i+1}", repeats=2, who="тест")
        jv = journal_view("blk", "a", 3, tmp)
        assert "записей 8, показаны последние 3" in jv["messages"][0] and "exp7" in jv["messages"][0] and "exp4" not in jv["messages"][0]
        assert jv["document"] and jv["document"].count("exp") >= 8 and jv["filename"] == "journal_blk_a.md"
        assert journal_view("blk", "a", 20, tmp)["document"] is None
        assert "записей нет" in journal_view("blk", sc.ALL, 5, tmp)["messages"][0]
        assert handle_command("журнал blk a 2", tmp)["messages"][0].startswith("Журнал blk/a") and handle_command("привет", tmp) is None
        assert "Формат" in handle_command("журнал blk", tmp)["messages"][0]
        assert "режим economy" in " ".join(matrix_view(tmp, mode="economy")) and "режима economy" in priority_view(5, tmp, mode="economy")[0]
        assert "режима accuracy" in handle_command("приоритеты accuracy", tmp)["messages"][0]
        assert "общий рейтинг" in handle_command("/priorities", tmp)["messages"][0]
        assert "потенциал роста, режим economy" in handle_command("схема economy", tmp)["messages"][0]
        assert "режим: не указан" in journal_view("blk", "a", 2, tmp)["messages"][0]
        assert "accuracy | economy" in handle_command("схема fast", tmp)["messages"][0]
        many = [sc.log_entry(tmp, "blk", "a", f"long{i}", "accept", "х" * 300, date="2026-04-01") for i in range(40)]
        assert all(len(m) <= TG_LIMIT for m in journal_view("blk", "a", 40, tmp)["messages"])
        assert len(journal_view("blk", "a", 40, tmp)["messages"]) > 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if (ROOT / "index.json").exists():  # real data: every view fits the Telegram limit
        idx = _idx(ROOT)
        for b in idx["blocks"]:
            assert all(len(m) <= TG_LIMIT for m in block_view(b))
            for t in [sc.ALL] + idx["types"]:
                assert all(len(m) <= TG_LIMIT for m in cell_view(b, t))
        for t in idx["types"]:
            assert all(len(m) <= TG_LIMIT for m in type_view(t))
        assert all(len(m) <= TG_LIMIT for m in matrix_view() + priority_view(50))
        for md in sc.MODES:
            assert all(len(m) <= TG_LIMIT for m in matrix_view(mode=md) + priority_view(50, mode=md) + type_view("math", mode=md) + block_view("batcher", mode=md))
        for b in idx["blocks"]:
            for t in [sc.ALL] + idx["types"]:
                assert all(len(m) <= TG_LIMIT for m in journal_view(b, t, 100)["messages"]), (b, t)
    print("status_views selftest: ok")
    return 0


def main(argv=None) -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--mode", choices=sc.MODES, help="режим для block/type/matrix/priority")
    ap.add_argument("cmd", choices=["block", "type", "cell", "matrix", "priority", "journal", "selftest"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    r = a.root
    msgs = {"block": lambda: block_view(a.args[0], r, mode=a.mode), "type": lambda: type_view(a.args[0], r, mode=a.mode),
            "cell": lambda: cell_view(a.args[0], a.args[1], r), "matrix": lambda: matrix_view(r, mode=a.mode),
            "priority": lambda: priority_view(10, r, mode=a.mode),
            "journal": lambda: journal_view(a.args[0], a.args[1], int(a.args[2]) if len(a.args) > 2 else 5, r)["messages"]}[a.cmd]()
    print(("\n" + "-" * 20 + "\n").join(msgs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
