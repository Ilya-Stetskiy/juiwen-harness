#!/usr/bin/env python3
"""Telegram adapter of the autolab control (see autolab_ctl.py). Runs INSIDE the existing JiuwenSwarm bot process.

It does not poll Telegram: it registers handlers on the python-telegram-bot `Application` that JiuwenSwarm's TelegramChannel already
runs, so there is still exactly one getUpdates consumer for the token (a second one would answer 409 Conflict and break the bot).
All decisions (whitelist of user ids, confirmations, journal) are in autolab_ctl.py; its whitelist is its own file
~/.config/autolab/telegram_allow.json, independent of the bot's `allow_from`.

Registered:
  * `/autolab ...` (CommandHandler) and the confirmation buttons (CallbackQueryHandler, data "al:...");
  * Russian words without a slash (`принять X`, `отчёт X`, `рецензия X`, `статус`, ...), only for whitelisted users: the handler runs first
    (group -1) and stops the other handlers, so the swarm agent does not also answer; for anybody else it does nothing and the message
    goes to the agent as before;
  * an outbox loop: autolab.py writes files into ~/.config/autolab/tg_outbox/*.json (a finished hypothesis: text + report .md); the loop
    sends them with the bot's own transport (sendMessage / sendDocument) and deletes them; a failed one stays and is retried.

Hook-up: the small patch tools/autolab_tg_vendor.patch to jiuwenswarm/gateway/channel_manager/im_platforms/telegram/telegram_connect.py
(that channel has no plugin or command-registration mechanism: handlers are hard-coded in `start()`):
    from autolab_tg_adapter import register; register(self._application)
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUSSIAN_STATUS = r"^\s*(@\w+\s+)?((блок|тип|ячейка|описание|журнал|история|гипотеза|карточка|отклонить|отложить)\s+[a-z_][a-z0-9_/ ]*|своя\s+[a-z_][a-z0-9_]*[/\s]+[a-z_]+[\s\S]*|привязать\s+ab_\w+\s+[a-z_/ ,]+|схема|приоритеты(\s+(accuracy|economy))?|блоки|типы|отложенные|журнал|гипотезы(\s+все)?)\s*$"   # ordinary words: only with a latin argument or alone
RUSSIAN = r"^\s*(@\w+\s+)?(принять|отчёт|отчет|рецензия|статус|очередь|стоп|продолжить|расход|меню|список|помощь|команды|ключ|вердикт|идеи|идея|цикл|повтор|проверка|автопилот|шеф|стек|локально)(\s|$)"

COMMAND_RE = "(?:" + RUSSIAN + ")|(?:" + RUSSIAN_STATUS + ")"   # what register() lets through to on_russian: BOTH lists (a word missing here goes to the agent)


def clean(text: str) -> str:
    """Drop a leading @botname (`@bot повтор X 2` is the same command as `повтор X 2`)."""
    return re.sub(r"^\s*@\w+\s+", "", text or "")


def _markup(rep: dict, button_cls, markup_cls):
    if not rep.get("buttons"):
        return None
    return markup_cls([[button_cls(b["label"], callback_data=b["data"]) for b in row] for row in rep["buttons"]])


async def _send(message, rep: dict, button_cls, markup_cls) -> None:
    await _send_one(message, rep, button_cls, markup_cls)
    for extra in rep.get("more") or []:   # long views (scheme status) arrive as several messages
        await message.reply_text(extra)


async def _send_one(message, rep: dict, button_cls, markup_cls) -> None:
    mk = _markup(rep, button_cls, markup_cls) if button_cls else None
    if rep.get("document") and Path(rep["document"]).exists():
        with open(rep["document"], "rb") as fh:
            await message.reply_document(document=fh, filename=Path(rep["document"]).name, caption=rep["text"][:1000])
        if len(rep["text"]) > 1000:
            await message.reply_text(rep["text"], reply_markup=mk)
        elif mk:
            await message.reply_text("↑", reply_markup=mk)
    else:
        await message.reply_text(rep["text"], reply_markup=mk)


async def on_command(update, context, ctl=None, button_cls=None, markup_cls=None) -> None:
    """`/autolab ...` or a Russian command from any chat; the user id decides (autolab_ctl.allowed), a rejected user gets no reply."""
    import autolab_ctl
    ctl = ctl or autolab_ctl
    if not update.message or not update.effective_user:
        return
    ch = update.effective_chat
    rep = await asyncio.to_thread(ctl.handle, update.effective_user.id, clean(update.message.text), getattr(ch, "id", None), getattr(ch, "type", "private"))
    if rep:
        await _send(update.message, rep, button_cls, markup_cls)


async def on_id(update, context, ctl=None) -> None:
    """/id: tells anybody their user id and the chat id (needed to fill telegram_allow.json); no privileges involved."""
    import autolab_ctl
    ctl = ctl or autolab_ctl
    if not update.message or not update.effective_user:
        return
    ch = update.effective_chat
    rep = ctl.whoami(update.effective_user.id, getattr(ch, "id", None), getattr(ch, "type", "private"))
    await update.message.reply_text(rep["text"])


async def on_russian(update, context, ctl=None, button_cls=None, markup_cls=None, stop_exc=None) -> None:
    """A message that starts with a Russian command word. Whitelisted users: handled here and the agent is NOT asked."""
    import autolab_ctl
    ctl = ctl or autolab_ctl
    if not update.message or not update.effective_user:
        return
    ch = update.effective_chat
    if not (re.match(RUSSIAN, update.message.text or "", re.I) or re.match(RUSSIAN_STATUS, update.message.text or "", re.I)):
        hc = getattr(ctl, "handle_capture", None)      # not a command: maybe the text of a hypothesis that this user is composing
        if hc:
            rep = await asyncio.to_thread(hc, update.effective_user.id, getattr(ch, "id", None), update.message.text or "", getattr(ch, "type", "private"))
            if rep:
                await _send(update.message, rep, button_cls, markup_cls)
                if stop_exc:
                    raise stop_exc()
        return
    m = re.match(r"^\s*@(\w+)\s", update.message.text or "")
    me = getattr(getattr(context, "bot", None), "username", None)
    if m and me and m.group(1).lower() != me.lower():
        return   # addressed to somebody else
    if not ctl.allowed(update.effective_user.id, getattr(ch, "id", None), getattr(ch, "type", "private")):
        return   # not ours: the normal handlers (the swarm agent) take it
    await on_command(update, context, ctl, button_cls, markup_cls)
    if stop_exc:
        raise stop_exc()


async def on_button(update, context, ctl=None, button_cls=None, markup_cls=None) -> None:
    import autolab_ctl
    ctl = ctl or autolab_ctl
    q = update.callback_query
    if not q or not q.from_user:
        return
    ch = getattr(q.message, "chat", None)
    rep = await asyncio.to_thread(ctl.handle_callback, q.from_user.id, q.data or "", getattr(ch, "id", None), getattr(ch, "type", "private"))
    try:
        await q.answer()
    except Exception:  # noqa: BLE001
        pass
    if rep:
        await _send(q.message, rep, button_cls, markup_cls)


def outbox_dir() -> Path:
    import autolab_ctl
    return autolab_ctl.CFG_DIR / "tg_outbox"


async def flush_outbox(bot, button_cls=None, markup_cls=None, directory: Path | None = None) -> int:
    """Send every queued item; returns how many were sent. An item that fails stays for the next round."""
    d = directory or outbox_dir()
    sent = 0
    import time as _t
    for st in (d.glob("*.sending") if d.exists() else []):      # a claim abandoned by a crashed process: give it back
        try:
            if _t.time() - st.stat().st_mtime > 600:
                st.rename(st.with_suffix(".json"))
        except OSError:
            pass
    for f0 in sorted(d.glob("*.json")) if d.exists() else []:
        f = f0.with_suffix(".sending")
        try:
            f0.rename(f)     # atomic claim: two bot processes (an old one during a restart, a new one) must not send the same item twice
        except OSError:
            continue
        try:
            it = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            f.rename(f.with_suffix(".bad"))
            continue
        try:
            for chat in it["to"]:
                mk = _markup({"buttons": it.get("buttons")}, button_cls, markup_cls) if (button_cls and it.get("buttons")) else None
                if it.get("file") and Path(it["file"]).exists():
                    with open(it["file"], "rb") as fh:
                        await bot.send_document(chat_id=int(chat), document=fh, filename=Path(it["file"]).name, caption=it["text"][:1000])
                    if len(it["text"]) > 1000:
                        await bot.send_message(chat_id=int(chat), text=it["text"], reply_markup=mk)
                    elif mk:
                        await bot.send_message(chat_id=int(chat), text="↑", reply_markup=mk)   # a document caption cannot carry the decision buttons
                else:
                    await bot.send_message(chat_id=int(chat), text=it["text"], reply_markup=mk)
            f.unlink()
            sent += 1
        except Exception as ex:  # noqa: BLE001 - keep the file, try again later
            try:
                f.rename(f0)
            except OSError:
                pass
            try:
                import autolab_ctl
                autolab_ctl.journal("-", "outbox_error", file=f.name, error=f"{type(ex).__name__}: {str(ex)[:120]}")
            except Exception:  # noqa: BLE001
                pass
    return sent


async def outbox_loop(bot, interval: float = 15.0) -> None:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    while True:
        try:
            await flush_outbox(bot, InlineKeyboardButton, InlineKeyboardMarkup)
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(interval)


def register(application) -> None:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ApplicationHandlerStop, CallbackQueryHandler, CommandHandler, MessageHandler, filters

    async def cmd(update, context):
        await on_command(update, context, None, InlineKeyboardButton, InlineKeyboardMarkup)

    async def btn(update, context):
        await on_button(update, context, None, InlineKeyboardButton, InlineKeyboardMarkup)

    async def rus(update, context):
        await on_russian(update, context, None, InlineKeyboardButton, InlineKeyboardMarkup, ApplicationHandlerStop)

    async def myid(update, context):
        await on_id(update, context)

    application.add_handler(CommandHandler("autolab", cmd))
    application.add_handler(CommandHandler("id", myid))
    application.add_handler(CallbackQueryHandler(btn, pattern=r"^al:"))
    import autolab_ctl

    class _Composing(filters.MessageFilter):      # a user who pressed «Своя гипотеза» and now writes its text
        def filter(self, message) -> bool:
            return bool(message.from_user and autolab_ctl.has_capture(message.from_user.id, message.chat_id))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & (filters.Regex(re.compile(COMMAND_RE, re.I)) | _Composing()), rus), group=-1)
    try:   # the outbox loop lives in the bot's event loop (register() is called from the async start())
        asyncio.get_running_loop().create_task(outbox_loop(application.bot))
    except RuntimeError:
        pass


def selftest() -> None:
    import tempfile
    sent = []

    class Msg:
        text = "/autolab status"

        async def reply_text(self, text, reply_markup=None):
            sent.append(("text", text, reply_markup))

        async def reply_document(self, document, filename=None, caption=None):
            sent.append(("doc", filename, caption, document.read()))

    class User:
        id = 5

    class Chat:
        id = 5
        type = "private"

    class Upd:
        message = Msg()
        effective_user = User()
        effective_chat = Chat()

    class Ctl:
        calls = []

        @staticmethod
        def allowed(uid, chat=None, chat_type="private"):
            return uid == 5 and chat_type == "private"

        @staticmethod
        def handle(uid, text, chat=None, chat_type="private"):
            Ctl.calls.append((uid, text))
            return None if uid != 5 else {"text": "ok", "buttons": [[{"label": "Run", "data": "al:ok:abc"}]], "document": ""}

        @staticmethod
        def handle_callback(uid, data, chat=None, chat_type="private"):
            Ctl.calls.append((uid, data))
            return {"text": "done", "buttons": [], "document": ""}

    class Btn:
        def __init__(self, label, callback_data=None):
            self.label, self.data = label, callback_data

    mk = lambda rows: rows
    asyncio.run(on_command(Upd(), None, Ctl, Btn, mk))
    assert Ctl.calls == [(5, "/autolab status")] and sent[0][1] == "ok" and sent[0][2][0][0].data == "al:ok:abc"
    Upd.effective_user = type("U", (), {"id": 9})()
    asyncio.run(on_command(Upd(), None, Ctl, Btn, mk))
    assert len(sent) == 1   # rejected user: silence

    class Stop(Exception):
        pass
    Upd.message = type("M", (Msg,), {"text": "принять ab_x"})()
    Upd.effective_user = User()
    try:
        asyncio.run(on_russian(Upd(), None, Ctl, Btn, mk, Stop))
        raise AssertionError("not stopped")
    except Stop:
        pass
    assert Ctl.calls[-1] == (5, "принять ab_x")
    Upd.effective_user = type("U", (), {"id": 9})()
    n = len(Ctl.calls)
    asyncio.run(on_russian(Upd(), None, Ctl, Btn, mk, Stop))   # not whitelisted: no stop, no call
    assert len(Ctl.calls) == n
    Upd.message = type("M", (Msg,), {"text": "привет, принять заказ"})()
    Upd.effective_user = User()
    asyncio.run(on_russian(Upd(), None, Ctl, Btn, mk, Stop))   # not a command word at the start
    assert len(Ctl.calls) == n
    # the text of a hypothesis that this user is composing is not a command: it goes to handle_capture; without a pending hypothesis nothing is answered
    Ctl.captures = []
    Ctl.handle_capture = staticmethod(lambda uid, chat, text, chat_type="private": (Ctl.captures.append((uid, chat, text)) or {"text": "принято как гипотеза", "buttons": [], "document": ""}) if text.startswith("сделай") else None)
    Upd.message = type("M", (Msg,), {"text": "сделай порядок по цене"})()
    k = len(sent)
    try:
        asyncio.run(on_russian(Upd(), None, Ctl, Btn, mk, Stop))
        raise AssertionError("not stopped")
    except Stop:
        pass
    assert Ctl.captures == [(5, 5, "сделай порядок по цене")] and sent[k][1] == "принято как гипотеза"
    Upd.message = type("M", (Msg,), {"text": "обычная болтовня"})()
    asyncio.run(on_russian(Upd(), None, Ctl, Btn, mk, Stop))   # nobody is composing: the agent gets it
    assert len(sent) == k + 1
    del Ctl.handle_capture

    # `@bot команда` is the same command; `@someone команда` is not ours
    Upd.message = type("M", (Msg,), {"text": "@mybot повтор ab_x 2"})()
    Upd.effective_user = User()
    ctxt = type("C", (), {"bot": type("B", (), {"username": "MyBot"})()})()
    try:
        asyncio.run(on_russian(Upd(), ctxt, Ctl, Btn, mk, Stop))
        raise AssertionError("not stopped")
    except Stop:
        pass
    assert Ctl.calls[-1] == (5, "повтор ab_x 2"), Ctl.calls[-1]
    Upd.message = type("M", (Msg,), {"text": "@ivan повтор ab_x 2"})()
    n = len(Ctl.calls)
    asyncio.run(on_russian(Upd(), ctxt, Ctl, Btn, mk, Stop))
    assert len(Ctl.calls) == n
    # a group chat the controller does not list: no reply, no stop of the agent's handlers
    Upd.message = type("M", (Msg,), {"text": "статус"})()
    Upd.effective_user = User()
    Upd.effective_chat = type("G", (), {"id": -100, "type": "supergroup"})()
    n = len(Ctl.calls)
    asyncio.run(on_russian(Upd(), None, Ctl, Btn, mk, Stop))
    assert len(Ctl.calls) == n
    Upd.effective_chat = Chat()
    d = Path(tempfile.mkdtemp())
    (d / "r.md").write_text("# отчёт\n", encoding="utf-8")
    asyncio.run(_send(Msg(), {"text": "short", "buttons": [], "document": str(d / "r.md")}, Btn, mk))
    assert sent[-1][0] == "doc" and sent[-1][1] == "r.md" and "отчёт".encode() in sent[-1][3]

    class Q:
        data = "al:ok:abc"
        from_user = User()

        class message:
            chat = Chat()

            @staticmethod
            async def reply_text(text, reply_markup=None):
                sent.append(("text", text, reply_markup))

        @staticmethod
        async def answer():
            return None

    class UpdB:
        callback_query = Q()
    asyncio.run(on_button(UpdB(), None, Ctl, Btn, mk))
    assert sent[-1][1] == "done" and Ctl.calls[-1] == (5, "al:ok:abc")

    ob = d / "ob"
    ob.mkdir()
    (ob / "1.json").write_text(json.dumps({"to": ["5", "6"], "text": "hypothesis", "file": str(d / "r.md"), "caption": "", "buttons": []}), encoding="utf-8")
    (ob / "2.json").write_text(json.dumps({"to": ["5"], "text": "plain", "file": "", "caption": "", "buttons": []}), encoding="utf-8")
    got = []

    class Bot:
        fail = False

        @staticmethod
        async def send_document(chat_id, document, filename=None, caption=None):
            got.append(("doc", chat_id, filename))

        @staticmethod
        async def send_message(chat_id, text, reply_markup=None):
            if Bot.fail:
                raise RuntimeError("network")
            got.append(("msg", chat_id, text))
    assert asyncio.run(flush_outbox(Bot, Btn, mk, ob)) == 2 and not list(ob.glob("*.json"))
    assert ("doc", 5, "r.md") in got and ("doc", 6, "r.md") in got and ("msg", 5, "plain") in got
    (ob / "3.json").write_text(json.dumps({"to": ["5"], "text": "x", "file": "", "caption": "", "buttons": []}), encoding="utf-8")
    Bot.fail = True
    assert asyncio.run(flush_outbox(Bot, Btn, mk, ob)) == 0 and (ob / "3.json").exists()
    # two bot processes (an old one during a restart and the new one) flushing the same outbox send each item once
    Bot.fail = False
    got.clear()
    (ob / "3.json").unlink()
    for i in range(4):
        (ob / f"d{i}.json").write_text(json.dumps({"to": ["5"], "text": f"item{i}", "file": "", "caption": "", "buttons": []}), encoding="utf-8")

    async def two():
        return await asyncio.gather(flush_outbox(Bot, Btn, mk, ob), flush_outbox(Bot, Btn, mk, ob))
    r2 = asyncio.run(two())
    assert sum(r2) == 4 and sorted(x[2] for x in got) == ["item0", "item1", "item2", "item3"], (r2, got)
    (ob / "stale.sending").write_text(json.dumps({"to": ["5"], "text": "stale", "file": "", "caption": "", "buttons": []}), encoding="utf-8")
    import os as _os
    _os.utime(ob / "stale.sending", (1, 1))
    assert asyncio.run(flush_outbox(Bot, Btn, mk, ob)) == 1      # a claim abandoned by a crashed process is given back and sent
    # scheme status words: commands only with a latin argument (or alone); ordinary Russian chat is left to the agent
    ok = lambda t: bool(re.match(RUSSIAN, t, re.I) or re.match(RUSSIAN_STATUS, t, re.I))
    for t in ("блок batcher", "тип math", "ячейка batcher math", "описание predictor/_all", "схема", "приоритеты", "блоки", "типы", "журнал batcher math", "история batcher/math 10", "гипотеза ab_x_test", "отклонить ab_x", "отложить ab_x", "отложенные", "журнал", "гипотезы", "приоритеты economy", "приоритеты accuracy", "гипотезы все", "своя batcher/math попробовать другой порог", "@mybot схема", "привязать ab_x batcher/math", "Блок censor"):
        assert ok(t), t
    for t in ("тип задачи такой", "журнал событий вчера", "история была долгой", "отложить встречу", "гипотезы про батчи", "журнал событий", "гипотеза про батчи", "отклонить предложение", "своя идея у меня есть", "схема работает плохо", "блок питания сломался", "привязать это к делу", "хорошая схема", "тип"):
        assert not ok(t), t
    assert all(re.search(COMMAND_RE, t, re.I) for t in ("блок batcher", "схема", "статус", "приоритеты"))   # the registration filter, not only on_russian
    n_sent = len(sent)
    Upd.message = type("M", (Msg,), {"text": "схема"})()
    Upd.effective_user = User()
    Ctl.handle = staticmethod(lambda uid, text, chat=None, chat_type="private": {"text": "part1", "buttons": [], "document": "", "more": ["part2", "part3"]})
    asyncio.run(on_command(Upd(), None, Ctl, Btn, mk))
    assert [x[1] for x in sent[n_sent:]] == ["part1", "part2", "part3"], sent[n_sent:]
    print("autolab_tg_adapter selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        print(__doc__)
