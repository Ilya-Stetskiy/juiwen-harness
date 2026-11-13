"""README figure: spend over one run with HARNESS_FALLBACK=late, from its journal (lab tool, not in the image).

usage: python tools/plot_readme.py <results>/prod_fallback_late_60k docs/img/spend-over-run.png [arm]

x = seconds since run_start, y = the ledger total after each finished task (task_end.total), arm `prod_late` by default.
Three zones: the main queue (up to late_phase), the tool agent (late phase, up to the budget) and, above the budget, what
the agent tasks left unfunded would have cost by the harness's own estimate (executor.fallback_estimate): an estimate,
drawn dashed. Points come from cells.jsonl (correct x reward), split by phase.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.executor import fallback_estimate  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
MAIN, AGENT, OUT = "#2a78d6", "#eb6834", "#8a8984"


def fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def load(exp: Path, arm: str):
    t0 = budget = late_t = late_spent = None
    pts, agent_pts, typ, unfunded = [(0.0, 0.0)], [], {}, []
    for line in (exp / "journals" / f"{arm}_1.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        ev = e.get("event")
        if ev == "run_start":
            t0, budget = e["ts"], e["budget"]
        elif ev == "task_start":
            typ[e["task"]] = e["type"]
        elif ev == "late_phase":
            late_t, late_spent = e["ts"] - t0, e["spent"]
            pts.append((late_t, late_spent))
        elif ev == "task_end" and e.get("status") == "unfunded":
            unfunded.append(e["task"])
        elif ev == "task_end":   # deferred tasks too: their pipeline calls are paid
            pts.append((e["ts"] - t0, e["total"]))
            if late_t is not None and e.get("status") != "deferred":
                agent_pts.append(pts[-1])
        elif ev == "run_end" and e["spent"] > pts[-1][1] + 1:
            pts.append((e["ts"] - t0, e["spent"]))   # calls cut by the hard cap: no task_end, the ledger still paid them
    late_tasks, main_n, score = set(), 0, {"main": 0.0, "agent": 0.0}
    for line in (exp / "journals" / f"{arm}_1.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        if e.get("event") == "task_resume":
            late_tasks.add(e["task"])
    for line in (exp / "cells.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        if c["arm"] != arm or c["repeat"] != 1:
            continue
        phase = "agent" if c["task_id"] in late_tasks else "main"
        score[phase] += c["reward"] if c["correct"] else 0.0
        main_n += phase == "main" and c["task_id"] not in unfunded
    est = sum(fallback_estimate(typ.get(t, "")) for t in unfunded)
    return dict(pts=pts, agent_pts=agent_pts, budget=budget, late_t=late_t, late_spent=late_spent, est=est,
                unfunded=unfunded, late_n=len(late_tasks), main_n=main_n, score=score)


def main(exp: str, out: str, arm: str = "prod_late") -> None:
    r = load(Path(exp), arm)
    pts, budget, lt, ls = r["pts"], r["budget"], r["late_t"], r["late_spent"]
    t_end, y_end = pts[-1]
    top = max(budget, y_end + r["est"]) * 1.1
    x_max = t_end + (t_end - lt) * 1.9

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "hatch.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # zones: main queue (bottom left), agent (bottom right), outside the budget (top right)
    zones = [((0, 0), lt, ls, MAIN, "//"), ((lt, 0), x_max - lt, budget, AGENT, "\\\\"),
             ((lt, budget), x_max - lt, top - budget, OUT, "..")]
    for (x0, y0), w, h, color, hatch in zones:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=color, alpha=0.07, edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=color, alpha=0.35, hatch=hatch, lw=0,
                               zorder=0))

    xs, ys = zip(*pts)
    ax.step(xs, ys, where="post", color=INK, lw=2, zorder=3)
    ax.scatter(*zip(*r["agent_pts"]), s=64, color=AGENT, edgecolor=SURFACE, linewidth=2, zorder=4)
    if r["est"]:
        ax.plot([t_end, t_end], [y_end, y_end + r["est"]], color=INK, lw=2, ls=(0, (3, 3)), zorder=3)

    ax.annotate(f"стоп по потолку: {t_end:.0f} с,\nпотрачено {fmt(y_end)}", (t_end, y_end), xytext=(10, -34),
                textcoords="offset points", fontsize=9.5, color=INK2, va="top",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1})
    ax.axhline(budget, color=INK2, lw=1, ls=(0, (5, 4)), zorder=2)
    ax.text(6, budget + top * 0.012, f"бюджет {fmt(budget)} · жёсткий потолок", color=INK2, fontsize=9.5)

    def label(x, y, title, body, ha="left"):
        ax.text(x, y, title, fontsize=11.5, fontweight="bold", color=INK, ha=ha, va="top", zorder=5)
        ax.text(x, y - top * 0.055, body, fontsize=9.5, color=INK2, ha=ha, va="top", linespacing=1.4, zorder=5)

    label(8, ls * 0.92, "основная очередь",
          f"конвейер, 1–2 вызова на задачу\n{r['main_n']} задач · {fmt(ls)} токенов\n{fmt(r['score']['main'])} очков")
    label(lt + 8, ls * 0.62, "агент",
          f"в конце очереди\n{r['late_n']} задачи\n{fmt(y_end - ls)} токенов\n{fmt(r['score']['agent'])} очков")
    label(t_end + 10, top * 0.985, "решение вне бюджета",
          f"{len(r['unfunded'])} задача не запущена:\nоценка агента {fmt(r['est'])}")

    ax.set_xlabel("секунды от начала прогона", color=INK2)
    ax.set_ylabel("накопленный расход, токенов", color=INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v)))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, top)
    ax.set_title(f"Расход по ходу боевого прогона ({Path(exp).name}, ветка {arm})", loc="left", color=INK,
                 fontsize=12.5, pad=12)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    main(*sys.argv[1:4])
