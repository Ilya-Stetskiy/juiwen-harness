"""Task loading. The final set has no `expected` field: never read it."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskInfo:
    task_id: str
    task_type: str
    instruction: str
    reward: float
    check: str
    dir: Path
    workspace: Path | None
    n_files: int
    workspace_bytes: int


def _workspace_stats(ws: Path) -> tuple[int, int]:
    n = size = 0
    for p in ws.rglob("*"):
        if p.is_file():
            n += 1
            try:
                size += p.stat().st_size
            except OSError:
                pass
    return n, size


def load_tasks(root: str | Path) -> list[TaskInfo]:
    tasks: list[TaskInfo] = []
    for d in sorted(Path(root).iterdir()):
        tj = d / "task.json"
        if not (d.is_dir() and tj.is_file()):
            continue
        try:
            raw = json.loads(tj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ws = d / "workspace"
        ws = ws if ws.is_dir() else None
        n, size = _workspace_stats(ws) if ws else (0, 0)
        try:
            reward = float(raw.get("reward", 1))
        except (TypeError, ValueError):
            reward = 1.0
        tasks.append(
            TaskInfo(
                task_id=str(raw.get("task_id") or d.name),
                task_type=str(raw.get("task_type") or d.name.rsplit("_", 1)[0]),
                instruction=str(raw.get("instruction", "")),
                reward=reward,
                check=str(raw.get("check", "")),
                dir=d,
                workspace=ws,
                n_files=n,
                workspace_bytes=size,
            )
        )
    return tasks
