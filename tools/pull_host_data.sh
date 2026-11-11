#!/usr/bin/env bash
# Mirror what the lab produces on the GPU host to this machine (read-only on the host; safe to run any time, as often as you like).
#   bash tools/pull_host_data.sh            -> D:/Proga/Juiwen/harness/host_data/ (next to the repo; ignored by git)
#   HOST=GPU-host DEST=/some/dir bash tools/pull_host_data.sh
# Pulled: results of every experiment (runs, cells, journals, manifests), the queue (specs + finished lab blocks with verdicts), the autolab
# folder (drafts, meta, reports, reviews, usage.jsonl, autopilot state), the experiment specs, the frozen sets and the task set, the dispatcher
# state, the Telegram control journal and a git bundle with all autolab branches. NOT pulled: keys, tokens, pending confirmation codes, the swarm .env.
# Each pull replaces the previous copy of the same folder; a small MANIFEST.txt tells when and how big.
set -euo pipefail
HOST="${HOST:-GPU-host}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$HERE/host_data}"
mkdir -p "$DEST/lab" "$DEST/autolab_cfg"

echo "[pull] lab data from $HOST -> $DEST"
ssh "$HOST" 'cd ~/harness-lab && tar czf - \
    experiments/results experiments/queue experiments/autolab experiments/sets experiments/taskset_math104_mix experiments/adversarial \
    $(ls experiments/*.yaml experiments/*.out 2>/dev/null) dispatcher_state.json autolab.yaml dispatcher.yaml 2>/dev/null' \
  | tar xzf - -C "$DEST/lab" 2>/dev/null || true

echo "[pull] Telegram control journal"
ssh "$HOST" 'tar czf - -C ~/.config/autolab tg_journal.jsonl 2>/dev/null' | tar xzf - -C "$DEST/autolab_cfg" 2>/dev/null || true

echo "[pull] autolab branches (git bundle)"
ssh "$HOST" 'cd ~/harness-lab/autolab_repo && git bundle create - --all 2>/dev/null' > "$DEST/autolab_repo.bundle" || true

{
  echo "pulled: $(date '+%Y-%m-%d %H:%M:%S') from $HOST"
  echo "results folders: $(ls "$DEST/lab/experiments/results" 2>/dev/null | wc -l)"
  echo "finished experiments: $(ls "$DEST/lab/experiments/queue/done" 2>/dev/null | wc -l)"
  echo "size: $(du -sh "$DEST" | cut -f1)"
} > "$DEST/MANIFEST.txt"
cat "$DEST/MANIFEST.txt"
