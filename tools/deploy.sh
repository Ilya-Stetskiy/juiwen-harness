#!/bin/bash
# Deploy the COMMITTED code to the lab copy on the GPU host and stamp it with the git hash (lab tool).
# usage: tools/deploy.sh [--force]
#
# What it does:
#   * refuses to run with uncommitted changes to tracked files (untracked files are not in `git archive`; the stamp must describe the code that really runs);
#   * refuses to overwrite host files that were edited by hand after the previous deploy (it compares the host tree
#     with the previously deployed commit); --force overrides;
#   * copies `git archive HEAD` into ~/harness-lab/harness (this includes tools/sim.py), lab.py, experiments/*.yaml and
#     experiments/sets/ into ~/harness-lab (queue/ and results/ are never overwritten);
#   * never touches ~/harness-lab/stream_run.py or local_shim.py (the stream session maintains those);
#   * writes ~/harness-lab/harness/VERSION (hash + date) which lab.py records in every experiment manifest.
set -eu
cd "$(dirname "$0")/.."
HOST=GPU-host
DEST='$HOME/harness-lab'
FORCE="${1:-}"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "uncommitted changes: commit first (the stamp must match the deployed code)"; git status --short --untracked-files=no; exit 1
fi
NEW=$(git rev-parse --short HEAD)
OLD=$(ssh "$HOST" "cat $DEST/harness/VERSION 2>/dev/null | cut -d' ' -f1" || true)

if [ -n "$OLD" ] && [ "$FORCE" != "--force" ] && git cat-file -e "$OLD^{commit}" 2>/dev/null; then
  TMP=$(mktemp -d)
  git archive "$OLD" | tar -x -C "$TMP"
  CHANGED=$(ssh "$HOST" "cd $DEST/harness && find . -type f \( -name '*.py' -o -name '*.json' -o -name '*.txt' -o -name '*.md' \) -not -path './logs/*' -not -path '*/__pycache__/*' -not -name VERSION | sort | xargs md5sum" \
            | while read -r sum file; do
                if [ -f "$TMP/$file" ]; then
                  [ "$(md5sum < "$TMP/$file" | cut -d' ' -f1)" = "$sum" ] || echo "$file"
                fi
              done)
  rm -rf "$TMP"
  if [ -n "$CHANGED" ]; then
    echo "host files differ from the previously deployed commit $OLD (edited by hand?):"; echo "$CHANGED"
    echo "look at them, or run with --force to overwrite"; exit 1
  fi
fi

# R-L6: the lab tools copied with scp are checked for hand edits on the host too (the check above covers only harness/).
if [ -n "$OLD" ] && [ "$FORCE" != "--force" ] && git cat-file -e "$OLD^{commit}" 2>/dev/null; then
  for f in lab.py dispatcher.py autolab.py autolab_ctl.py autolab_tg_adapter.py autolab_chief.py; do
    HOSTSUM=$(ssh "$HOST" "md5sum $DEST/$f 2>/dev/null | cut -d' ' -f1" || true)
    OLDSUM=$(git show "$OLD:tools/$f" 2>/dev/null | md5sum | cut -d' ' -f1)
    if [ -n "$HOSTSUM" ] && [ "$HOSTSUM" != "$OLDSUM" ]; then
      echo "host $f differs from the previously deployed commit $OLD (edited by hand?); look at it or run with --force"; exit 1
    fi
  done
fi

git archive HEAD | ssh "$HOST" "mkdir -p $DEST/harness && tar -x -C $DEST/harness"
ssh "$HOST" "mkdir -p $DEST/experiments"
scp -q tools/lab.py "$HOST:harness-lab/lab.py"
scp -q tools/dispatcher.py "$HOST:harness-lab/dispatcher.py"
scp -q tools/autolab.py "$HOST:harness-lab/autolab.py"   # next to the host stream_run.py it imports            # scp paths are relative to the home directory
scp -q tools/autolab_ctl.py tools/autolab_tg_adapter.py tools/autolab_chief.py "$HOST:harness-lab/"   # chat control of the autolab (no Telegram polling here: see autolab_tg_adapter.py)
scp -q experiments/*.yaml "$HOST:harness-lab/experiments/"
ssh "$HOST" "mkdir -p $DEST/experiments/sets $DEST/experiments/queue/pending $DEST/experiments/queue/running $DEST/experiments/queue/done"
scp -q experiments/sets/*.yaml "$HOST:harness-lab/experiments/sets/"   # frozen sets (never touches queue/ or results/)
ssh "$HOST" "mkdir -p $DEST/experiments/sets/narrow"
scp -q experiments/sets/narrow/*.json experiments/sets/narrow/*.txt "$HOST:harness-lab/experiments/sets/narrow/"   # R-K4: noise_floor.json (thresholds of the narrow sets) and the exclusion lists
ssh "$HOST" "mkdir -p $DEST/experiments/adversarial"
scp -q experiments/adversarial/*.jsonl "$HOST:harness-lab/experiments/adversarial/"   # hand-written adversarial tasks (S12)
ssh "$HOST" "echo '$NEW $(date +%Y-%m-%dT%H:%M:%S)' > $DEST/harness/VERSION"
echo "deployed $NEW (previous: ${OLD:-none})"
