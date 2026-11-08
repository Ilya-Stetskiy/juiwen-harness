#!/bin/bash
# Deploy the COMMITTED code to the lab copy on the GPU host and stamp it with the git hash (lab tool).
# usage: tools/deploy.sh [--force]
#
# What it does:
#   * refuses to run with uncommitted changes (the stamp must describe the code that really runs);
#   * refuses to overwrite host files that were edited by hand after the previous deploy (it compares the host tree
#     with the previously deployed commit); --force overrides;
#   * copies `git archive HEAD` into ~/harness-lab/harness, lab.py and experiments/ into ~/harness-lab;
#   * never touches ~/harness-lab/stream_run.py or local_shim.py (the stream session maintains those);
#   * writes ~/harness-lab/harness/VERSION (hash + date) which lab.py records in every experiment manifest.
set -eu
cd "$(dirname "$0")/.."
HOST=GPU-host
DEST='$HOME/harness-lab'
FORCE="${1:-}"

if [ -n "$(git status --porcelain)" ]; then
  echo "uncommitted changes: commit first (the stamp must match the deployed code)"; git status --short; exit 1
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

git archive HEAD | ssh "$HOST" "mkdir -p $DEST/harness && tar -x -C $DEST/harness"
ssh "$HOST" "mkdir -p $DEST/experiments"
scp -q tools/lab.py "$HOST:harness-lab/lab.py"            # scp paths are relative to the home directory
scp -q experiments/*.yaml "$HOST:harness-lab/experiments/"
ssh "$HOST" "echo '$NEW $(date +%Y-%m-%dT%H:%M:%S)' > $DEST/harness/VERSION"
echo "deployed $NEW (previous: ${OLD:-none})"
