#!/bin/bash
# A/B runs of the pipeline on a pristine task set (lab tool).
# usage: ab_run.sh <pristine_tasks_dir> <label> <repeats> [VAR=value ...]
# Every repeat starts from a fresh copy (answers and edited workspaces are discarded) and writes one JSON line.
set -u
SRC="$1"; LABEL="$2"; N="$3"; shift 3
HOME_DIR="$HOME/harness"
for i in $(seq 1 "$N"); do
  WORK="/tmp/ab/${LABEL}_work"
  LOGS="/tmp/ab/${LABEL}_$i"
  rm -rf "$WORK" "${WORK}_grading.json" "$LOGS"
  mkdir -p /tmp/ab
  cp -r "$SRC" "$WORK"; cp "${SRC}_grading.json" "${WORK}_grading.json"
  ( cd "$HOME_DIR" && env HARNESS_FALLBACK=off HARNESS_FORCE_MODEL=qwen3.6-35b-a3b-noreason HARNESS_WORKERS=5 \
      BUDGET=100000 LLM_BASE_URL=http://127.0.0.1:18443 HARNESS_LOG_DIR="$LOGS" "$@" \
      timeout 3000 "$HOME/harness-env/bin/python" run_agent.py --tasks-dir "$WORK" > "$LOGS.out" 2>&1 )
  RES=$(python3 "$HOME_DIR/tools/grade_set.py" "$WORK" "$LOGS" --json)
  echo "{\"label\": \"$LABEL\", \"repeat\": $i, \"env\": \"$*\", \"result\": $RES}" >> /tmp/ab/results.jsonl
  rm -f "$LOGS.out"   # raw openjiuwen stdout contains prompts and answers; keep metadata only
done
