#!/bin/sh
# Retrain profiles/cost_model.json on a task set with recorded runs (e.g. the organisers' tasks).
#   retrain_cost_model.sh TASKS_DIR LOG_DIR [OUT]
# needs: the harness venv (HARNESS_PY, default ~/harness-lab/.venv/bin/python) and a scratch venv with scikit-learn
# (ML_PY, default /tmp/mlenv/bin/python; create: python3 -m venv /tmp/mlenv && /tmp/mlenv/bin/pip install scikit-learn numpy).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
HARNESS_PY=${HARNESS_PY:-$HOME/harness-lab/.venv/bin/python}
ML_PY=${ML_PY:-/tmp/mlenv/bin/python}
OUT=${3:-$HERE/../profiles/cost_model.json}
ROWS=${ROWS:-/tmp/cost_rows.jsonl}
$HARNESS_PY "$HERE/cost_features.py" --tasks "$1" --journals "$2" --out "$ROWS"
$ML_PY "$HERE/predictor_export.py" --rows "$ROWS" --out "$OUT"
echo "retrained $OUT (restart is not needed: loaded at start)"
