#!/bin/bash
# Local noise-floor measurement of the narrow sets (experiments/noise_local_nrw_*.yaml) on the host's local Qwen. No gateway tokens.
#   run_local_noise.sh [name ...]      names = set names without prefix (e.g. nrw_search_factoid); default: all, cheapest first
# Stops the local data stream first (STREAM_STOP_local; the autolab cron does not restart it while that file exists) and removes the
# file at the end, so the stream comes back by itself if `локально вкл` is on. Stop early: touch ~/harness-lab/narrow_noise/STOP
# Runs from ~/harness-lab/narrow_noise/ (lab.py copy with the select / exclude_ids_file keys, stream_run.py copy); the deployed lab.py is untouched,
# results go to the usual ~/harness-lab/experiments/results/noise_local_<set>/.
set -u
H=$HOME/harness-lab
D=$H/narrow_noise
export LAB_WORK=/tmp/hl_lab_noise
ORDER="nrw_math_word_simple nrw_search_comparison nrw_search_bridge nrw_search_factoid nrw_search_factoid_abstain nrw_search_multihop_abstain nrw_search_multihop_deep nrw_math_gsm_chain nrw_data_analysis_wtq_big_table nrw_data_analysis_wtq_list nrw_math_algebra nrw_data_analysis_wtq_aggregate nrw_data_analysis_wtq_lookup nrw_math_discrete nrw_data_analysis_tabfact nrw_math_geometry nrw_math_hard_symbolic nrw_data_analysis_dabench nrw_coding_cg_large_file nrw_coding_cg_multi_bug nrw_coding_cg_infeasible nrw_coding_cg_single_bug nrw_coding_cg_toy_lib nrw_coding_mbpp"
NAMES=${*:-$ORDER}
log() { echo "[$(date '+%F %T')] $*" >> $D/driver.log; }
rm -f $D/STOP
touch $H/STREAM_STOP_local
log "start: stream stop requested ($NAMES)"
for i in $(seq 1 90); do pgrep -af "stream_run.py --backend local" | grep -v pgrep > /dev/null || break; sleep 20; done
if pgrep -af "stream_run.py --backend local" | grep -v pgrep > /dev/null; then log "local stream still running after 30 min: abort"; rm -f $H/STREAM_STOP_local; exit 1; fi
for n in $NAMES; do
  [ -e $D/STOP ] && { log "STOP file: finishing"; break; }
  log "run noise_local_$n"
  (cd $D && python3 lab.py run $H/experiments/noise_local_$n.yaml >> $D/lab_$n.out 2>&1)
  log "done noise_local_$n exit $?"
done
rm -f $H/STREAM_STOP_local
log "end: STREAM_STOP_local removed (the stream restarts with the autolab cron if local data is on)"
