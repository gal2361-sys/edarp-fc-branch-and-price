#!/bin/zsh
# Sequential experiment queue: one batch at a time, never two Gurobi runs at once.
#   nohup ./run_queue.sh > /dev/null 2>&1 &
# Progress: results/queue.log. After each batch its summary CSV is committed (-f).
# A batch is skipped only if its summary CSV is complete (all rows exit 0, 10 per
# size) and its recorded arguments (summary_<name>.args) are identical.
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=${QUEUE_LOG:-results/queue.log}
mkdir -p results
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> $LOG }

complete() {   # complete <summary stem> <expected rows> <expected args line>
    local csv=results/summary_$1.csv argsf=results/summary_$1.args
    [[ -f $csv && -f $argsf ]] || return 1
    [[ "$(cat $argsf)" == "$3" ]] || return 1
    $PY - "$csv" "$2" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ok = len(rows) == int(sys.argv[2]) and all(r.get("exit_code") == "0" for r in rows)
sys.exit(0 if ok else 1)
PYEOF
}

step() {       # step <summary stem> <result tag> <expected rows> <mode> <sizes...> -- <flags...>
    local stem=$1 tag=$2 rows=$3; shift 3
    local mode=$1; shift
    local sizes=() flags=()
    while [[ $# -gt 0 && $1 != "--" ]]; do sizes+=$1; shift; done
    [[ $1 == "--" ]] && shift
    flags=("$@")
    local argsline="$mode ${sizes[*]} -- ${flags[*]}"
    argsline=${argsline% }
    if complete $stem $rows "$argsline"; then
        log "SKIP $stem: complete summary with identical arguments"
        return
    fi
    if [[ -n $DRY_RUN ]]; then
        log "DRY would run $stem: run_batch.py $mode ${sizes[*]} --tag $tag --summary $stem -- ${flags[*]} (args line: '$argsline')"
        return
    fi
    log "START $stem: run_batch.py $mode ${sizes[*]} --tag $tag --summary $stem -- ${flags[*]}"
    $PY run_batch.py $mode ${sizes[@]} --tag $tag --summary $stem -- ${flags[@]} >> $LOG 2>&1
    local code=$?
    log "END $stem: exit $code"
    if [[ -f results/summary_$stem.csv ]]; then
        git add -f results/summary_$stem.csv
        git commit -q -m "results: $stem summary" >> $LOG 2>&1 \
            && log "COMMIT summary_$stem.csv" || log "COMMIT FAILED for summary_$stem.csv"
    fi
}

log "QUEUE START (pid $$)"
# v2 (2026-09-29): initial full-fleet heuristic on, grid rounding off (default)
step tree_v2_b75      tree_v2      10 tree b75 -- --node-time-limit 1800
step tree_v2_p120_b75 tree_v2_p120 10 tree b75 -- --node-time-limit 1800 --pricing-time-limit 120
step tree_v2_b50      tree_v2      10 tree b50 -- --node-time-limit 1800
log "QUEUE DONE"
