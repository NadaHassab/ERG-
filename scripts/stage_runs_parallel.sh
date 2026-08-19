#!/usr/bin/env bash
# Parallel stage driver: runs Phase 8a/8b/9 configs four at a time.
# Every invocation uses --resume; each config logs to its own file.
# Safe to kill and restart at any time.
set -u

PY=.venv/bin/python
CLI="$PY -m pathway_erg.cli"
DATA=configs/data/local.yaml
DIR=logs
fail=0

log() { echo "[driver] $(date +%H:%M:%S) $*"; }

run_cfg() {
    local label="$1" cfg="$2"
    local logf="$DIR/${label}.log"
    log "start $label"
    if $CLI run-separate-neural --data $DATA --experiment "$cfg" --resume > "$logf" 2>&1; then
        log "OK   $label"
    else
        log "FAIL $label"
        fail=1
    fi
}

run_probe() {
    local label="$1" task="$2" fold="$3"
    local logf="$DIR/${label}.log"
    local ckpt="artifacts/results/separate_raw_ot_hierarchical_v1/runs/separate_raw_ot_hierarchical_v1-${task}-fold${fold}-seed1001/final.pt"
    log "start $label"
    if $CLI run-probes --data $DATA --checkpoint "$ckpt" --fold "$fold" --n-reps 100 --seed 7 > "$logf" 2>&1; then
        log "OK   $label"
    else
        log "FAIL $label"
        fail=1
    fi
}

# Batch 1: first four graph controls (correct continues from resume)
run_cfg graph_correct configs/experiments/e8_pathway_graph_correct_v1.yaml &
run_cfg graph_none    configs/experiments/e8_pathway_graph_none_v1.yaml &
run_cfg graph_full    configs/experiments/e8_pathway_graph_full_v1.yaml &
run_cfg graph_wrong   configs/experiments/e8_pathway_graph_wrong_v1.yaml &
wait

# Batch 2: random control + three label-efficiency configs
run_cfg graph_random  configs/experiments/e8_pathway_graph_random_v1.yaml &
run_cfg label_0.1     configs/experiments/e8_label_frac_0.1_v1.yaml &
run_cfg label_0.25    configs/experiments/e8_label_frac_0.25_v1.yaml &
run_cfg label_0.5     configs/experiments/e8_label_frac_0.5_v1.yaml &
wait

# Batch 3-4: probe batteries (LEOP + PERG x folds 0..4)
for task in leop perg; do
    run_probe probe_${task}_fold0 "$task" 0 &
    run_probe probe_${task}_fold1 "$task" 1 &
    run_probe probe_${task}_fold2 "$task" 2 &
    run_probe probe_${task}_fold3 "$task" 3 &
    wait
    run_probe probe_${task}_fold4 "$task" 4 &
    wait
done

echo "[driver] done, aggregate exit status: $fail"
exit $fail