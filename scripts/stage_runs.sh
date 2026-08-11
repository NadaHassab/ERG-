#!/usr/bin/env bash
# Stage driver: Phase 7 (joint SSL), Phase 8 (graph controls + label
# efficiency), Phase 9 opener (probe batteries on authoritative e6
# checkpoints). Every supervised invocation uses --resume so a killed
# process can restart without redoing completed runs.
set -u

PY=.venv/bin/python
CLI="$PY -m pathway_erg.cli"
DATA=configs/data/local.yaml
RUN_LOG=${1:-logs/stage_runs.log}

fail=0
stage() {
    echo "[driver] ===== $1 ====="
    shift
    "$@" || { echo "[driver] FAILED: $*"; fail=1; }
}

ssl_done() {
    test -f "artifacts/results/ssl_pretrain_v1/fold$1/COMPLETE"
}

ssl_stage() {
    if ssl_done "$1"; then
        echo "[driver] ===== Phase 7a: SSL pretrain fold $1 (already complete, skipped) ====="
    else
        stage "Phase 7a: SSL pretrain fold $1" \
            $CLI run-ssl-pretrain --data $DATA --exclude-fold "$1"
    fi
}

ssl_stage 0
ssl_stage 1
ssl_stage 2
ssl_stage 3
ssl_stage 4
for f in 0 1 2 3 4; do
    stage "Phase 7b: SSL-init fine-tune fold $f" \
        $CLI run-separate-neural --data $DATA \
        --experiment configs/experiments/e6_sslinit_fold${f}_v1.yaml --resume
done
stage "Phase 7b: SSL-init 5-fold summary ensemble" \
    $CLI run-separate-neural --data $DATA \
    --experiment configs/experiments/e6_sslinit_summary_v1.yaml --resume

for g in correct none full wrong random; do
    stage "Phase 8a: pathway graph control '$g'" \
        $CLI run-separate-neural --data $DATA \
        --experiment configs/experiments/e8_pathway_graph_${g}_v1.yaml --resume
done

for f in 0.1 0.25 0.5; do
    stage "Phase 8b: label efficiency $f" \
        $CLI run-separate-neural --data $DATA \
        --experiment configs/experiments/e8_label_frac_${f}_v1.yaml --resume
done

for task in leop perg; do
    for f in 0 1 2 3 4; do
        ckpt=artifacts/results/separate_raw_ot_hierarchical_v1/runs/separate_raw_ot_hierarchical_v1-${task}-fold${f}-seed1001/final.pt
        stage "Phase 9 opener: probes ${task} fold ${f}" \
            $CLI run-probes --data $DATA --checkpoint $ckpt --fold $f --n-reps 100 --seed 7
    done
done

echo "[driver] done, aggregate exit status: $fail"
exit $fail