#!/usr/bin/env bash
# Plan section 11 external arm: four-domain SSL followed by LEOP/PERG
# fine-tuning. URFU supervised training remains blocked pending label sign-off.
set -u

PY=.venv/bin/python
CLI="$PY -m pathway_erg.cli"
DATA=configs/data/local.yaml

fail=0
stage() {
    printf '[external-driver] ===== %s =====\n' "$1"
    shift
    "$@" || { printf '[external-driver] FAILED: %s\n' "$*"; fail=1; }
}

ssl_stage() {
    fold=$1
    complete="artifacts/results/ssl_pretrain_external_v1/fold${fold}/COMPLETE"
    if test -f "$complete"; then
        printf '[external-driver] ===== e9 SSL fold %s (already complete, skipped) =====\n' "$fold"
        return
    fi
    stage "e9 four-domain SSL fold $fold" \
        $CLI run-ssl-pretrain --data "$DATA" \
        --experiment configs/experiments/e9_ssl_pretrain_external_v1.yaml \
        --exclude-fold "$fold"
}

for fold in 0 1 2 3 4; do
    ssl_stage "$fold"
done

for fold in 0 1 2 3 4; do
    stage "e9 external SSL-init fine-tune fold $fold" \
        $CLI run-separate-neural --data "$DATA" \
        --experiment "configs/experiments/e9_sslinit_external_fold${fold}_v1.yaml" \
        --resume
done

stage "e9 external SSL-init 5-fold summary ensemble" \
    $CLI run-separate-neural --data "$DATA" \
    --experiment configs/experiments/e9_sslinit_external_summary_v1.yaml \
    --resume

printf '[external-driver] done, aggregate exit status: %s\n' "$fail"
exit "$fail"
