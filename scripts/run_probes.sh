#!/usr/bin/env bash
# Phase 9 opener: expert-fidelity probe batteries on the authoritative e6
# checkpoints (LEOP + PERG x folds 0..4).
set -u

PY=.venv/bin/python
CLI="$PY -m pathway_erg.cli"
DATA=configs/data/local.yaml
fail=0

echo "[driver] ===== Phase 9 opener: probes ====="
for task in leop perg; do
    for f in 0 1 2 3 4; do
        ckpt="artifacts/results/separate_raw_ot_hierarchical_v1/runs/separate_raw_ot_hierarchical_v1-${task}-fold${f}-seed1001/final.pt"
        echo "[driver] ===== probes ${task} fold ${f} ====="
        $CLI run-probes --data $DATA --checkpoint "$ckpt" --fold "$f" --n-reps 100 --seed 7 \
            || { echo "[driver] FAILED probes ${task} fold ${f}"; fail=1; }
    done
done
echo "[driver] done, aggregate exit status: $fail"
exit $fail