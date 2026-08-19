#!/usr/bin/env bash
# Rerun the missing probe batteries (leop fold 0 was lost to the old
# flat report naming; perg folds 1-2 were killed mid-run).
set -u
PY=.venv/bin/python
CLI="$PY -m pathway_erg.cli"
DATA=configs/data/local.yaml
fail=0
for spec in "leop 0" "perg 1" "perg 2"; do
    task=${spec% *}
    f=${spec#* }
    ckpt="artifacts/results/separate_raw_ot_hierarchical_v1/runs/separate_raw_ot_hierarchical_v1-${task}-fold${f}-seed1001/final.pt"
    echo "[driver] ===== probes ${task} fold ${f} ====="
    $CLI run-probes --data $DATA --checkpoint "$ckpt" --fold "$f" --n-reps 100 --seed 7 \
        || { echo "[driver] FAILED probes ${task} fold ${f}"; fail=1; }
done
echo "[driver] done, aggregate exit status: $fail"
exit $fail