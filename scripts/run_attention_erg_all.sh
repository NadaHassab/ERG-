#!/usr/bin/env bash
set -euo pipefail

# Direction 3: Attention ERG classifier — all 30 runs (5 folds × 3 seeds × 2 tasks)
# Runs sequentially so each can use full GPU memory.
# Uses nohup + log file for persistence.

SCRIPT="scripts/run_attention_erg.py"
LOG="logs/run_attention_erg_all.log"
mkdir -p logs artifacts/results/attention_erg_v1

echo "=== Attention ERG full run started: $(date) ===" | tee "$LOG"

# We'll run individual fold/seed combos via inline Python to ensure each completes
.venv/bin/python -u "$SCRIPT" >> "$LOG" 2>&1

echo "=== Attention ERG full run finished: $(date) ===" | tee -a "$LOG"
echo "Results: artifacts/results/attention_erg_v1/metrics.json" | tee -a "$LOG"
