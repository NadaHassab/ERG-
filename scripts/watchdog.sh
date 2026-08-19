#!/usr/bin/env bash
# Watchdog: keeps a driver script alive until it prints the final
# "[driver] done" line. Every restart is a resume (--resume), so killed
# runs are redone from the last COMPLETE marker.
set -u
DRIVER=${1:-scripts/stage_runs.sh}
LOG=${2:-logs/stage_runs_20260811.log}
while true; do
    if grep -q "\[driver\] done" "$LOG" 2>/dev/null; then
        echo "[watchdog] driver finished all stages" >> "$LOG"
        exit 0
    fi
    echo "[watchdog] $(date +%H:%M:%S) launching $DRIVER" >> "$LOG"
    bash "$DRIVER" >> "$LOG" 2>&1
    echo "[watchdog] $(date +%H:%M:%S) $DRIVER exited with $?, restarting in 30s" >> "$LOG"
    sleep 30
done