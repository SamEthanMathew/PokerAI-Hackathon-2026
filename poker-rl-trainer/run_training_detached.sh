#!/bin/bash
# Run the full RL pipeline in a way that survives laptop disconnect.
# On the Jetson, run:  ./poker-rl-trainer/run_training_detached.sh
# Or:  nohup ./poker-rl-trainer/run_training_detached.sh &
# Then you can close your laptop; training continues on the Jetson.

set -e
cd "$(dirname "$0")/.."
LOG_DIR="poker-rl-trainer/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_detached.log"
echo "Starting training at $(date). Log: $LOG"
exec nohup .venv/bin/python poker-rl-trainer/train.py --phase all >> "$LOG" 2>&1 &
echo "PID: $! — Training is running in background. Safe to disconnect."
echo "To watch: tail -f $LOG"
