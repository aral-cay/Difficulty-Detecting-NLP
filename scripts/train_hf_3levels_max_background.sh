#!/bin/bash
# Script to run DistilBERT training in background that survives terminal closure
# Usage: ./scripts/train_hf_3levels_max_background.sh

cd "$(dirname "$0")/.."

# Create logs directory
mkdir -p logs

# Run training with nohup (survives terminal closure)
nohup conda run -n base python scripts/train_hf_3levels_max.py > logs/train_hf_max_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Training started in background!"
echo "Process ID: $!"
echo "Log file: logs/train_hf_max_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "To check progress:"
echo "  tail -f logs/train_hf_max_*.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep train_hf_3levels_max"
echo ""
echo "Note: Training will stop if you shut down your computer."
echo "For training that survives shutdown, use screen or tmux:"
echo "  screen -S training"
echo "  conda run -n base python scripts/train_hf_3levels_max.py"
echo "  (Press Ctrl+A then D to detach)"

