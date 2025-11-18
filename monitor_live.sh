#!/usr/bin/env bash
# Simple live monitoring using tail -f

LOG_FILE="results/train_hf_3levels.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "   Training may not have started yet."
    exit 1
fi

echo "=================================================================================="
echo "DistilBERT Training Progress Monitor"
echo "=================================================================================="
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""
echo "Watching for progress updates..."
echo ""

# Use tail -f to follow the log file and filter for progress lines
tail -f "$LOG_FILE" | grep --line-buffered -E "^\s+\d+%\|" | while read line; do
    # Extract and display progress
    echo -ne "\r\033[K$line"  # Clear line and print progress
done

