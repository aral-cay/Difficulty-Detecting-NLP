#!/usr/bin/env bash
# Run download in a way that persists even if terminal closes

cd /Users/aralcay/Desktop/CS74Final

# Kill existing download if running
pkill -f "fetch_lecturebank.py" 2>/dev/null || true

# Start download with nohup (survives terminal close)
nohup /Users/aralcay/code/lecture-depth/venv/bin/python3 scripts/fetch_lecturebank.py \
    --alldata data/raw/LectureBank/alldata_clean.tsv \
    > download_full.log 2>&1 &

echo "Download started with PID: $!"
echo "Log file: download_full.log"
echo "Monitor with: tail -f download_full.log"
