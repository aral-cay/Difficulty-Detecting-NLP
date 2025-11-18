#!/usr/bin/env bash
# Quick download progress checker

echo "=== Download Status ==="
FILE_COUNT=$(find data/downloads -type f | wc -l | tr -d ' ')
echo "Files downloaded: $FILE_COUNT"

if [ -f download_full.log ]; then
    echo ""
    echo "Latest progress from log:"
    tail -1 download_full.log | grep -o '[0-9]*%' || echo "Processing..."
fi

echo ""
echo "Process running:"
ps aux | grep "fetch_lecturebank.py" | grep -v grep | head -1 | awk '{print "  PID: "$2" | Running for: "$10}'

echo ""
echo "To watch live: tail -f download_full.log"

