#!/usr/bin/env python3
"""Monitor download progress and show status."""
import os
from pathlib import Path

downloads_dir = Path("data/downloads")
current_count = len(list(downloads_dir.rglob("*"))) - len(list(downloads_dir.rglob("*/")))  # files minus directories

print(f"Current files downloaded: {current_count}")
print(f"\nTo check download progress:")
print(f"  tail -f download_full.log")
print(f"\nTo check when download completes:")
print(f"  ps aux | grep fetch_lecturebank.py")

if Path("download_full.log").exists():
    print(f"\nLast log entries:")
    with open("download_full.log", "r") as f:
        lines = f.readlines()
        for line in lines[-5:]:
            print(f"  {line.strip()}")

