#!/usr/bin/env python3
"""
Live monitoring script for DistilBERT training progress.
Shows current step, progress percentage, and estimated time remaining.
"""

import time
import os
import re
from datetime import datetime, timedelta

LOG_FILE = "results/train_hf_3levels.log"

def extract_progress(line):
    """Extract progress information from log line."""
    # Pattern: "  8%|▊         | 243/3200 [07:35<1:42:21,  2.08s/it]"
    pattern = r'(\d+)%\|.*?\| (\d+)/(\d+) \[([\d:]+)<([\d:]+),'
    match = re.search(pattern, line)
    if match:
        percent = int(match.group(1))
        current = int(match.group(2))
        total = int(match.group(3))
        elapsed = match.group(4)
        remaining = match.group(5)
        return {
            'percent': percent,
            'current': current,
            'total': total,
            'elapsed': elapsed,
            'remaining': remaining
        }
    return None

def format_time(time_str):
    """Format time string (HH:MM:SS) to readable format."""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"
    return time_str

def monitor_training():
    """Monitor training progress in real-time."""
    print("=" * 80)
    print("DistilBERT Training Progress Monitor")
    print("=" * 80)
    print(f"Log file: {LOG_FILE}")
    print("Press Ctrl+C to stop monitoring\n")
    
    if not os.path.exists(LOG_FILE):
        print(f"❌ Log file not found: {LOG_FILE}")
        print("   Training may not have started yet.")
        return
    
    last_progress = None
    
    try:
        # Open log file and seek to end
        with open(LOG_FILE, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                
                if line:
                    progress = extract_progress(line)
                    if progress:
                        # Only update if progress changed
                        if last_progress is None or progress['current'] != last_progress['current']:
                            last_progress = progress
                            
                            # Clear line and print progress
                            print(f"\r{' ' * 80}", end='')  # Clear line
                            print(f"\r📊 Progress: {progress['current']}/{progress['total']} steps "
                                  f"({progress['percent']}%) | "
                                  f"Elapsed: {format_time(progress['elapsed'])} | "
                                  f"Remaining: {format_time(progress['remaining'])}", end='', flush=True)
                
                time.sleep(0.5)  # Check every 0.5 seconds
                
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")
        if last_progress:
            print(f"\nLast seen progress: {last_progress['current']}/{last_progress['total']} steps "
                  f"({last_progress['percent']}%)")
    except FileNotFoundError:
        print(f"\n❌ Log file disappeared: {LOG_FILE}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    monitor_training()

