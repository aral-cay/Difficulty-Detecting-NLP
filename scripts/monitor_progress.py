#!/usr/bin/env python3
"""Real-time progress monitor for download."""
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

def count_files(directory):
    """Count files in directory."""
    try:
        return len(list(Path(directory).rglob("*"))) - len(list(Path(directory).rglob("*/")))
    except:
        return 0

def get_latest_log_line(log_file):
    """Get the latest progress line from log."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Find the last progress bar line
            for line in reversed(lines):
                if '%|' in line and '/' in line:
                    return line.strip()
        return None
    except:
        return None

def main():
    log_file = "/tmp/fetch_lecturebank.log"
    downloads_dir = "data/downloads"
    total_files = 7500
    
    print("=" * 80)
    print("REAL-TIME DOWNLOAD PROGRESS MONITOR")
    print("=" * 80)
    print(f"Total files to download: {total_files}")
    print("Press Ctrl+C to stop monitoring\n")
    
    start_time = time.time()
    last_count = 0
    
    try:
        while True:
            current_count = count_files(downloads_dir)
            elapsed = time.time() - start_time
            
            # Get latest log line
            latest_line = get_latest_log_line(log_file)
            
            # Calculate stats
            if elapsed > 0:
                rate = current_count / elapsed
                remaining = total_files - current_count
                if rate > 0:
                    eta_seconds = remaining / rate
                    eta = timedelta(seconds=int(eta_seconds))
                else:
                    eta = "calculating..."
            else:
                rate = 0
                eta = "calculating..."
            
            # Clear screen and print status
            os.system('clear' if os.name != 'nt' else 'cls')
            print("=" * 80)
            print("REAL-TIME DOWNLOAD PROGRESS MONITOR")
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"Elapsed: {timedelta(seconds=int(elapsed))}")
            print()
            print(f"Files downloaded: {current_count:,} / {total_files:,}")
            print(f"Progress: {(current_count/total_files*100):.2f}%")
            print(f"Rate: {rate:.2f} files/sec")
            print(f"ETA: {eta}")
            print()
            if latest_line:
                print("Latest progress:")
                print(f"  {latest_line}")
            print()
            print("Press Ctrl+C to stop")
            
            time.sleep(2)
            last_count = current_count
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()

