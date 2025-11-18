#!/usr/bin/env python3
"""Show download progress in real-time."""
import time
from pathlib import Path
import pandas as pd

def count_files():
    downloads_dir = Path("data/downloads")
    files = [f for f in downloads_dir.rglob("*") if f.is_file()]
    return len(files)

def main():
    # Get expected total
    alldata = pd.read_csv("data/raw/LectureBank/alldata_clean.tsv", sep="\t")
    total_expected = len(alldata)
    
    print("=" * 80)
    print("DOWNLOAD PROGRESS MONITOR")
    print("=" * 80)
    print(f"Expected total files: {total_expected}")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    try:
        prev_count = count_files()
        start_time = time.time()
        
        while True:
            current_count = count_files()
            elapsed = time.time() - start_time
            
            if current_count != prev_count:
                progress_pct = (current_count / total_expected) * 100
                remaining = total_expected - current_count
                
                # Estimate time remaining
                if current_count > prev_count and elapsed > 0:
                    rate = (current_count - prev_count) / elapsed if elapsed > 0 else 0
                    if rate > 0:
                        eta_seconds = remaining / rate
                        eta_minutes = eta_seconds / 60
                        eta_str = f"{eta_minutes:.1f} minutes" if eta_minutes < 60 else f"{eta_minutes/60:.1f} hours"
                    else:
                        eta_str = "calculating..."
                else:
                    eta_str = "calculating..."
                
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {current_count}/{total_expected} ({progress_pct:.1f}%) | Remaining: {remaining} | ETA: {eta_str}")
                
                prev_count = current_count
                start_time = time.time()
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        final_count = count_files()
        progress_pct = (final_count / total_expected) * 100
        print(f"\n\nStopped monitoring.")
        print(f"Final count: {final_count}/{total_expected} ({progress_pct:.1f}%)")

if __name__ == "__main__":
    main()

