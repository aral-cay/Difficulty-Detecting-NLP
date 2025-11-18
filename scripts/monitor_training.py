#!/usr/bin/env python3
"""Monitor training progress in real-time."""
import os
import time
from pathlib import Path

def check_training():
    model_dir = Path("models/distilbert_depth")
    best_dir = model_dir / "best"
    
    # Check if training completed
    if best_dir.exists() and list(best_dir.glob("*")):
        return "COMPLETED", f"Model saved at {best_dir}"
    
    # Check for checkpoints
    if model_dir.exists():
        checkpoints = sorted(model_dir.glob("checkpoint-*"))
        if checkpoints:
            latest = checkpoints[-1]
            return "TRAINING", f"Latest checkpoint: {latest.name}"
    
    # Check log file
    log_file = Path("training_output.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                # Look for progress indicators
                if "epoch" in last_line.lower() or "%" in last_line:
                    return "TRAINING", f"Progress: {last_line[:100]}"
                elif "error" in last_line.lower() or "traceback" in last_line.lower():
                    return "ERROR", f"Error in log: {last_line[:100]}"
    
    return "NOT_STARTED", "No training activity detected"

def monitor():
    print("=== Training Progress Monitor ===\n")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            status, info = check_training()
            timestamp = time.strftime("%H:%M:%S")
            
            if status == "COMPLETED":
                print(f"[{timestamp}] ✅ {status}: {info}")
                break
            elif status == "TRAINING":
                print(f"[{timestamp}] ⏳ {status}: {info}")
            elif status == "ERROR":
                print(f"[{timestamp}] ❌ {status}: {info}")
                break
            else:
                print(f"[{timestamp}] ⏸️  {status}: {info}")
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor()

