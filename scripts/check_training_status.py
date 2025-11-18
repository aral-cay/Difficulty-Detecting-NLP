#!/usr/bin/env python3
"""Check if transformer training has completed."""
import os
from pathlib import Path

model_dir = Path("models/distilbert_depth")
best_dir = model_dir / "best"

print("=== Training Status Check ===\n")

# Check if model directory exists
if not model_dir.exists():
    print("❌ Model directory doesn't exist yet.")
    print("   Training may not have started or failed early.")
    exit(1)

# Check if best model exists
if best_dir.exists():
    files = list(best_dir.glob("*"))
    if files:
        print("✅ Training COMPLETED!")
        print(f"\nModel saved at: {best_dir}")
        print(f"Files found: {len(files)}")
        print("\nModel files:")
        for f in sorted(files)[:10]:
            size = f.stat().st_size / (1024*1024)  # MB
            print(f"  - {f.name} ({size:.2f} MB)")
        if len(files) > 10:
            print(f"  ... and {len(files)-10} more files")
    else:
        print("⏳ Training directory exists but empty - still training...")
else:
    # Check for checkpoint directories
    checkpoints = list(model_dir.glob("checkpoint-*"))
    if checkpoints:
        print("⏳ Training IN PROGRESS")
        print(f"   Found {len(checkpoints)} checkpoint(s):")
        for cp in sorted(checkpoints)[-3:]:  # Show last 3
            print(f"   - {cp.name}")
    else:
        print("⏳ Training may be starting or failed")
        print(f"   Model directory exists: {model_dir}")

# Check for training logs
log_files = list(model_dir.glob("*.log")) + list(model_dir.glob("training_state.json"))
if log_files:
    print("\n📄 Log files found:")
    for log in log_files:
        print(f"   - {log.name}")

print("\n=== Quick Commands ===")
print("To check training progress:")
print("  python3 scripts/check_training_status.py")
print("\nTo test the model once training completes:")
print("  python3 scripts/predict.py 'Your text here'")
print("\nTo see training output (if running in terminal):")
print("  Check your terminal window")

