# Running Training in Background

## ⚠️ Important: Training Stops When Computer Closes

**No, training will NOT continue if you close your computer.** All processes are terminated when the computer shuts down or goes to sleep.

---

## Options to Keep Training Running

### Option 1: Use `screen` (Recommended) ✅

**Screen allows you to detach and reattach sessions:**

```bash
# Start a screen session
screen -S training

# Inside screen, run training
conda activate base
python scripts/train_hf_3levels_max.py

# Detach from screen (training continues): Press Ctrl+A, then D

# Later, reattach to see progress:
screen -r training

# List all screen sessions:
screen -ls
```

**Benefits:**
- ✅ Training continues even if you close terminal
- ✅ Can reattach anytime to see progress
- ✅ Training stops if computer shuts down (but survives terminal closure)

---

### Option 2: Use `tmux` (Alternative)

```bash
# Start tmux session
tmux new -s training

# Inside tmux, run training
conda activate base
python scripts/train_hf_3levels_max.py

# Detach: Press Ctrl+B, then D

# Reattach later:
tmux attach -t training
```

---

### Option 3: Use `nohup` (Simple Background)

```bash
# Run with nohup (survives terminal closure, but not computer shutdown)
nohup conda run -n base python scripts/train_hf_3levels_max.py > logs/training.log 2>&1 &

# Check progress
tail -f logs/training.log

# Check if still running
ps aux | grep train_hf_3levels_max
```

**Or use the provided script:**
```bash
./scripts/train_hf_3levels_max_background.sh
```

---

### Option 4: Keep Computer On (Simplest)

**If you can keep your computer on:**
- ✅ Plug in power adapter
- ✅ Prevent sleep: System Settings → Energy Saver → Prevent sleep
- ✅ Close laptop lid (if external monitor)
- ✅ Training will continue normally

---

## Current Training Status

**To check if training is still running:**

```bash
# Check process
ps aux | grep train_hf_3levels_max | grep -v grep

# Check latest log
tail -f /tmp/train_hf_max.log
```

**To restart training if it stopped:**

```bash
# Using screen (recommended)
screen -S training
conda activate base
python scripts/train_hf_3levels_max.py
# Press Ctrl+A, then D to detach
```

---

## Training Time Estimates

- **Sklearn Max**: ~5-10 minutes ✅ (Already done: 81.50%)
- **DistilBERT Max**: ~2-3 hours ⏳

---

## Recommendations

1. **If training now**: Use `screen` to detach and let it run
2. **If computer will be off**: Training will stop - restart when computer is back on
3. **Best practice**: Use `screen` or `tmux` for long-running training

---

## Quick Start with Screen

```bash
# Install screen if needed (usually pre-installed on Mac)
# brew install screen  # if not installed

# Start training in screen
screen -S training
conda activate base
cd /Users/aralcay/Desktop/CS74Final
python scripts/train_hf_3levels_max.py

# Detach: Ctrl+A, then D
# Reattach later: screen -r training
```

