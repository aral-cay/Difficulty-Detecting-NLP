#!/usr/bin/env bash

set -euo pipefail

echo "[1/5] Extracting text..."
python3 scripts/extract_text.py

echo "[2/5] Computing depth levels..."
python3 scripts/compute_depth.py

echo "[3/5] Joining dataset..."
python3 scripts/make_dataset.py

echo "[4/5] Chunking + splitting..."
python3 scripts/chunk_texts.py
python3 scripts/split_dataset.py

echo "[5/5] Training (sklearn + HF)..."
python3 scripts/train_sklearn.py
python3 scripts/train_hf.py

echo "Done. Model at models/distilbert_depth/best/"

