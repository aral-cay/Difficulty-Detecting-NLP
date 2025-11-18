#!/usr/bin/env bash
# Full pipeline to rebuild dataset and retrain with full data

set -euo pipefail

echo "=== Step 1: Check download status ==="
find data/downloads -type f | wc -l

echo ""
echo "=== Step 2: Extract text from all files ==="
python3 scripts/extract_text.py

echo ""
echo "=== Step 3: Recompute levels (3-level) ==="
python3 scripts/recompute_levels_on_observed.py

echo ""
echo "=== Step 4: Build dataset ==="
python3 scripts/make_dataset.py

echo ""
echo "=== Step 5: Chunk texts ==="
python3 scripts/chunk_texts.py

echo ""
echo "=== Step 6: Split dataset ==="
python3 scripts/split_dataset.py

echo ""
echo "=== Step 7: Train sklearn model ==="
python3 scripts/train_sklearn.py

echo ""
echo "=== Step 8: Train HF model with class balancing ==="
python3 scripts/train_hf.py --epochs 3 --batch_size 16 --output models/distilbert_depth_3level_full

echo ""
echo "=== Done! ==="
echo "Models saved to:"
echo "  - models/tfidf_logreg/"
echo "  - models/distilbert_depth_3level_full/best/"

