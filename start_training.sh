#!/bin/bash
# Start model training with real-time progress
# Usage: ./start_training.sh [sklearn|distilbert|both]

cd "$(dirname "$0")"

MODEL=${1:-both}

echo "=========================================="
echo "Starting Model Training"
echo "=========================================="
echo ""

if [ "$MODEL" = "sklearn" ] || [ "$MODEL" = "both" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Training Maximum Sklearn Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    conda run -n base python scripts/train_sklearn_max.py --use_smote
    echo ""
fi

if [ "$MODEL" = "distilbert" ] || [ "$MODEL" = "both" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Training Maximum DistilBERT Model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Note: This will take 2-3 hours. Progress will be shown in real-time."
    echo ""
    conda run -n base python scripts/train_hf_3levels_max.py
    echo ""
fi

echo "=========================================="
echo "Training Complete!"
echo "=========================================="

