#!/usr/bin/env python3
"""Quick script to evaluate DistilBERT on test set."""
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load test set
test_path = Path("data/processed/lecture_depth3_test.csv")
if not test_path.exists():
    print(f"ERROR: Test file not found at {test_path}")
    exit(1)

print(f"Loading test set from {test_path}...")
test_df = pd.read_csv(test_path)
print(f"  Loaded {len(test_df)} samples")

# Load model
model_path = Path("models/distilbert_depth3_fast/best")
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    exit(1)

print(f"\nLoading DistilBERT model from {model_path}...")
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"  Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
print("✓ Model loaded successfully")

# Make predictions
print("\nMaking predictions...")
predictions = []
batch_size = 32
y_true = test_df['depth_level'].astype(int).values

with torch.no_grad():
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df.iloc[i:i+batch_size]['text'].fillna('').astype(str).tolist()
        
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Convert from 0-2 to 1-3
        batch_preds = batch_preds + 1
        predictions.extend(batch_preds.tolist())
        
        if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(test_df):
            print(f"  Processed {min(i + batch_size, len(test_df))}/{len(test_df)} samples...")

predictions = np.array(predictions)

# Calculate accuracy
accuracy = (predictions == y_true).mean()
print(f"\n{'='*60}")
print(f"DistilBERT Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

