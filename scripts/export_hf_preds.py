import pandas as pd
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Load test data
test_df = pd.read_csv("data/processed/lecture_depth3_test.csv")

# sklearn predictions
print("Generating sklearn predictions...")
vectorizer = joblib.load("models/tfidf_logreg_3level/vectorizer.joblib")
classifier = joblib.load("models/tfidf_logreg_3level/classifier.joblib")
sk_vec = vectorizer.transform(test_df["text"].tolist())
sk_pred = classifier.predict(sk_vec)

sk_results = pd.DataFrame({
    "y_true": test_df["depth_level"].values,
    "y_pred": sk_pred
})
os.makedirs("results", exist_ok=True)
sk_results.to_csv("results/sklearn_preds_test_3level.csv", index=False)
print(f"Saved sklearn predictions to results/sklearn_preds_test_3level.csv")

# HF predictions
if os.path.isdir("models/distilbert_depth3/best"):
    print("Generating HF predictions...")
    tok = AutoTokenizer.from_pretrained("models/distilbert_depth3/best")
    mdl = AutoModelForSequenceClassification.from_pretrained("models/distilbert_depth3/best")
    mdl.eval()
    
    preds = []
    batch_size = 8
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i+batch_size]
        enc = tok(batch["text"].tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())  # 0-2
    
    hf_results = pd.DataFrame({
        "y_true": (test_df["depth_level"].values - 1).astype(int),  # 1-3 -> 0-2
        "y_pred": preds
    })
    hf_results.to_csv("results/hf_preds_test_3level.csv", index=False)
    print(f"Saved HF predictions to results/hf_preds_test_3level.csv")
else:
    print("HF model not available yet")

