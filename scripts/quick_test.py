import pandas as pd, numpy as np, joblib, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

np.random.seed(0)

# load 10 random test rows
df = pd.read_csv("data/processed/lecture_depth_test.csv").sample(10, random_state=0).reset_index(drop=True)
print("Sampled 10 rows from TEST\n")

# sklearn
vectorizer = joblib.load("models/tfidf_logreg/vectorizer.joblib")
classifier = joblib.load("models/tfidf_logreg/classifier.joblib")
sk_vec = vectorizer.transform(df["text"].tolist())
sk_pred = classifier.predict(sk_vec)

# hf (if available)
hf_pred = None
if os.path.isdir("models/distilbert_depth/best"):
    print("Loading HF model...")
    tok = AutoTokenizer.from_pretrained("models/distilbert_depth/best")
    mdl = AutoModelForSequenceClassification.from_pretrained("models/distilbert_depth/best")
    mdl.eval()
    preds = []
    for i in range(0, len(df), 8):
        batch = df.iloc[i:i+8]
        enc = tok(batch["text"].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
        preds.extend((logits.argmax(-1).cpu().numpy()+1).tolist())  # back to 1..5
    hf_pred = np.array(preds)
    print("HF model loaded ✓\n")
else:
    print("HF model not available yet (still training?)\n")

# print side-by-side
print("=" * 80)
print(f"{'True':<6} {'SK_Pred':<8}" + (f" {'HF_Pred':<8}" if hf_pred is not None else ""))
print("=" * 80)
for i in range(len(df)):
    row = f"{df.iloc[i]['depth_level']:<6} {sk_pred[i]:<8}"
    if hf_pred is not None:
        row += f" {hf_pred[i]:<8}"
    print(row)
print("=" * 80)

# Accuracy
sk_acc = (sk_pred == df["depth_level"]).mean()
print(f"\nSklearn accuracy: {sk_acc:.3f}")
if hf_pred is not None:
    hf_acc = (hf_pred == df["depth_level"]).mean()
    print(f"HF accuracy: {hf_acc:.3f}")

