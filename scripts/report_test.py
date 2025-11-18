import pandas as pd
import os
from sklearn.metrics import classification_report

# sklearn
sk = pd.read_csv("results/sklearn_preds_test_3level.csv")
print("\n=== Sklearn TEST (3-level) ===")
print(classification_report(sk["y_true"], sk["y_pred"], digits=3))

# hf (if available)
hf_path = "results/hf_preds_test_3level.csv"
if os.path.exists(hf_path):
    hf = pd.read_csv(hf_path).copy()
    hf["y_true"] = hf["y_true"] + 1  # Convert 0-2 to 1-3
    hf["y_pred"] = hf["y_pred"] + 1  # Convert 0-2 to 1-3
    print("\n=== HF TEST (3-level) ===")
    print(classification_report(hf["y_true"], hf["y_pred"], digits=3))
else:
    print("\n=== HF TEST (3-level) ===")
    print("HF predictions not available yet. Run scripts/export_hf_preds.py first.")

