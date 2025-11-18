# Model Accuracy Summary

## Test Set Accuracy Comparison

Test set size: **1,319 samples**

| Model | Accuracy | Correct Predictions | Status |
|-------|----------|---------------------|--------|
| **1. Original Sklearn (TF-IDF + Logistic Regression)** | **79.08%** | 1,043/1,319 | ✅ Trained |
| **2. Improved Sklearn (with feature engineering)** | **81.43%** | 1,074/1,319 | ✅ Trained |
| **3. DistilBERT (Transformer)** | **77.79%** | 1,026/1,319 | ✅ Trained |

---

## Detailed Breakdown

### 1. Original Sklearn Model
- **Accuracy: 79.08%**
- **Macro F1: 0.755**
- **Weighted F1: 0.801**

**Per-Class Performance:**
- Beginner: Precision 0.883, Recall 0.774, F1 0.825
- Intermediate: Precision 0.477, Recall 0.848, F1 0.610
- Advanced: Precision 0.867, Recall 0.795, F1 0.829

---

### 2. Improved Sklearn Model ⭐ BEST
- **Accuracy: 81.43%** (+2.35% improvement)
- **Macro F1: 0.7818**
- **Weighted F1: 0.82**

**Per-Class Performance:**
- Beginner: Precision 0.86, Recall 0.82, F1 0.84
- Intermediate: Precision 0.58, Recall 0.80, F1 0.67
- Advanced: Precision 0.87, Recall 0.81, F1 0.84

**Improvements:**
- ✅ +2.35% overall accuracy
- ✅ +10.3% Intermediate precision (47.7% → 58%)
- ✅ +6% Intermediate F1-score

---

### 3. DistilBERT Model
- **Accuracy: 77.79%**
- **Macro F1: 0.724**
- **Weighted F1: 0.778**

**Per-Class Performance:**
- Beginner: Precision 0.774, Recall 0.845, F1 0.808
- Intermediate: Precision 0.546, Recall 0.563, F1 0.555
- Advanced: Precision 0.867, Recall 0.760, F1 0.810

---

## Ranking

1. 🥇 **Improved Sklearn: 81.43%** (Best overall)
2. 🥈 **Original Sklearn: 79.08%** (Good baseline)
3. 🥉 **DistilBERT: 77.79%** (Transformer model)

---

## Key Insights

- **Improved Sklearn is the best model** with 81.43% accuracy
- **Feature engineering made a significant difference** (+2.35% improvement)
- **Sklearn models outperform DistilBERT** on this dataset (likely due to dataset size and task nature)
- **Intermediate class is the most challenging** for all models, but improved model handles it best

---

## Recommendation

**Use the Improved Sklearn Model** for production:
- Highest accuracy (81.43%)
- Best Intermediate precision (58%)
- Fast inference
- Good interpretability

