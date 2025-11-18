# Maximum Accuracy Training Scripts

## Overview

Created advanced training scripts to maximize accuracy for both models separately (no ensemble).

---

## 🎯 Sklearn Maximum Accuracy (`train_sklearn_max.py`)

### Features:
- ✅ **Enhanced complexity features** (20 features instead of 14)
- ✅ **More TF-IDF features** (15,000 instead of 12,000)
- ✅ **Higher C parameter** (3.0 instead of 2.0)
- ✅ **Increased Intermediate weight** (2.5x instead of 2x)
- ✅ **Optional SMOTE** data augmentation (`--use_smote`)
- ✅ **Optional XGBoost** classifier (`--use_xgboost`)

### Usage:

```bash
# Standard maximum training
python scripts/train_sklearn_max.py

# With SMOTE augmentation
python scripts/train_sklearn_max.py --use_smote

# With XGBoost (requires: pip install xgboost)
python scripts/train_sklearn_max.py --use_xgboost

# Both SMOTE and XGBoost
python scripts/train_sklearn_max.py --use_smote --use_xgboost
```

### Expected Results:
- **Current**: 81.43%
- **Expected**: **84-86%** (+2.5-4.5%)
- **With SMOTE**: **85-87%** (+3.5-5.5%)
- **With XGBoost**: **85-87%** (+3.5-5.5%)

---

## 🎯 DistilBERT Maximum Accuracy (`train_hf_3levels_max.py`)

### Features:
- ✅ **Lower learning rate** (2e-5 instead of 5e-5)
- ✅ **Longer context** (512 tokens)
- ✅ **More epochs** (8 with early stopping)
- ✅ **Cosine LR scheduler** (better convergence)
- ✅ **Increased Intermediate weight** (2.5x instead of 2x)
- ✅ **Mixed precision training** (fp16, faster)
- ✅ **F1-macro optimization** (better for imbalanced classes)

### Usage:

```bash
# Standard maximum training
python scripts/train_hf_3levels_max.py

# Custom parameters
python scripts/train_hf_3levels_max.py --learning_rate 1.5e-5 --epochs 10 --max_length 512
```

### Expected Results:
- **Current**: 77.79%
- **Expected**: **80-82%** (+2-4%)

---

## 📊 Comparison

| Model | Current | Expected Max | Improvement |
|-------|---------|--------------|-------------|
| **Sklearn** | 81.43% | **84-87%** | +2.5-5.5% |
| **DistilBERT** | 77.79% | **80-82%** | +2-4% |

---

## 🚀 Quick Start

### Train Maximum Sklearn Model:
```bash
# Install optional dependencies first
pip install imbalanced-learn xgboost

# Train with all optimizations
python scripts/train_sklearn_max.py --use_smote --use_xgboost
```

### Train Maximum DistilBERT Model:
```bash
python scripts/train_hf_3levels_max.py
```

---

## 📝 Key Improvements

### Sklearn:
1. **20 complexity features** (vs 14)
   - Added: advanced density, long word ratio, question type starters
2. **15,000 TF-IDF features** (vs 12,000)
3. **C=3.0** (vs 2.0) - less regularization
4. **2.5x Intermediate weight** (vs 2x)
5. **SMOTE option** - synthetic data generation
6. **XGBoost option** - gradient boosting

### DistilBERT:
1. **2e-5 learning rate** (vs 5e-5) - gentler fine-tuning
2. **512 token context** (vs 256) - less truncation
3. **8 epochs** (vs 5) - more training
4. **Cosine LR scheduler** - better convergence
5. **2.5x Intermediate weight** (vs 2x)
6. **Mixed precision** - faster training

---

## ⏱️ Training Time

- **Sklearn Max**: ~5-10 minutes
- **Sklearn Max + SMOTE**: ~10-15 minutes
- **Sklearn Max + XGBoost**: ~15-20 minutes
- **DistilBERT Max**: ~2-3 hours

---

## 🎯 Next Steps

1. **Train both maximum models**
2. **Compare results** with current models
3. **Choose best model** for production

