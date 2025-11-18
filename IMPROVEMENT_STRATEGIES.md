# Strategies to Improve Model Accuracy

## Current Performance Summary

**Sklearn Model: 79.1% accuracy**
- Beginner: 88.3% precision, 77.4% recall, 82.5% F1 ✅
- Intermediate: **47.7% precision**, 84.8% recall, 61.0% F1 ⚠️ (main issue)
- Advanced: 86.7% precision, 79.5% recall, 82.9% F1 ✅

**Main Challenge:** Intermediate class has low precision (high false positives)

---

## Improvement Strategies (Ranked by Expected Impact)

### 1. **Ensemble Methods** ⭐⭐⭐⭐⭐ (Expected: +2-4% accuracy)

**Combine Sklearn + DistilBERT predictions:**
- **Voting Ensemble**: Average probabilities from both models
- **Stacking**: Train meta-learner on both models' predictions
- **Weighted Average**: Weight by validation performance

**Implementation:**
```python
# Weighted ensemble (Sklearn 60%, DistilBERT 40% based on test accuracy)
final_prob = 0.6 * sklearn_probs + 0.4 * distilbert_probs
```

**Expected Improvement:**
- Overall accuracy: 79.1% → **81-83%**
- Intermediate precision: 47.7% → **52-58%**

---

### 2. **Hyperparameter Tuning** ⭐⭐⭐⭐ (Expected: +1-2% accuracy)

**For Sklearn:**
- **C parameter**: Try [0.1, 0.5, 1.0, 2.0, 5.0] (regularization strength)
- **Solver**: Try 'lbfgs', 'liblinear', 'sag'
- **max_features**: Try [5000, 10000, 15000, 20000]
- **ngram_range**: Try (1,2), (1,3), (2,3)

**For DistilBERT:**
- **learning_rate**: Try [2e-5, 3e-5, 5e-5] (lower might help)
- **max_length**: Try [256, 512] (longer context)
- **batch_size**: Try [8, 16, 32]
- **epochs**: Early stopping at optimal point (was epoch 4)

**Expected Improvement:**
- Overall accuracy: 79.1% → **80-81%**
- Better Intermediate precision

---

### 3. **Advanced Feature Engineering** ⭐⭐⭐⭐ (Expected: +1-3% accuracy)

**Add linguistic complexity features:**
```python
# Text complexity metrics
- Text length (characters, words, sentences)
- Average word length
- Lexical diversity (unique words / total words)
- Average sentence length
- Question word indicators ("what", "how", "explain")
- Technical term density
- Passive voice ratio
- Subordination index
```

**Combine with TF-IDF:**
```python
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

# Combine TF-IDF + handcrafted features
features = FeatureUnion([
    ('tfidf', TfidfVectorizer(...)),
    ('complexity', StandardScaler().fit_transform(complexity_features))
])
```

**Expected Improvement:**
- Intermediate precision: 47.7% → **52-60%**
- Better distinction between levels

---

### 4. **Better Data Augmentation** ⭐⭐⭐ (Expected: +0.5-1.5% accuracy)

**Current:** Simple resampling (duplicates)

**Better approaches:**
- **SMOTE (Synthetic Minority Oversampling)**: Generate synthetic Intermediate examples
- **Paraphrasing**: Use GPT/LLM to paraphrase Intermediate examples
- **Synonym replacement**: Replace words with synonyms
- **Back-translation**: Translate to another language and back

**Focus on Intermediate class:**
```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to Intermediate class only
smote = SMOTE(sampling_strategy={2: intermediate_target}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_vec, y_train)
```

**Expected Improvement:**
- Intermediate precision: 47.7% → **50-55%**
- More diverse Intermediate examples

---

### 5. **Multi-Stage Classification** ⭐⭐⭐ (Expected: +1-2% accuracy)

**Two-stage approach:**
1. **Stage 1**: Binary classifier (Intermediate vs. Not-Intermediate)
2. **Stage 2**: 3-class classifier for Beginner/Advanced

**Implementation:**
```python
# Stage 1: Is it Intermediate?
is_intermediate = binary_classifier.predict(text)

if is_intermediate:
    return "Intermediate"
else:
    # Stage 2: Beginner or Advanced?
    return three_class_classifier.predict(text)
```

**Expected Improvement:**
- Intermediate precision: 47.7% → **55-65%**
- Better separation of Intermediate from others

---

### 6. **Try Different Algorithms** ⭐⭐⭐ (Expected: +0.5-2% accuracy)

**Alternative models:**
- **SVM with RBF kernel**: Better for non-linear boundaries
- **Random Forest**: Handles feature interactions well
- **XGBoost**: Gradient boosting, often best for tabular data
- **Naive Bayes**: Fast baseline, good for text

**Implementation:**
```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Try each with same features
models = {
    'svm': SVC(kernel='rbf', class_weight='balanced'),
    'rf': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
    'xgb': XGBClassifier(scale_pos_weight=2.0)
}
```

**Expected Improvement:**
- Overall accuracy: 79.1% → **80-81%**
- Different models may handle Intermediate better

---

### 7. **Probability Calibration** ⭐⭐ (Expected: +0.5-1% accuracy)

**Calibrate probabilities for better threshold tuning:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate probabilities
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3)
calibrated_clf.fit(X_train, y_train)
```

**Better threshold optimization:**
- Use validation set to find optimal thresholds
- Optimize for F1-score or balanced accuracy
- Different thresholds for each class

**Expected Improvement:**
- Better probability estimates
- More reliable confidence scores

---

### 8. **Feature Selection** ⭐⭐ (Expected: +0.5-1% accuracy)

**Remove noisy features:**
```python
from sklearn.feature_selection import SelectKBest, chi2

# Select top K features
selector = SelectKBest(chi2, k=5000)
X_selected = selector.fit_transform(X_train_vec, y_train)
```

**Expected Improvement:**
- Reduce overfitting
- Faster training
- Better generalization

---

### 9. **Better Class Weight Optimization** ⭐⭐ (Expected: +0.5-1% accuracy)

**Current:** 2x weight for Intermediate

**Try:**
- Grid search for optimal weights: [1.5x, 2.0x, 2.5x, 3.0x]
- Optimize for F1-score instead of accuracy
- Different weights for precision vs. recall

**Expected Improvement:**
- Better Intermediate precision/recall balance

---

### 10. **Domain-Specific Pre-training** ⭐⭐⭐ (Long-term, +2-5% accuracy)

**For DistilBERT:**
- Continue pre-training on lecture/educational content
- Use domain-specific vocabulary
- Fine-tune on similar tasks first

**Expected Improvement:**
- Better semantic understanding
- Higher accuracy overall

---

## Recommended Implementation Order

### Quick Wins (1-2 hours):
1. ✅ **Ensemble Method** - Combine Sklearn + DistilBERT
2. ✅ **Hyperparameter Tuning** - Grid search for C parameter
3. ✅ **Feature Engineering** - Add text complexity metrics

### Medium Effort (4-8 hours):
4. ✅ **SMOTE** - Better data augmentation
5. ✅ **Multi-Stage Classification** - Two-stage approach
6. ✅ **Try XGBoost** - Alternative algorithm

### Long-term (Days):
7. ✅ **Domain Pre-training** - Continue training DistilBERT
8. ✅ **Large-scale Data Collection** - More training data

---

## Expected Combined Improvement

**If implementing top 3 strategies:**
- Current: **79.1% accuracy**
- Expected: **82-85% accuracy** (+3-6%)
- Intermediate precision: **47.7% → 55-65%** (+7-17%)

**If implementing top 5 strategies:**
- Expected: **83-86% accuracy** (+4-7%)
- Intermediate precision: **47.7% → 58-70%** (+10-22%)

---

## Quick Start: Ensemble Implementation

The easiest and most effective improvement is to create an ensemble. Would you like me to implement this first?

