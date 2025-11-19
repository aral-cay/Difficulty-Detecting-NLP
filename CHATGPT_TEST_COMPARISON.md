# ChatGPT Test Results: TF-IDF vs DistilBERT Comparison

## Overview
Comparison of model performance on 601 ChatGPT-generated questions:
- **TF-IDF + Logistic Regression**: Maximum accuracy model
- **DistilBERT**: Fast training model (4 epochs, 256 max length)

---

## Overall Performance

| Model | Overall Accuracy | Macro F1 | Weighted F1 |
|-------|-----------------|----------|-------------|
| **TF-IDF** | **45.09%** | 0.4447 | 0.4446 |
| **DistilBERT** | **48.59%** | 0.3962 | 0.3966 |

**Winner: DistilBERT** (+3.5% accuracy)

---

## Per-Category Performance

### Level 1 (Easy) - 200 questions

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **TF-IDF** | 44.50% | 0.3991 | 0.4450 | 0.4208 |
| **DistilBERT** | **63.50%** | 0.4829 | 0.6350 | 0.5486 |

**Winner: DistilBERT** (+19% accuracy)

### Level 2 (Medium) - 200 questions

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **TF-IDF** | **60.50%** | 0.4276 | 0.6050 | 0.5010 |
| **DistilBERT** | 1.50% | 0.3333 | 0.0150 | 0.0287 |

**Winner: TF-IDF** (DistilBERT almost never predicts Medium)

### Level 3 (Hard) - 201 questions

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **TF-IDF** | 30.35% | 0.6421 | 0.3035 | 0.4122 |
| **DistilBERT** | **80.60%** | 0.4924 | 0.8060 | 0.6113 |

**Winner: DistilBERT** (+50.25% accuracy - huge improvement!)

---

## Confusion Matrices

### TF-IDF Model

| True \ Predicted | Level 1 | Level 2 | Level 3 |
|------------------|---------|---------|---------|
| **Level 1** | **79.6%** | 12.6% | 7.8% |
| **Level 2** | 16.6% | **78.7%** | 4.7% |
| **Level 3** | 12.6% | 6.1% | **81.2%** |

### DistilBERT Model

| True \ Predicted | Level 1 | Level 2 | Level 3 |
|------------------|---------|---------|---------|
| **Level 1** | **63.5%** | 3.0% | 33.5% |
| **Level 2** | 48.5% | **1.5%** | 50.0% |
| **Level 3** | 19.4% | 0.0% | **80.6%** |

---

## Key Observations

### DistilBERT Strengths:
1. **Excellent Hard Question Detection**: 80.60% accuracy on Hard questions (vs 30.35% for TF-IDF)
   - Much better at identifying advanced content
   - High recall (80.6%) for Hard questions

2. **Better Easy Question Detection**: 63.50% accuracy (vs 44.50% for TF-IDF)
   - Improved by 19 percentage points

3. **Higher Overall Accuracy**: 48.59% vs 45.09%

### DistilBERT Weaknesses:
1. **Severe Medium Question Problem**: Only 1.5% accuracy on Medium questions
   - Almost never predicts Medium (only 3 out of 200 correct)
   - 97 misclassified as Easy, 100 as Hard
   - This is a critical failure for the Medium category

2. **Lower F1-Scores**: Despite higher accuracy, macro F1 is lower (0.3962 vs 0.4447)
   - This is due to the terrible Medium performance

3. **Class Imbalance**: Strong bias toward predicting Level 1 and Level 3, avoiding Level 2

### TF-IDF Strengths:
1. **Balanced Performance**: Reasonable accuracy across all three categories
2. **Better Medium Detection**: 60.50% accuracy (vs 1.5% for DistilBERT)
3. **Higher F1-Scores**: Better balance between precision and recall

### TF-IDF Weaknesses:
1. **Poor Hard Question Detection**: Only 30.35% accuracy
2. **Lower Overall Accuracy**: 45.09% vs 48.59%

---

## Analysis

### Why DistilBERT Struggles with Medium Questions:

1. **Training Data Distribution**: The model was trained on lecture content where most samples are either clearly Beginner or Advanced. The "Intermediate" category may have been underrepresented or ambiguous.

2. **Binary Classification Tendency**: DistilBERT seems to treat this as a binary problem (Easy vs Hard), rarely predicting the middle category.

3. **Semantic Understanding**: While DistilBERT's semantic understanding helps with Hard questions, it may not have learned clear distinguishing features for Intermediate content.

### Why DistilBERT Excels at Hard Questions:

1. **Semantic Features**: Transformer models capture semantic meaning better than TF-IDF
2. **Advanced Terminology Recognition**: Better at understanding complex concepts and terminology
3. **Context Understanding**: Can better understand the depth and complexity of explanations

---

## Recommendations

### For DistilBERT:
1. **Class Weighting**: Increase weight for Medium class during training
2. **More Medium Examples**: Oversample Medium examples in training data
3. **Threshold Tuning**: Adjust decision thresholds to encourage Medium predictions
4. **Ensemble**: Combine with TF-IDF model to leverage both strengths

### For Production Use:
- **Hybrid Approach**: Use DistilBERT for Hard question detection, TF-IDF for Medium
- **Two-Stage Classification**: First classify Easy vs Not-Easy, then classify Medium vs Hard
- **Calibration**: Apply post-processing to balance predictions across all three classes

---

## Conclusion

**DistilBERT shows promise** with significantly better performance on Hard questions (+50%) and Easy questions (+19%), but **fails catastrophically on Medium questions** (1.5% vs 60.5%).

**TF-IDF provides more balanced performance** across all categories, making it more reliable for general use despite lower overall accuracy.

**Best approach**: Use an ensemble or hybrid system that leverages DistilBERT's strength on Hard questions while maintaining TF-IDF's balanced performance.

---

*Evaluation Date: Based on 601 ChatGPT-generated questions*  
*TF-IDF Model: `models/tfidf_logreg_3level_max`*  
*DistilBERT Model: `models/distilbert_depth3_fast/best`*

