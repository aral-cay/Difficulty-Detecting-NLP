# ChatGPT Test Evaluation Results

## Overview
Evaluation of the TF-IDF + Logistic Regression model on 601 ChatGPT-generated questions:
- **Easy (Level 1)**: 200 questions
- **Medium (Level 2)**: 200 questions  
- **Hard (Level 3)**: 201 questions

## Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **45.09%** |
| **Macro F1-Score** | **0.4447** |
| **Weighted F1-Score** | **0.4446** |

## Per-Category Performance

| Category | Accuracy | Precision | Recall | F1-Score | Support |
|----------|----------|-----------|--------|----------|---------|
| **Easy (Level 1)** | 44.50% | 0.3991 | 0.4450 | 0.4208 | 200 |
| **Medium (Level 2)** | **60.50%** | 0.4276 | **0.6050** | **0.5010** | 200 |
| **Hard (Level 3)** | 30.35% | **0.6421** | 0.3035 | 0.4122 | 201 |

## Confusion Matrix

| True \ Predicted | Level 1 (Easy) | Level 2 (Medium) | Level 3 (Hard) |
|------------------|----------------|------------------|----------------|
| **Level 1 (Easy)** | **89** ✅ | 95 ❌ | 16 ❌ |
| **Level 2 (Medium)** | 61 ❌ | **121** ✅ | 18 ❌ |
| **Level 3 (Hard)** | 73 ❌ | 67 ❌ | **61** ✅ |

## Key Observations

### Strengths
1. **Best Performance on Medium Questions**: The model achieves 60.5% accuracy on Medium questions, which aligns with the training data (lecture content tends to be intermediate-level).
2. **High Precision for Hard Questions**: When the model predicts "Hard", it's correct 64.21% of the time (highest precision).
3. **Good Recall for Medium**: The model correctly identifies 60.5% of Medium questions (highest recall).

### Weaknesses
1. **Low Overall Accuracy**: 45.09% overall accuracy is below the 81.50% achieved on the test set, suggesting domain mismatch between training data (lecture content) and test data (ChatGPT-generated questions).
2. **Poor Hard Question Detection**: Only 30.35% of Hard questions are correctly identified (low recall). The model tends to misclassify Hard questions as Easy (73) or Medium (67).
3. **Difficulty Distinguishing Easy from Medium**: 95 Easy questions are misclassified as Medium, and 61 Medium questions are misclassified as Easy, indicating the model struggles with the Easy/Medium boundary.

## Analysis

### Why the Performance Gap?

1. **Domain Mismatch**: The model was trained on LectureBank content (academic lecture materials), while the test questions are ChatGPT-generated standalone questions. The linguistic patterns and complexity indicators may differ.

2. **Question Format**: ChatGPT questions are often more direct and concise compared to lecture content, which may affect feature extraction (TF-IDF and complexity features).

3. **Class Imbalance in Predictions**: The model shows a bias toward predicting Medium (Level 2), which is consistent with the training data distribution.

### Recommendations

1. **Fine-tune on Question Data**: Retrain or fine-tune the model on a dataset of questions labeled by difficulty level.
2. **Feature Engineering**: Add question-specific features (e.g., question type, answer length expectations, prerequisite knowledge indicators).
3. **Ensemble Methods**: Combine predictions from multiple models trained on different data types.
4. **Threshold Tuning**: Adjust classification thresholds to improve recall for Hard questions.

## Detailed Metrics

### Macro Averages
- **Precision**: 0.4896
- **Recall**: 0.4512
- **F1-Score**: 0.4447

### Weighted Averages
- **Precision**: 0.4898
- **Recall**: 0.4509
- **F1-Score**: 0.4446

---

*Evaluation performed on: 601 ChatGPT-generated questions*  
*Model: TF-IDF + Logistic Regression (Maximum Accuracy Model)*  
*Results saved to: `results/chatgpt_test_evaluation.txt`*

