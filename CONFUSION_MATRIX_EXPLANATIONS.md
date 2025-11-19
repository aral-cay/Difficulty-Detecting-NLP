# Confusion Matrix Explanations: Why Misclassifications Occur

## Overview

This document explains the off-diagonal values in the confusion matrix - why certain difficulty levels are confused with others. The analysis is based on 900 misclassifications out of 4,527 test samples (80.12% accuracy).

---

## Confusion Matrix Summary

| True \ Predicted | Level 1 | Level 2 | Level 3 |
|------------------|---------|---------|---------|
| **Level 1** | **79.6%** ✅ | **12.6%** ❌ | **7.8%** ❌ |
| **Level 2** | **16.6%** ❌ | **78.7%** ✅ | **4.7%** ❌ |
| **Level 3** | **12.6%** ❌ | **6.1%** ❌ | **81.2%** ✅ |

---

## 1. Level 1 (Beginner) → Level 2 (Intermediate)

**Count:** 282 cases (12.6% of Level 1 samples)  
**Average Confidence:** 0.702

### Why This Happens:

**Primary Reason:** Beginner texts that contain intermediate-level terminology or concepts get pushed toward Intermediate classification.

**Key Characteristics:**
- Average word count: **395.8 words** (moderate length)
- Contains some technical terms (1.29 on average)
- May introduce intermediate concepts even in beginner context

**Example:**
```
Text about probability with mathematical notation (P(Heads) = 2/3) 
→ Model sees mathematical notation and technical terms
→ Classifies as Intermediate (confidence: 0.987)
```

**Explanation:** The model associates mathematical notation and technical terminology with Intermediate level, even when the content is explaining basic concepts. The complexity features (word count, technical terms) push these toward Intermediate.

---

## 2. Level 1 (Beginner) → Level 3 (Advanced)

**Count:** 176 cases (7.8% of Level 1 samples)  
**Average Confidence:** 0.616

### Why This Happens:

**Primary Reason:** Beginner texts that mention advanced concepts or use advanced terminology get misclassified as Advanced, even if the explanation itself is basic.

**Key Characteristics:**
- Average word count: **407.4 words**
- Contains advanced terms (1.14 on average) like "Variational Autoencoders", "Latent Variable"
- May include mathematical formulas or advanced notation

**Example:**
```
Text mentioning "Variational Autoencoders" or "Latent Variable Cross-Entropy"
→ Model sees advanced terminology
→ Classifies as Advanced (confidence: 0.938)
```

**Explanation:** The model's feature engineering detects advanced terminology and mathematical complexity, associating these with Advanced level. Even if the text is introducing these concepts at a beginner level, the presence of advanced terms triggers Advanced classification.

---

## 3. Level 2 (Intermediate) → Level 1 (Beginner)

**Count:** 92 cases (16.6% of Level 2 samples)  
**Average Confidence:** 0.665

### Why This Happens:

**Primary Reason:** Intermediate texts that use simpler language or basic terminology get misclassified as Beginner, especially when they lack the complexity indicators the model expects for Intermediate content.

**Key Characteristics:**
- Average word count: **597.4 words** (longer than average)
- Fewer advanced terms (0.80 on average)
- Uses simpler language structure

**Example:**
```
Text explaining machine learning basics with simple language
→ Model sees basic terminology and simple structure
→ Classifies as Beginner (confidence: 0.967)
```

**Explanation:** The model relies on lexical patterns and complexity features. When Intermediate content uses simpler language or lacks technical complexity indicators, it gets classified as Beginner. The model may be too sensitive to surface-level language simplicity.

---

## 4. Level 2 (Intermediate) → Level 3 (Advanced)

**Count:** 26 cases (4.7% of Level 2 samples)  
**Average Confidence:** 0.620

### Why This Happens:

**Primary Reason:** Intermediate texts that contain advanced terminology or concepts get pushed toward Advanced classification, even if the overall complexity is intermediate.

**Key Characteristics:**
- Average word count: **410.1 words**
- Contains advanced terms (1.12 on average)
- May discuss advanced concepts in an intermediate context

**Example:**
```
Text about "factor analysis" or "Bayesian networks" with intermediate-level explanation
→ Model sees advanced terminology
→ Classifies as Advanced (confidence: 0.865)
```

**Explanation:** This is the least common misclassification type. When Intermediate content includes advanced terminology (like "factor analysis", "Bayesian networks"), the model's feature detection pushes it toward Advanced, even if the explanation depth is intermediate.

---

## 5. Level 3 (Advanced) → Level 1 (Beginner)

**Count:** 218 cases (12.6% of Level 3 samples)  
**Average Confidence:** 0.668

### Why This Happens:

**Primary Reason:** Advanced texts that use simpler language structure or basic terminology get misclassified as Beginner, despite covering advanced topics.

**Key Characteristics:**
- Average word count: **422.2 words**
- Contains some advanced terms (1.23 on average) but may use simple language
- May have simpler sentence structure despite advanced content

**Example:**
```
Text about structured prediction or loss functions with simple language
→ Model focuses on language simplicity
→ Classifies as Beginner (confidence: 0.967)
```

**Explanation:** This is a significant issue - the model focuses too much on surface-level language features. When Advanced content is written in simpler language or uses basic terminology, the model misses the advanced concepts and classifies it as Beginner. This suggests the model may need better semantic understanding beyond lexical features.

---

## 6. Level 3 (Advanced) → Level 2 (Intermediate)

**Count:** 106 cases (6.1% of Level 3 samples)  
**Average Confidence:** 0.660

### Why This Happens:

**Primary Reason:** Advanced texts that lack extreme complexity indicators (very long text, many advanced terms) get classified as Intermediate.

**Key Characteristics:**
- Average word count: **389.3 words** (shorter than expected for Advanced)
- Fewer advanced terms (0.90 on average)
- Moderate complexity features

**Example:**
```
Text about knowledge bases or semantic representations with moderate length
→ Model sees moderate complexity
→ Classifies as Intermediate (confidence: 0.967)
```

**Explanation:** The model expects Advanced content to have very high complexity indicators (very long text, many advanced terms, complex sentence structure). When Advanced content is more concise or has moderate complexity features, it gets classified as Intermediate. This suggests the model's threshold for "Advanced" may be too high.

---

## Key Insights

### 1. **Terminology Over Context**
The model heavily relies on terminology detection. Texts with advanced terms are often classified as Advanced, even if the explanation depth is basic or intermediate.

### 2. **Language Simplicity Penalty**
Advanced content written in simpler language gets misclassified as Beginner. The model needs better semantic understanding beyond lexical features.

### 3. **Length and Complexity Correlation**
The model expects Advanced content to be very long with many complexity indicators. More concise Advanced content gets downgraded to Intermediate.

### 4. **Boundary Confusion**
The most common misclassifications occur at boundaries:
- **Level 1 → Level 2**: 282 cases (largest misclassification)
- **Level 3 → Level 1**: 218 cases (second largest)

This suggests the model struggles with distinguishing adjacent difficulty levels.

### 5. **Feature Engineering Limitations**
The current feature engineering (TF-IDF + complexity features) may not capture:
- Semantic depth of explanations
- Conceptual complexity beyond terminology
- Context and explanation quality

---

## Recommendations for Improvement

1. **Semantic Features**: Add features that capture semantic depth, not just terminology presence
2. **Context Awareness**: Consider the explanation quality and depth, not just term presence
3. **Threshold Tuning**: Adjust classification thresholds to better handle boundary cases
4. **Ensemble Methods**: Combine multiple models with different feature sets
5. **Domain-Specific Training**: Fine-tune on more examples of boundary cases

---

*Analysis based on 4,527 test samples with 80.12% overall accuracy*  
*Generated from: `scripts/analyze_misclassifications.py`*

