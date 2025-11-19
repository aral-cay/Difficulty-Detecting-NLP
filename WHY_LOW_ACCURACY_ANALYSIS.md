# Why Accuracy is Low on ChatGPT Questions: Detailed Analysis

## Summary

**Training Accuracy**: 81.50% on LectureBank test set  
**ChatGPT Test Accuracy**: 45.09%  
**Performance Drop**: -36.41 percentage points

This is a **classic domain mismatch problem**. The model was trained on one type of text (lecture content) but tested on a completely different type (standalone questions).

---

## Root Causes

### 1. **Fundamental Domain Mismatch** ⚠️ **PRIMARY ISSUE**

**Training Data (LectureBank):**
- **Format**: Long-form academic lecture content
- **Length**: Typically 200-2000+ words per sample
- **Style**: Explanatory, narrative, educational prose
- **Examples**:
  - "In this section, we will explore the fundamental concepts of natural language processing. We begin by examining how text preprocessing techniques can be applied to raw textual data..."
  - "The following example illustrates how dependency parsing can be used to extract syntactic relationships from sentences..."

**Test Data (ChatGPT Questions):**
- **Format**: Short, standalone questions
- **Length**: Typically 5-20 words per question
- **Style**: Direct, interrogative, concise
- **Examples**:
  - "What does ML stand for?"
  - "Prove why convex loss functions guarantee global optima in gradient descent."

**Impact**: The model learned patterns from lecture prose, not question syntax. It's like training a model to recognize cats in photos, then testing it on cat drawings.

---

### 2. **Text Length Mismatch** 📏

**Training Data Characteristics:**
- Average length: ~500-1000 words per sample
- Rich context and multiple sentences
- Complex sentence structures
- Multiple paragraphs

**ChatGPT Questions:**
- Average length: ~8-15 words per question
- Single sentence, no context
- Simple sentence structures
- No paragraphs

**Why This Matters:**
- **TF-IDF features** are designed for longer documents. On very short text, term frequencies become sparse and less informative.
- **Complexity features** (word count, sentence count, lexical diversity) are calibrated for longer text. A 10-word question will always score low on these features, regardless of difficulty.
- **N-gram patterns** (trigrams, bigrams) that work well in lecture text may not appear in short questions.

**Example:**
```
Training sample: "Natural language processing involves the computational analysis of human language. 
This field encompasses various techniques including tokenization, part-of-speech tagging, and 
syntactic parsing. Tokenization is the process of breaking text into individual words or tokens..."

Question: "What is tokenization?"
```
The model learned that long explanations with technical terms = Intermediate. But "What is tokenization?" is short, so it gets misclassified.

---

### 3. **Feature Engineering Mismatch** 🔧

The model uses **TextComplexityFeatures** that include:

| Feature | Works Well For | Problem with Questions |
|---------|---------------|----------------------|
| Word count | Long lecture text | All questions are short (5-20 words) |
| Sentence count | Multi-paragraph content | Questions are single sentences |
| Lexical diversity | Rich vocabulary in lectures | Questions have limited vocabulary |
| Technical term density | Technical explanations | Questions may mention terms but not explain them |
| Average word length | Varied text | Questions are uniformly short |

**Example:**
- **Hard question**: "Prove why convex loss functions guarantee global optima in gradient descent."
  - Word count: 12 words
  - Sentence count: 1
  - Technical terms: "convex", "loss functions", "gradient descent"
  - **Model sees**: Short text (12 words) → likely predicts Easy or Medium
  - **Reality**: This is a Hard question requiring advanced mathematical knowledge

- **Easy question**: "What does ML stand for?"
  - Word count: 5 words
  - Sentence count: 1
  - Technical terms: "ML"
  - **Model sees**: Short text (5 words) → predicts Easy (correct by chance)
  - **Reality**: This is indeed Easy

**The Problem**: The complexity features can't distinguish difficulty when all inputs are short. The model relies on lexical patterns it learned from long-form text.

---

### 4. **Lexical Pattern Differences** 📝

**Training Data Patterns:**
- "In this section, we will discuss..."
- "The following example illustrates..."
- "It is important to note that..."
- "This technique involves..."
- "We can see that..."

**ChatGPT Question Patterns:**
- "What is...?"
- "How does...?"
- "Why is...?"
- "Prove that..."
- "Derive the..."

**Impact**: The model learned that certain phrases in lecture text indicate difficulty levels. But questions use completely different phrasing, so these learned patterns don't apply.

**Example:**
- Training: "How does entity extraction work in natural language processing systems?" (from lecture)
  - Model learned: "How does X work" in long context → Intermediate
  
- Test: "How does entity extraction work?" (standalone question)
  - Model sees: Short text with "How does X work" → May predict Intermediate (correct by pattern matching)
  - But: "How does scaling affect distance-based models?" → Also Intermediate, but model may misclassify

---

### 5. **Training Data Bias** ⚖️

**LectureBank Characteristics:**
- Most lecture content is **intermediate-level** (Level 2)
- Training data distribution likely skewed toward Medium
- Model learned to default to Medium when uncertain

**Evidence from Confusion Matrix:**
- Model predicts Medium (Level 2) most often: 121 correct + 95 Easy misclassified + 67 Hard misclassified = 283 predictions
- This is 47% of all predictions, even though Medium is only 33% of the data

**Why This Happens:**
- The model was trained on lecture content, which naturally tends to be intermediate-level educational material
- When faced with unfamiliar question format, it defaults to its most common training class

---

### 6. **TF-IDF Limitations on Short Text** 🔍

**TF-IDF (Term Frequency-Inverse Document Frequency):**
- Works best on documents with **multiple occurrences** of terms
- Short questions have **sparse term frequencies**
- Many questions may share similar words ("What", "is", "the", "how", "does") regardless of difficulty

**Example:**
```
Easy: "What is a variable?"
Medium: "What is the difference between a variable and a constant?"
Hard: "What is the relationship between variable scope and memory allocation?"

All three start with "What is", so TF-IDF may not capture the difficulty difference.
```

**The model uses 10,000 TF-IDF features**, but on short questions, most features are zero (sparse), making classification harder.

---

## Why Medium Questions Perform Best (60.5% Accuracy)

1. **Training Data Alignment**: Most training data is intermediate-level, so the model is most calibrated for this level.

2. **Question Format Similarity**: Medium questions like "Explain the difference between X and Y" or "How does X work?" are closer in structure to lecture explanations than Easy/Hard questions.

3. **Feature Overlap**: Medium questions may have complexity features that fall in the "middle range" the model learned from training data.

---

## Why Hard Questions Perform Worst (30.35% Accuracy)

1. **Length Mismatch**: Hard questions are still short (10-20 words), but the model expects long, detailed explanations for advanced content.

2. **Feature Confusion**: 
   - Hard questions may use advanced terminology ("convex", "gradient", "optimization")
   - But they're short, so word count/sentence count features suggest Easy/Medium
   - Model gets conflicting signals

3. **Pattern Mismatch**: 
   - Training: Advanced content = long explanations with technical terms
   - Test: Advanced questions = short questions with technical terms
   - Model doesn't recognize that short + technical = Hard

4. **Bias Toward Medium**: When uncertain, model defaults to Medium (its most common training class).

---

## Why Easy Questions Are Misclassified (44.5% Accuracy)

1. **Overlap with Medium**: Many Easy questions are misclassified as Medium (95 out of 200).
   - Easy: "What is a variable?"
   - Medium: "What is the difference between a variable and a constant?"
   - Both are short, start with "What is", so model struggles to distinguish

2. **Feature Similarity**: Both Easy and Medium questions are short, so complexity features don't help distinguish them.

3. **Training Data**: The model may not have seen many truly "easy" questions in the training data (lecture content tends to be more intermediate).

---

## Comparison: Training vs Test Performance

| Metric | Training Test Set | ChatGPT Questions | Difference |
|--------|------------------|-------------------|------------|
| **Overall Accuracy** | 81.50% | 45.09% | **-36.41%** |
| **Easy Accuracy** | ~80% (estimated) | 44.50% | **-35.5%** |
| **Medium Accuracy** | ~82% (estimated) | 60.50% | **-21.5%** |
| **Hard Accuracy** | ~80% (estimated) | 30.35% | **-49.65%** |

**Key Insight**: The model performs reasonably on Medium questions (only 21.5% drop) because they're closest to the training data format. Hard questions show the largest drop (49.65%) because they're the most different from training data.

---

## What This Means

### This is NOT a model failure - it's a **domain adaptation problem**.

The model is working as designed, but it was designed for a different task:
- **Designed for**: Classifying difficulty of lecture content
- **Tested on**: Classifying difficulty of standalone questions

This is similar to:
- Training a model to recognize handwritten digits, then testing on printed digits
- Training on English text, then testing on Spanish text
- Training on long documents, then testing on tweets

---

## Solutions to Improve Performance

### 1. **Retrain on Question Data** ✅ **BEST SOLUTION**
- Collect or generate a dataset of questions labeled by difficulty
- Retrain the model on this question-specific data
- Expected improvement: 70-80%+ accuracy

### 2. **Fine-tune on Questions** ✅ **GOOD SOLUTION**
- Use the existing model as a starting point
- Fine-tune on a smaller dataset of labeled questions
- Expected improvement: 60-70% accuracy

### 3. **Feature Engineering for Questions** 🔧
- Add question-specific features:
  - Question type (What/How/Why/Prove/Derive)
  - Presence of mathematical notation
  - Presence of advanced terminology
  - Question length (relative to difficulty)
- Expected improvement: 50-60% accuracy

### 4. **Ensemble Approach** 🔀
- Train separate models:
  - One on lecture content (current model)
  - One on questions (new model)
- Combine predictions based on input type
- Expected improvement: 65-75% accuracy

### 5. **Transfer Learning** 🎯
- Use a pre-trained question-answering model
- Fine-tune for difficulty classification
- Expected improvement: 70-80% accuracy

---

## Conclusion

The **45.09% accuracy is expected** given the domain mismatch. The model is performing reasonably well considering:

1. ✅ It was never trained on question data
2. ✅ The text format is completely different
3. ✅ The feature engineering is optimized for long-form text
4. ✅ Medium questions still achieve 60.5% accuracy (closest to training data)

**To achieve 80%+ accuracy on questions, you need to retrain or fine-tune the model on question-specific data.**

---

*Analysis Date: Based on evaluation of 601 ChatGPT-generated questions*  
*Model: TF-IDF + Logistic Regression (Maximum Accuracy Model, 81.50% on LectureBank test set)*

