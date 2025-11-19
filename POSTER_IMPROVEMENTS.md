# Poster Improvements and Recommendations

## Current Poster Analysis

Your poster has good structure but can be enhanced with additional visualizations, formulas, and clearer comparisons. Here are specific recommendations:

---

## 📊 **ADDITIONAL GRAPHS/CHARTS TO ADD**

### 1. **Model Comparison Bar Chart** (HIGH PRIORITY)
**Location:** Add next to "Model Performance Comparison"

**Content:**
- **Title:** "TF-IDF vs DistilBERT Performance Comparison"
- **Y-axis:** Accuracy (%)
- **X-axis:** Three categories: "Overall", "Easy", "Medium", "Hard"
- **Bars:** Side-by-side comparison
  - TF-IDF: 45.09% (Overall), 44.50% (Easy), 60.50% (Medium), 30.35% (Hard)
  - DistilBERT: 48.59% (Overall), 63.50% (Easy), 1.50% (Medium), 80.60% (Hard)
- **Colors:** Different colors for each model (e.g., blue for TF-IDF, green for DistilBERT)
- **Insight:** Shows DistilBERT excels at Hard questions but fails on Medium

### 2. **Per-Category Performance Radar Chart** (MEDIUM PRIORITY)
**Location:** Replace or supplement the bar chart

**Content:**
- **Title:** "Per-Category Performance Comparison"
- **Axes:** Three axes (Easy, Medium, Hard)
- **Two lines:**
  - TF-IDF: [44.5, 60.5, 30.35]
  - DistilBERT: [63.5, 1.5, 80.6]
- **Visual:** Shows strengths/weaknesses of each model clearly

### 3. **Precision-Recall Curves** (OPTIONAL)
**Location:** In Results section

**Content:**
- Three curves (one per difficulty level)
- Shows trade-off between precision and recall
- Helps understand model calibration

### 4. **Feature Importance Chart** (HIGH PRIORITY)
**Location:** In Methods section

**Content:**
- **Title:** "Top 10 Most Important Features"
- **Type:** Horizontal bar chart
- **Features to show:**
  - Top TF-IDF n-grams (e.g., "how does", "what is", "explain the")
  - Complexity features (word count, technical term density, etc.)
- **Insight:** Shows what the model uses to make decisions

### 5. **Training Progress Curves** (MEDIUM PRIORITY)
**Location:** In Methods or Results section

**Content:**
- **Title:** "Model Training Progress"
- **X-axis:** Epochs
- **Y-axis:** Accuracy/Loss
- **Lines:**
  - Training accuracy
  - Validation accuracy
  - Training loss
- **Insight:** Shows model convergence and overfitting

### 6. **Dataset Distribution Pie Chart** (LOW PRIORITY)
**Location:** In Methods section

**Content:**
- **Title:** "Dataset Class Distribution"
- **Sections:** Level 1, Level 2, Level 3
- **Values:** Show original distribution and after balancing
- **Insight:** Shows class imbalance and balancing strategy

---

## 📐 **FORMULAS TO ADD**

### 1. **TF-IDF Formula** (ESSENTIAL)
**Location:** Methods section, near Feature Engineering

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

where:
  TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
  IDF(t) = log(Total documents / Number of documents containing term t)
```

### 2. **Precision, Recall, F1-Score Formulas** (ESSENTIAL)
**Location:** Results section, near confusion matrix

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

where:
  TP = True Positives
  FP = False Positives
  FN = False Negatives
```

### 3. **Logistic Regression Formula** (MEDIUM PRIORITY)
**Location:** Methods section

```
P(y=1|x) = 1 / (1 + e^(-z))

where:
  z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
  w = learned weights
  x = feature vector
```

### 4. **Text Complexity Features** (OPTIONAL)
**Location:** Methods section

```
Lexical Diversity = Unique Words / Total Words
Technical Density = Technical Terms / Total Words
Advanced Density = Advanced Terms / Total Words
```

---

## 📝 **CONTENT IMPROVEMENTS**

### 1. **Abstract - Update Numbers**
**Current:** "achieving 96.8% accuracy"
**Issue:** This seems incorrect based on your results (80.12%)
**Fix:** Update to "achieving 80.12% accuracy on test set"

### 2. **Add Model Architecture Diagram** (HIGH PRIORITY)
**Location:** Methods section

**Content:**
```
Input Text
    ↓
[Text Preprocessing]
    ↓
[Feature Extraction]
    ├─→ TF-IDF Vectorization (10,000 features)
    └─→ Complexity Features (20 features)
    ↓
[Feature Union]
    ↓
[Logistic Regression Classifier]
    ↓
Output: Difficulty Level (1, 2, or 3)
```

### 3. **Add Data Pipeline Diagram** (MEDIUM PRIORITY)
**Location:** Methods section

**Content:**
```
LectureBank Dataset
    ↓
[Text Extraction] (PDF/PPTX → Text)
    ↓
[Depth Computation] (Topic ID → Depth Level)
    ↓
[Text Chunking] (Long texts → Smaller segments)
    ↓
[Relabeling] (5-level → 3-level)
    ↓
[Train/Val/Test Split] (70/15/15)
    ↓
Training Data
```

### 4. **Enhance Results Section**
**Add:**
- **Per-class metrics table:**
  | Level | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | Beginner | 0.85 | 0.80 | 0.82 | 2,246 |
  | Intermediate | 0.53 | 0.79 | 0.63 | 555 |
  | Advanced | 0.87 | 0.81 | 0.84 | 1,726 |

### 5. **Add Key Findings Section** (NEW)
**Location:** After Results, before Conclusion

**Content:**
- ✅ **80.12% accuracy** on LectureBank test set
- ✅ **Best performance on Advanced** (81.2% accuracy)
- ⚠️ **Intermediate class** shows lower precision (53%) due to boundary confusion
- ✅ **Feature engineering** (TF-IDF + complexity) improves performance
- ⚠️ **Domain mismatch** on ChatGPT questions (45% vs 80%)

### 6. **Update Conclusion**
**Add:**
- Comparison with baseline (mention what baseline was)
- Limitations (domain mismatch, Medium class challenges)
- Future work (ensemble methods, domain adaptation)

---

## 🎨 **VISUAL IMPROVEMENTS**

### 1. **Confusion Matrix - Current Issues**
- **Fix:** The percentages in your poster don't match your actual results
- **Current poster shows:** Level 2 → Level 2: 73.7% (437)
- **Actual results:** Level 2 → Level 2: 78.74% (437) ✓ (This is correct)
- **Verify:** Make sure all numbers match your `test_confusion_matrix.txt`

### 2. **Color Scheme**
- Use consistent colors throughout
- **Suggestions:**
  - Level 1 (Beginner): Green
  - Level 2 (Intermediate): Yellow/Orange
  - Level 3 (Advanced): Red
  - TF-IDF: Blue
  - DistilBERT: Purple

### 3. **Add Visual Hierarchy**
- Use larger fonts for key numbers (80.12%, 4,527 samples)
- Bold important findings
- Use icons/symbols for different sections

---

## 📋 **SPECIFIC NUMBERS TO VERIFY/UPDATE**

### From Your Results:
- **Test Set Accuracy:** 80.12% ✓
- **Test Samples:** 4,527 ✓
- **Training Samples:** ~10,234 (after oversampling)
- **Per-Class Accuracy:**
  - Level 1: 79.61%
  - Level 2: 78.74%
  - Level 3: 81.23%

### ChatGPT Test Results (Add as separate section):
- **TF-IDF:** 45.09% overall
- **DistilBERT:** 48.59% overall
- **Key insight:** Domain mismatch reduces performance significantly

---

## 🔬 **TECHNICAL DETAILS TO ADD**

### 1. **Hyperparameters Table**
**Location:** Methods section

| Parameter | TF-IDF Model | DistilBERT Model |
|-----------|--------------|------------------|
| Max Features | 10,000 | 256 (max length) |
| N-gram Range | (1, 3) | - |
| Learning Rate | - | 3e-5 |
| Batch Size | - | 32 |
| Epochs | - | 4 |
| Class Weights | Balanced + 2x Intermediate | Balanced |

### 2. **Feature Engineering Details**
- **TF-IDF:** 10,000 features, trigrams, English stopwords
- **Complexity Features:** 20 features (word count, lexical diversity, technical terms, etc.)
- **Total Features:** 10,020

### 3. **Training Details**
- **Oversampling:** SMOTE for class balancing
- **Cross-validation:** Train/Val/Test split (70/15/15)
- **Early Stopping:** Used for DistilBERT (patience=2)

---

## 📊 **RECOMMENDED POSTER LAYOUT**

### Left Column:
1. **Title & Authors**
2. **Abstract**
3. **Introduction**
4. **Methods** (with diagrams)

### Middle Column:
5. **Data Pipeline Diagram**
6. **Feature Engineering** (with formulas)
7. **Model Architecture**
8. **Results** (confusion matrix + metrics table)

### Right Column:
9. **Model Comparison Charts**
10. **ChatGPT Test Results** (NEW)
11. **Key Findings**
12. **Conclusion & Future Work**

---

## 🎯 **PRIORITY RANKING**

### **MUST ADD (High Priority):**
1. ✅ Fix accuracy number in abstract (96.8% → 80.12%)
2. ✅ Add TF-IDF formula
3. ✅ Add Precision/Recall/F1 formulas
4. ✅ Add model comparison bar chart (TF-IDF vs DistilBERT)
5. ✅ Add per-class metrics table
6. ✅ Verify confusion matrix numbers match actual results

### **SHOULD ADD (Medium Priority):**
7. ✅ Add model architecture diagram
8. ✅ Add data pipeline diagram
9. ✅ Add feature importance chart
10. ✅ Add ChatGPT test results section
11. ✅ Add key findings section

### **NICE TO HAVE (Low Priority):**
12. ✅ Add training progress curves
13. ✅ Add precision-recall curves
14. ✅ Add dataset distribution chart
15. ✅ Add hyperparameters table

---

## 📝 **SAMPLE TEXT FOR NEW SECTIONS**

### Key Findings Section:
```
KEY FINDINGS

• Achieved 80.12% accuracy on LectureBank test set
• Best performance on Advanced level (81.2% accuracy)
• Intermediate class shows lower precision (53%) due to 
  boundary confusion with Beginner/Advanced
• Feature engineering (TF-IDF + complexity features) 
  significantly improves classification
• Domain mismatch observed: 45% accuracy on ChatGPT 
  questions vs 80% on lecture content
• DistilBERT excels at Hard questions (80.6%) but struggles 
  with Medium (1.5%)
```

### ChatGPT Test Results Section:
```
EXTERNAL VALIDATION

To test generalizability, we evaluated models on 601 
ChatGPT-generated questions:

TF-IDF Model:
• Overall: 45.09% accuracy
• Best on Medium: 60.50%

DistilBERT Model:
• Overall: 48.59% accuracy  
• Best on Hard: 80.60%
• Struggles with Medium: 1.50%

Key Insight: Domain mismatch (lecture content vs questions) 
reduces performance, highlighting need for domain adaptation.
```

---

## 🔧 **TOOLS FOR CREATING CHARTS**

1. **Python (Matplotlib/Seaborn):** For all statistical charts
2. **PowerPoint/Keynote:** For diagrams and layout
3. **LaTeX (Beamer):** For professional academic posters
4. **Canva/Adobe Illustrator:** For design and visual polish

---

## ✅ **CHECKLIST BEFORE PRINTING**

- [ ] All numbers match actual results
- [ ] Formulas are correct and properly formatted
- [ ] Charts are clear and readable
- [ ] Color scheme is consistent
- [ ] Font sizes are appropriate (readable from 3-6 feet)
- [ ] All citations are included
- [ ] Dartmouth logo is present
- [ ] Contact information is included
- [ ] Poster dimensions match requirements

---

*Good luck with your poster presentation!*

