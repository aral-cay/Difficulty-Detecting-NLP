# Methods Section Improvements

## 🔍 **CURRENT ISSUES WITH YOUR METHODS SECTION**

1. **Too text-heavy** - All bullet points, no visuals
2. **Missing diagrams** - No pipeline or architecture visualization
3. **No formulas** - TF-IDF and other formulas not shown
4. **Lacks specificity** - Missing hyperparameters and technical details
5. **No clear structure** - Could be organized into subsections

---

## ✅ **RECOMMENDED CHANGES**

### **1. RESTRUCTURE INTO SUBSECTIONS** (HIGH PRIORITY)

Split into clear subsections:

```
METHODS

1. Data Preprocessing
2. Feature Engineering  
3. Model Architecture
4. Training Strategy
```

---

### **2. ADD VISUAL DIAGRAMS** (HIGH PRIORITY)

#### **A. Data Pipeline Diagram**

Replace text with a visual flowchart:

```
┌─────────────┐
│ Raw Files   │
│ (PDF/PPTX)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Text Extraction │
└──────┬──────────┘
       │
       ▼
┌──────────────────┐
│ Depth Computation│
│ (ConceptBank)    │
└──────┬───────────┘
       │
       ▼
┌─────────────┐
│ Text        │
│ Chunking    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Relabeling  │
│ (5→3 levels)│
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Train/Val/Test  │
│ Split (70/15/15)│
└─────────────────┘
```

#### **B. Model Architecture Diagram**

```
Input Text
    │
    ├─→ [TF-IDF Vectorizer]
    │   • 10,000 features
    │   • N-grams (1-3)
    │   • Stopwords removed
    │
    ├─→ [Complexity Features]
    │   • 20 features
    │   • Word count, lexical diversity
    │   • Technical term density
    │
    └─→ [Feature Union]
        │
        ▼
    [Logistic Regression]
    • C=3.0
    • Class weights: 2x Intermediate
        │
        ▼
    Output: Difficulty Level (1, 2, or 3)
```

---

### **3. ADD FORMULAS** (HIGH PRIORITY)

Add these formulas to your Methods section:

```
FEATURE ENGINEERING

TF-IDF Formula:
TF-IDF(t,d) = TF(t,d) × IDF(t)

where:
  TF(t,d) = (count of t in d) / (total terms in d)
  IDF(t) = log(N / df(t))

Complexity Features:
• Lexical Diversity = Unique Words / Total Words
• Technical Density = Technical Terms / Total Words
• Advanced Density = Advanced Terms / Total Words
```

---

### **4. ADD HYPERPARAMETERS TABLE** (HIGH PRIORITY)

Replace vague descriptions with a specific table:

```
HYPERPARAMETERS

┌─────────────────────┬──────────────────┬──────────────────┐
│ Parameter           │ TF-IDF Model     │ DistilBERT Model │
├─────────────────────┼──────────────────┼──────────────────┤
│ Max Features        │ 10,000           │ 256 (max length) │
│ N-gram Range        │ (1, 3)           │ -                │
│ Learning Rate       │ -                │ 3e-5             │
│ Batch Size          │ -                │ 32               │
│ Epochs              │ -                │ 4                │
│ Class Weights       │ 2x Intermediate  │ Balanced         │
│ Regularization (C)  │ 3.0              │ -                │
│ Oversampling        │ 1.5x Intermediate│ 1.5x Intermediate│
└─────────────────────┴──────────────────┴──────────────────┘
```

---

### **5. IMPROVE TEXT CLARITY** (MEDIUM PRIORITY)

**Current text is too vague. Make it specific:**

**Before:**
"The LectureBank text is extracted after lower-casing and stripping the punctuation and special characters."

**After:**
"Text Extraction: Extract raw text from PDF and PPTX files using PyMuPDF and python-pptx libraries. Preprocess by lowercasing, removing punctuation, and filtering special characters."

---

## 📋 **RECOMMENDED METHODS SECTION STRUCTURE**

### **Option 1: Two-Column Layout**

```
┌─────────────────────────┬─────────────────────────┐
│ DATA PREPROCESSING      │ FEATURE ENGINEERING     │
│                         │                         │
│ [Pipeline Diagram]      │ [Architecture Diagram]  │
│                         │                         │
│ 1. Text Extraction      │ TF-IDF Features:        │
│ 2. Depth Computation    │ • 10,000 features       │
│ 3. Text Chunking        │ • N-grams (1-3)         │
│ 4. Relabeling (5→3)     │ • Stopwords removed     │
│ 5. Train/Val/Test Split │                         │
│                         │ Complexity Features:    │
│                         │ • 20 handcrafted        │
│                         │ • Lexical diversity     │
│                         │ • Technical terms       │
│                         │                         │
│                         │ [TF-IDF Formula]        │
└─────────────────────────┴─────────────────────────┘

┌─────────────────────────┬─────────────────────────┐
│ MODEL TRAINING          │ HYPERPARAMETERS         │
│                         │                         │
│ TF-IDF + Logistic Reg:  │ [Hyperparameters Table] │
│ • Feature Union         │                         │
│ • Class Weighting       │                         │
│ • SMOTE Oversampling    │                         │
│                         │                         │
│ DistilBERT:             │                         │
│ • Fine-tuning           │                         │
│ • Weighted Loss         │                         │
│ • Early Stopping        │                         │
└─────────────────────────┴─────────────────────────┘
```

---

### **Option 2: Improved Bullet Format**

```
METHODS

1. DATA PREPROCESSING
   • Text Extraction: PDF/PPTX → Raw text (PyMuPDF, python-pptx)
   • Depth Computation: ConceptBank taxonomy → Difficulty levels (1-5)
   • Text Chunking: Long texts → 512-word segments (50-word overlap)
   • Relabeling: 5-level → 3-level classification
   • Dataset Split: 70% train, 15% val, 15% test

2. FEATURE ENGINEERING
   • TF-IDF: 10,000 features, trigrams, English stopwords
   • Complexity Features: 20 handcrafted features
     - Word count, sentence count, lexical diversity
     - Technical term density, advanced term count
   • Total: 10,020 features

3. MODEL ARCHITECTURE
   • TF-IDF + Logistic Regression (scikit-learn)
   • DistilBERT (HuggingFace Transformers)
   • Feature Union: Combines TF-IDF + complexity features

4. TRAINING STRATEGY
   • Oversampling: 1.5x boost for Intermediate class
   • Class Weighting: 2x penalty for Intermediate
   • Hyperparameters: [See table below]
```

---

## 🎨 **VISUALS TO ADD**

### **1. Data Pipeline Diagram** (MUST ADD)
- Visual flowchart showing preprocessing steps
- Makes the process clear at a glance

### **2. Model Architecture Diagram** (MUST ADD)
- Shows how features flow through the model
- Visual representation of Feature Union

### **3. Hyperparameters Table** (MUST ADD)
- Specific numbers instead of vague descriptions
- Easy to read and compare

### **4. Feature Engineering Diagram** (OPTIONAL)
- Shows TF-IDF + Complexity features combination
- Visual representation of feature union

---

## 📝 **IMPROVED METHODS SECTION TEXT**

### **Complete Rewrite (Recommended):**

```
METHODS

1. DATA PREPROCESSING

Text Extraction:
• Extract text from PDF files using PyMuPDF
• Extract text from PPTX files using python-pptx
• Preprocess: lowercase, remove punctuation/special characters

Depth Computation:
• Map topic IDs to difficulty levels using ConceptBank taxonomy
• Original: 5-level hierarchy
• Converted: 3-level classification (Beginner, Intermediate, Advanced)

Text Chunking:
• Split long texts into 512-word segments
• 50-word overlap between chunks
• Preserves context while expanding dataset

Dataset Split:
• Training: 70% (10,234 samples after oversampling)
• Validation: 15% (678 samples)
• Test: 15% (4,527 samples)

2. FEATURE ENGINEERING

TF-IDF Features:
• 10,000 most frequent features
• N-gram range: (1, 3) - unigrams, bigrams, trigrams
• English stopwords removed
• Sublinear TF scaling

Complexity Features (20 features):
• Text statistics: word count, sentence count, avg word length
• Lexical diversity: unique words / total words
• Technical term density
• Advanced term count
• Question type indicators

Feature Union:
• Combines TF-IDF (10,000) + Complexity (20) = 10,020 features
• StandardScaler applied to complexity features

3. MODEL TRAINING

TF-IDF + Logistic Regression:
• Regularization: C=3.0
• Class weights: 2x boost for Intermediate
• Oversampling: 1.5x for Intermediate class
• Max iterations: 3000

DistilBERT:
• Base model: distilbert-base-uncased
• Max length: 256 tokens
• Learning rate: 3e-5
• Batch size: 32
• Epochs: 4 (with early stopping)
• Class weights: Balanced
```

---

## 🎯 **SPECIFIC CHANGES TO MAKE**

### **Change 1: Add Pipeline Diagram**
**Replace:** Text description of preprocessing
**With:** Visual flowchart diagram

### **Change 2: Add Formulas**
**Add:** TF-IDF formula and complexity feature formulas
**Location:** Feature Engineering subsection

### **Change 3: Add Hyperparameters Table**
**Replace:** Vague descriptions
**With:** Specific table with numbers

### **Change 4: Split into Subsections**
**Current:** 5 bullet points
**New:** 4 clear subsections (Preprocessing, Features, Models, Training)

### **Change 5: Add Model Architecture Diagram**
**Add:** Visual showing Feature Union → Logistic Regression
**Location:** After Feature Engineering

---

## 📊 **QUICK WINS (Easy Improvements)**

1. **Add hyperparameters table** (5 minutes)
   - Copy the table from above
   - Makes methods more specific

2. **Add TF-IDF formula** (2 minutes)
   - Copy formula from above
   - Shows technical depth

3. **Split into subsections** (10 minutes)
   - Reorganize existing text
   - Makes it easier to read

4. **Add pipeline diagram** (15 minutes)
   - Simple flowchart
   - Much more visual

---

## ✅ **CHECKLIST**

- [ ] Split into clear subsections
- [ ] Add data pipeline diagram
- [ ] Add model architecture diagram
- [ ] Add TF-IDF formula
- [ ] Add hyperparameters table
- [ ] Make text more specific (add numbers)
- [ ] Add feature engineering details
- [ ] Show class balancing strategy

---

## 💡 **PRO TIP**

**Current Methods section is ~80% text, 20% visual**
**Ideal Methods section: ~50% text, 50% visual**

Add diagrams and tables to make it more engaging and easier to understand!

