# Improved Methods Section (Ready to Use)

## 📋 **OPTION 1: CONCISE VERSION (Recommended for Poster)**

```
METHODS

1. DATA PREPROCESSING
   [Include: data_pipeline_diagram.png]
   
   • Text Extraction: PDF/PPTX → Raw text (PyMuPDF, python-pptx)
   • Depth Computation: ConceptBank taxonomy → Difficulty levels (1-5)
   • Text Chunking: Long texts → 512-word segments (50-word overlap)
   • Relabeling: 5-level → 3-level classification (Beginner, Intermediate, Advanced)
   • Dataset Split: 70% train, 15% val, 15% test

2. FEATURE ENGINEERING
   [Include: feature_engineering_diagram.png]
   
   TF-IDF Features (10,000):
   • N-gram range: (1, 3) - unigrams, bigrams, trigrams
   • English stopwords removed
   • Sublinear TF scaling
   
   Complexity Features (20):
   • Word count, sentence count, avg word length
   • Lexical diversity = Unique Words / Total Words
   • Technical term density
   • Advanced term count
   
   Feature Union: 10,020 total features

3. MODEL ARCHITECTURE
   [Include: model_architecture_diagram.png]
   
   • TF-IDF + Logistic Regression (scikit-learn)
   • Feature Union: Combines TF-IDF + complexity features
   • DistilBERT: Fine-tuned distilbert-base-uncased

4. TRAINING STRATEGY
   [Include: Hyperparameters Table]
   
   TF-IDF Model:
   • Regularization: C=3.0
   • Class weights: 2x boost for Intermediate
   • Oversampling: 1.5x for Intermediate class
   
   DistilBERT Model:
   • Learning rate: 3e-5
   • Batch size: 32
   • Epochs: 4 (early stopping)
   • Max length: 256 tokens
```

---

## 📋 **OPTION 2: DETAILED VERSION (If You Have More Space)**

```
METHODS

1. DATA PREPROCESSING
   [Include: data_pipeline_diagram.png]
   
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
   [Include: feature_engineering_diagram.png]
   [Include: TF-IDF Formula]
   
   TF-IDF Features:
   • 10,000 most frequent features
   • N-gram range: (1, 3) - unigrams, bigrams, trigrams
   • English stopwords removed
   • Sublinear TF scaling
   
   TF-IDF Formula:
   TF-IDF(t,d) = TF(t,d) × IDF(t)
   where:
     TF(t,d) = (count of t in d) / (total terms in d)
     IDF(t) = log(N / df(t))
   
   Complexity Features (20 features):
   • Text statistics: word count, sentence count, avg word length
   • Lexical diversity: unique words / total words
   • Technical term density: technical terms / total words
   • Advanced term count
   • Question type indicators
   
   Feature Union:
   • Combines TF-IDF (10,000) + Complexity (20) = 10,020 features
   • StandardScaler applied to complexity features

3. MODEL ARCHITECTURE
   [Include: model_architecture_diagram.png]
   
   TF-IDF + Logistic Regression:
   • Feature Union: Combines TF-IDF + complexity features
   • Regularization: C=3.0
   • Class weights: 2x boost for Intermediate class
   • Max iterations: 3000
   
   DistilBERT:
   • Base model: distilbert-base-uncased
   • Fine-tuning on difficulty classification task
   • Weighted loss function for class imbalance

4. TRAINING STRATEGY
   [Include: Hyperparameters Table]
   
   Class Balancing:
   • Oversampling: 1.5x boost for Intermediate class
   • Class weights: 2x penalty for Intermediate (TF-IDF model)
   • SMOTE oversampling applied to training set
   
   Hyperparameters:
   [See table below]
```

---

## 📊 **HYPERPARAMETERS TABLE (Add to Methods Section)**

```
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
│ Early Stopping      │ -                │ Yes (patience=2) │
└─────────────────────┴──────────────────┴──────────────────┘
```

---

## 🎨 **VISUALS TO ADD**

### **1. Data Pipeline Diagram** ✅ GENERATED
- **File:** `results/poster_charts/data_pipeline_diagram.png`
- **Shows:** Raw files → Text extraction → Depth computation → Chunking → Relabeling → Split
- **Place:** After "DATA PREPROCESSING" heading

### **2. Model Architecture Diagram** ✅ GENERATED
- **File:** `results/poster_charts/model_architecture_diagram.png`
- **Shows:** Input → TF-IDF + Complexity → Feature Union → Logistic Regression → Output
- **Place:** After "MODEL ARCHITECTURE" heading

### **3. Feature Engineering Diagram** ✅ GENERATED
- **File:** `results/poster_charts/feature_engineering_diagram.png`
- **Shows:** Input text → TF-IDF features + Complexity features → Feature Union
- **Place:** After "FEATURE ENGINEERING" heading

### **4. TF-IDF Formula** (Text - Add manually)
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

where:
  TF(t,d) = (count of term t in document d) / (total terms in d)
  IDF(t) = log(total documents N / documents containing t)
```

---

## ✅ **WHAT TO CHANGE IN YOUR CURRENT METHODS SECTION**

### **Current Issues:**
1. ❌ Too text-heavy (all bullet points, no visuals)
2. ❌ Missing diagrams (no pipeline or architecture visualization)
3. ❌ No formulas (TF-IDF not shown)
4. ❌ Lacks specificity (missing hyperparameters)
5. ❌ No clear structure (could be organized better)

### **Recommended Changes:**

1. **ADD 3 DIAGRAMS** (High Priority)
   - Data pipeline diagram
   - Model architecture diagram
   - Feature engineering diagram
   - ✅ All generated and ready to use!

2. **ADD HYPERPARAMETERS TABLE** (High Priority)
   - Replace vague descriptions with specific numbers
   - Copy table from above

3. **ADD TF-IDF FORMULA** (High Priority)
   - Shows technical depth
   - Copy formula from above

4. **SPLIT INTO SUBSECTIONS** (Medium Priority)
   - Current: 5 bullet points
   - New: 4 clear subsections (Preprocessing, Features, Models, Training)

5. **MAKE TEXT MORE SPECIFIC** (Medium Priority)
   - Add numbers (10,000 features, 512 words, etc.)
   - Add library names (PyMuPDF, python-pptx)
   - Add specific hyperparameters

---

## 🎯 **QUICK ACTION ITEMS**

1. ✅ **Diagrams generated** - Use the 3 PNG files in `results/poster_charts/`
2. ⏳ **Add hyperparameters table** - Copy from above
3. ⏳ **Add TF-IDF formula** - Copy from above
4. ⏳ **Reorganize text** - Use Option 1 or Option 2 from above
5. ⏳ **Replace current Methods section** - Use improved version

---

## 💡 **PRO TIP**

**Current Methods section:** ~80% text, 20% visual
**Ideal Methods section:** ~50% text, 50% visual

The diagrams will make your Methods section much more engaging and easier to understand!

