# Formulas/Equations for Your Poster

## 🎯 **ESSENTIAL FORMULAS (MUST ADD)**

### **1. TF-IDF Formula** ⭐⭐⭐⭐⭐

**Location:** Methods section, near Feature Engineering

**Formula:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)

where:
  TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
  IDF(t) = log(N / df(t))
  
  N = Total number of documents
  df(t) = Number of documents containing term t
```

**Alternative (More Compact):**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)

TF(t, d) = count(t, d) / |d|
IDF(t) = log(N / df(t))
```

**Why Include:**
- ✅ Core to your TF-IDF model
- ✅ Shows technical understanding
- ✅ Standard in NLP research
- ✅ Easy to understand

---

### **2. Precision, Recall, F1-Score Formulas** ⭐⭐⭐⭐⭐

**Location:** Results section, near confusion matrix or per-class metrics

**Formulas:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

where:
  TP = True Positives
  FP = False Positives
  FN = False Negatives
```

**Alternative (More Compact):**
```
P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2PR / (P + R)
```

**Why Include:**
- ✅ Essential for understanding your results
- ✅ Standard evaluation metrics
- ✅ Referenced in your confusion matrix
- ✅ Shows you understand evaluation

---

## 📊 **RECOMMENDED FORMULAS (SHOULD ADD)**

### **3. Logistic Regression Formula** ⭐⭐⭐⭐

**Location:** Methods section, near Model Architecture

**Formula:**
```
P(y = k | x) = exp(z_k) / Σ exp(z_j)

where:
  z_k = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
  w = learned weights
  x = feature vector
```

**Alternative (Binary Classification):**
```
P(y = 1 | x) = 1 / (1 + exp(-z))

where:
  z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

**Why Include:**
- ✅ Shows your classifier model
- ✅ Demonstrates technical depth
- ✅ Standard ML formula
- ⚠️ Can be simplified if space is limited

---

### **4. Text Complexity Features** ⭐⭐⭐

**Location:** Methods section, near Feature Engineering

**Formulas:**
```
Lexical Diversity = Unique Words / Total Words
Technical Density = Technical Terms / Total Words
Advanced Density = Advanced Terms / Total Words
Average Word Length = Σ(word_length) / Word Count
Average Sentence Length = Word Count / Sentence Count
```

**Why Include:**
- ✅ Shows your feature engineering
- ✅ Explains complexity features
- ✅ Demonstrates domain knowledge
- ⚠️ Can be simplified to just the key ones

---

## 🔬 **OPTIONAL: DISTILBERT FORMULAS**

### **5. Self-Attention Mechanism** ⭐⭐ (Optional - May be too technical)

**Location:** Methods section, if you want to show DistilBERT details

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
  Q = Query matrix
  K = Key matrix
  V = Value matrix
  d_k = dimension of keys
```

**Why Include/Not Include:**
- ✅ Shows understanding of transformers
- ✅ Technical depth
- ⚠️ May be too complex for poster
- ⚠️ DistilBERT is pre-trained, so less critical
- ⚠️ Takes up space

**Recommendation:** Only include if you have space and want to show technical depth. Otherwise, just mention "DistilBERT uses self-attention mechanism" without the formula.

---

### **6. Cross-Entropy Loss** ⭐⭐ (Optional)

**Location:** Methods section, near training details

**Formula:**
```
L = -Σ y_i log(ŷ_i)

where:
  y_i = true label (one-hot encoded)
  ŷ_i = predicted probability
```

**Why Include/Not Include:**
- ✅ Shows loss function used
- ⚠️ May be too technical
- ⚠️ Standard in ML, not unique to your work

**Recommendation:** Only if you have space and want to show training details.

---

## 📋 **RECOMMENDED FORMULA SET FOR POSTER**

### **Minimum Set (Must Have):**

1. ✅ **TF-IDF Formula** - In Methods section
2. ✅ **Precision/Recall/F1 Formulas** - In Results section

### **Recommended Set (Should Have):**

1. ✅ **TF-IDF Formula** - In Methods section
2. ✅ **Precision/Recall/F1 Formulas** - In Results section
3. ✅ **Logistic Regression Formula** - In Methods section
4. ✅ **Text Complexity Features** (2-3 key ones) - In Methods section

### **Complete Set (If Space Allows):**

1. ✅ **TF-IDF Formula** - In Methods section
2. ✅ **Precision/Recall/F1 Formulas** - In Results section
3. ✅ **Logistic Regression Formula** - In Methods section
4. ✅ **Text Complexity Features** - In Methods section
5. ⚠️ **Self-Attention** (simplified) - In Methods section (optional)

---

## 📐 **FORMATTED VERSIONS FOR POSTER**

### **Version 1: Compact (Recommended)**

**For Methods Section:**
```
FEATURE ENGINEERING

TF-IDF: TF-IDF(t,d) = TF(t,d) × IDF(t)
  where TF(t,d) = count(t,d) / |d|
        IDF(t) = log(N / df(t))

Complexity Features:
  Lexical Diversity = Unique Words / Total Words
  Technical Density = Technical Terms / Total Words
```

**For Results Section:**
```
EVALUATION METRICS

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2PR / (P + R)
```

---

### **Version 2: Detailed**

**For Methods Section:**
```
FEATURE ENGINEERING

TF-IDF Formula:
TF-IDF(t, d) = TF(t, d) × IDF(t)

where:
  TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
  IDF(t) = log(Total documents / Number of documents containing term t)

Text Complexity Features:
• Lexical Diversity = Unique Words / Total Words
• Technical Density = Technical Terms / Total Words
• Advanced Density = Advanced Terms / Total Words
• Average Word Length = Σ(word_length) / Word Count
• Average Sentence Length = Word Count / Sentence Count

Model:
P(y = k | x) = exp(z_k) / Σ exp(z_j)
where z_k = w₀ + w₁x₁ + ... + wₙxₙ
```

**For Results Section:**
```
EVALUATION METRICS

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

where:
  TP = True Positives
  FP = False Positives
  FN = False Negatives
```

---

## 🎨 **VISUAL PRESENTATION TIPS**

1. **Use LaTeX-style formatting** if possible:
   - `TF-IDF(t, d) = TF(t, d) × IDF(t)`
   - Use proper subscripts/superscripts

2. **Box important formulas:**
   - Put formulas in a box or highlighted area
   - Makes them stand out

3. **Keep it readable:**
   - Use large enough font
   - Don't crowd formulas together
   - Add spacing between formulas

4. **Add context:**
   - Label what each formula represents
   - Explain variables briefly

---

## ✅ **FINAL RECOMMENDATIONS**

### **Must Include:**
1. ✅ **TF-IDF Formula** - Essential for your model
2. ✅ **Precision/Recall/F1 Formulas** - Essential for evaluation

### **Should Include:**
3. ✅ **Logistic Regression Formula** - Shows your classifier
4. ✅ **2-3 Complexity Feature Formulas** - Shows feature engineering

### **Optional:**
5. ⚠️ **Self-Attention Formula** - Only if space allows and you want technical depth
6. ⚠️ **Cross-Entropy Loss** - Only if discussing training details

### **Don't Include:**
- ❌ Full transformer architecture formulas (too complex)
- ❌ Backpropagation formulas (too detailed)
- ❌ Gradient descent formulas (not unique to your work)

---

## 📝 **READY-TO-USE TEXT**

### **For Methods Section:**

```
FEATURE ENGINEERING

TF-IDF Formula:
TF-IDF(t, d) = TF(t, d) × IDF(t)

where:
  TF(t, d) = count(t, d) / |d|
  IDF(t) = log(N / df(t))

Complexity Features:
• Lexical Diversity = Unique Words / Total Words
• Technical Density = Technical Terms / Total Words
```

### **For Results Section:**

```
EVALUATION METRICS

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2PR / (P + R)
```

---

## 💡 **SUMMARY**

**Essential (2 formulas):**
- TF-IDF formula
- Precision/Recall/F1 formulas

**Recommended (2-3 more):**
- Logistic Regression formula
- Complexity feature formulas

**Total: 4-5 formulas** is a good balance for a poster - shows technical depth without overwhelming the reader.

