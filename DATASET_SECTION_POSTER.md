# Dataset Section for Poster

## 📊 **DATASET SECTION (Ready to Use)**

### **Option 1: Concise Version (Recommended for Poster)**

```
DATASET

Source: Yale-Lily LectureBank
• 1,800+ lecture files across 174 topics
• Format: PDF and PPTX files
• Domain: Natural Language Processing & Machine Learning

Preprocessing Pipeline:
1. Text extraction from PDF/PPTX files
2. Depth computation from ConceptBank taxonomy
3. Text chunking for dataset expansion
4. Relabeling: 5-level → 3-level classification

Final Dataset:
• Training: 10,234 samples (after oversampling)
• Validation: 678 samples
• Test: 4,527 samples
• Total: 15,439 samples

Class Distribution (Test Set):
• Level 1 (Beginner): 2,246 samples (49.6%)
• Level 2 (Intermediate): 555 samples (12.3%)
• Level 3 (Advanced): 1,726 samples (38.1%)
```

---

### **Option 2: Detailed Version (If You Have More Space)**

```
DATASET

Source & Collection:
• Dataset: Yale-Lily LectureBank
• Content: Educational lecture materials
• Topics: 174 topics in NLP and ML
• Files: 5,079 downloaded files (PDF/PPTX)
• Domain: Natural Language Processing, Machine Learning

Data Preprocessing:
1. Text Extraction: Extract text from PDF and PPTX files
2. Depth Computation: Map topic IDs to difficulty levels using 
   ConceptBank taxonomy (5-level hierarchy)
3. Text Chunking: Split long texts into coherent segments 
   (preserves context while expanding dataset)
4. Relabeling: Convert 5-level taxonomy to 3-level classification
   (Beginner, Intermediate, Advanced)

Dataset Statistics:
• Original Samples: ~15,439 (after chunking)
• Training Set: 10,234 samples (after oversampling with SMOTE)
• Validation Set: 678 samples
• Test Set: 4,527 samples
• Split Ratio: 70% / 15% / 15%

Class Distribution (Test Set):
• Level 1 (Beginner): 2,246 samples (49.6%)
• Level 2 (Intermediate): 555 samples (12.3%)
• Level 3 (Advanced): 1,726 samples (38.1%)

Note: Class imbalance addressed through oversampling and 
weighted loss functions during training.
```

---

### **Option 3: Visual-Heavy Version (With Icons/Charts)**

```
DATASET

📚 Source: Yale-Lily LectureBank
   • 1,800+ lecture files
   • 174 topics (NLP & ML)
   • PDF/PPTX format

⚙️ Preprocessing:
   [Text Extraction] → [Depth Computation] → 
   [Chunking] → [Relabeling] → [Train/Val/Test Split]

📊 Dataset Size:
   Training: 10,234 samples
   Validation: 678 samples
   Test: 4,527 samples

📈 Class Distribution (Test Set):
   Level 1: 2,246 (49.6%) ████████████
   Level 2: 555 (12.3%)   ███
   Level 3: 1,726 (38.1%) █████████
```

---

## 📋 **WHAT TO PUT IN YOUR POSTER**

### **Essential Information (Must Include):**

1. **Dataset Source**
   - ✅ Yale-Lily LectureBank
   - ✅ Number of files/topics (if known)

2. **Preprocessing Steps**
   - ✅ Text extraction
   - ✅ Depth computation
   - ✅ Text chunking
   - ✅ Relabeling (5-level → 3-level)

3. **Dataset Statistics**
   - ✅ Training: 10,234 samples
   - ✅ Validation: 678 samples
   - ✅ Test: 4,527 samples
   - ✅ Total: 15,439 samples

4. **Class Distribution**
   - ✅ Level 1: 2,246 (49.6%)
   - ✅ Level 2: 555 (12.3%)
   - ✅ Level 3: 1,726 (38.1%)

### **Optional Information (If Space Permits):**

5. **Original Dataset Size** (before chunking/oversampling)
   - Original training: ~6,149 samples
   - After oversampling: 10,234 samples

6. **File Format Details**
   - PDF and PPTX files
   - Text extraction methods

7. **Domain Information**
   - NLP and Machine Learning topics
   - Educational lecture content

---

## 🎨 **VISUAL ELEMENTS TO ADD**

### **1. Class Distribution Chart** (Recommended)
Create a pie chart or bar chart showing:
- Level 1: 49.6%
- Level 2: 12.3%
- Level 3: 38.1%

**Why:** Shows class imbalance visually

### **2. Preprocessing Pipeline Diagram** (Optional)
```
Raw Files → Text Extraction → Depth Mapping → 
Chunking → Relabeling → Train/Val/Test Split
```

**Why:** Shows data flow clearly

### **3. Dataset Size Comparison** (Optional)
Bar chart showing:
- Original: ~6,149
- After Oversampling: 10,234

**Why:** Shows data augmentation impact

---

## 📝 **SAMPLE TEXT FOR DIFFERENT STYLES**

### **Academic Style:**
```
We utilize the Yale-Lily LectureBank dataset, comprising 
1,800+ lecture files across 174 topics in natural language 
processing and machine learning. The dataset undergoes 
extensive preprocessing including text extraction, depth 
computation from ConceptBank taxonomy, text chunking, and 
relabeling from 5-level to 3-level classification. Our 
final dataset contains 15,439 samples split into training 
(10,234), validation (678), and test (4,527) sets with a 
70/15/15 ratio.
```

### **Bullet Point Style (Recommended for Posters):**
```
• Source: Yale-Lily LectureBank (1,800+ files, 174 topics)
• Preprocessing: Extraction → Depth → Chunking → Relabeling
• Dataset: 15,439 samples (Train: 10,234, Val: 678, Test: 4,527)
• Distribution: Level 1 (49.6%), Level 2 (12.3%), Level 3 (38.1%)
```

### **Table Style:**
```
┌─────────────────────────────────────────────┐
│ Dataset Statistics                          │
├─────────────────────────────────────────────┤
│ Source          │ Yale-Lily LectureBank     │
│ Files           │ 1,800+ lecture files      │
│ Topics          │ 174 (NLP & ML)            │
│ Training        │ 10,234 samples            │
│ Validation      │ 678 samples               │
│ Test            │ 4,527 samples             │
│ Total           │ 15,439 samples            │
└─────────────────────────────────────────────┘
```

---

## ✅ **CHECKLIST FOR DATASET SECTION**

- [ ] Dataset source mentioned (Yale-Lily LectureBank)
- [ ] Number of files/topics included
- [ ] Preprocessing steps listed
- [ ] Dataset split sizes (Train/Val/Test)
- [ ] Class distribution shown
- [ ] Visual element (chart/diagram) added
- [ ] Numbers match your actual data
- [ ] Formatting is clear and readable

---

## 🎯 **RECOMMENDED LAYOUT**

```
┌─────────────────────────────────────┐
│         DATASET                     │
├─────────────────────────────────────┤
│ Source: Yale-Lily LectureBank       │
│ • 1,800+ files, 174 topics          │
│                                     │
│ Preprocessing:                      │
│ [Diagram or list]                   │
│                                     │
│ Dataset Size:                       │
│ • Train: 10,234                     │
│ • Val: 678                          │
│ • Test: 4,527                       │
│                                     │
│ [Class Distribution Chart]          │
│                                     │
└─────────────────────────────────────┘
```

---

## 💡 **PRO TIPS**

1. **Keep it concise** - Posters have limited space
2. **Use visuals** - Charts are more engaging than text
3. **Highlight imbalance** - Mention class imbalance and how you addressed it
4. **Be specific** - Use exact numbers, not approximations
5. **Show preprocessing** - Visual pipeline helps understanding

---

*Use Option 1 (Concise Version) for most posters. Add Option 2 details if you have extra space.*

