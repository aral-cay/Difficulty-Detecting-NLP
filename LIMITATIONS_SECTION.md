# Limitations Section for Poster

## 📋 **OPTION 1: CONCISE VERSION (Recommended for Poster)**

```
LIMITATIONS

• Domain Specificity: Model trained on lecture content; performance drops 
  significantly on questions (45% vs 80% accuracy)

• Class Imbalance: Intermediate class underrepresented (12.3% of test set), 
  resulting in lower precision (53%) despite high recall (78.7%)

• Text Length Dependency: Features optimized for long-form text; struggles 
  with short questions due to sparse TF-IDF representations

• Dataset Size: Limited to ~10K training samples; may benefit from larger, 
  more diverse datasets

• Intermediate Class Precision: Lower precision (53%) indicates model 
  sometimes misclassifies Beginner/Advanced content as Intermediate
```

---

## 📋 **OPTION 2: DETAILED VERSION (If You Have More Space)**

```
LIMITATIONS

1. Domain Mismatch:
   • Model trained on lecture content (long-form, explanatory text)
   • Performance drops significantly on questions (45% vs 80% accuracy)
   • Highlights need for domain adaptation or retraining

2. Class Imbalance:
   • Intermediate class underrepresented (12.3% of test set)
   • Lower precision (53%) despite high recall (78.7%)
   • Requires aggressive balancing techniques (oversampling, weighted loss)

3. Text Length Dependency:
   • Features optimized for long-form text (500-1000 words)
   • Struggles with short questions (5-20 words)
   • TF-IDF features become sparse on short text

4. Dataset Limitations:
   • Limited to ~10K training samples
   • Domain-specific (NLP/ML topics only)
   • May not generalize to other educational domains

5. Model Performance:
   • Intermediate class precision needs improvement (53%)
   • DistilBERT underperforms TF-IDF (78.79% vs 80.12%)
   • Suggests dataset size may be insufficient for transformer models
```

---

## 📋 **OPTION 3: BULLET POINT VERSION (Most Compact)**

```
LIMITATIONS

• Domain mismatch: 45% accuracy on questions vs 80% on lecture content
• Class imbalance: Intermediate class (12.3%) has lower precision (53%)
• Text length dependency: Optimized for long-form text, struggles with short questions
• Dataset size: ~10K samples may limit generalization
• Intermediate precision: Model sometimes misclassifies Beginner/Advanced as Intermediate
```

---

## 📋 **OPTION 4: STRUCTURED VERSION (With Categories)**

```
LIMITATIONS

Data Limitations:
• Domain-specific training (NLP/ML lecture content)
• Class imbalance: Intermediate class only 12.3% of test set
• Limited dataset size (~10K training samples)

Model Limitations:
• Intermediate class precision: 53% (vs 78.7% recall)
• Text length dependency: Features optimized for long-form text
• Domain mismatch: 45% accuracy on questions vs 80% on lectures

Technical Challenges:
• TF-IDF features become sparse on short text
• DistilBERT underperforms TF-IDF (78.79% vs 80.12%)
• Feature engineering not optimized for question format
```

---

## 📋 **OPTION 5: BALANCED VERSION (Acknowledges Strengths Too)**

```
LIMITATIONS

While achieving 80.12% accuracy on lecture content, the model faces 
several limitations:

• Domain Specificity: Performance drops to 45% on questions, indicating 
  need for domain adaptation

• Class Imbalance: Intermediate class (12.3% of test set) shows lower 
  precision (53%) despite high recall (78.7%)

• Text Format Dependency: Features optimized for long-form text; struggles 
  with short questions due to sparse representations

• Dataset Scope: Limited to NLP/ML topics; generalization to other domains 
  requires further validation

• Model Comparison: DistilBERT (78.79%) underperforms TF-IDF (80.12%), 
  suggesting dataset size may be insufficient for transformer models
```

---

## ✅ **RECOMMENDED FOR POSTER**

### **Use Option 1 (Concise Version)**

**Why:**
- ✅ Concise (5 bullet points)
- ✅ Covers all key limitations
- ✅ Professional and honest
- ✅ Appropriate length for poster
- ✅ Easy to read

**Text to use:**

```
LIMITATIONS

• Domain Specificity: Model trained on lecture content; performance drops 
  significantly on questions (45% vs 80% accuracy)

• Class Imbalance: Intermediate class underrepresented (12.3% of test set), 
  resulting in lower precision (53%) despite high recall (78.7%)

• Text Length Dependency: Features optimized for long-form text; struggles 
  with short questions due to sparse TF-IDF representations

• Dataset Size: Limited to ~10K training samples; may benefit from larger, 
  more diverse datasets

• Intermediate Class Precision: Lower precision (53%) indicates model 
  sometimes misclassifies Beginner/Advanced content as Intermediate
```

---

## 🎯 **KEY LIMITATIONS TO HIGHLIGHT**

1. **Domain Mismatch** (Most Important)
   - 45% on questions vs 80% on lectures
   - Shows need for domain adaptation

2. **Class Imbalance**
   - Intermediate class only 12.3%
   - Lower precision (53%)

3. **Text Length Dependency**
   - Optimized for long-form text
   - Struggles with short questions

4. **Dataset Size**
   - ~10K training samples
   - May limit generalization

5. **Intermediate Precision**
   - 53% precision (vs 78.7% recall)
   - Trade-off from class balancing

---

## 💡 **TIPS FOR POSTER**

1. **Be Honest but Balanced**
   - Acknowledge limitations
   - But don't undermine your achievements
   - 80.12% is still good!

2. **Be Specific**
   - Include numbers (45% vs 80%, 53% precision)
   - Shows you understand the issues

3. **Be Concise**
   - 3-5 bullet points max
   - Keep it readable

4. **Connect to Future Work**
   - Limitations → Future Work
   - Shows you have solutions in mind

---

## 📝 **READY-TO-USE TEXT**

Copy this for your poster:

```
LIMITATIONS

• Domain Specificity: Model trained on lecture content; performance drops 
  significantly on questions (45% vs 80% accuracy)

• Class Imbalance: Intermediate class underrepresented (12.3% of test set), 
  resulting in lower precision (53%) despite high recall (78.7%)

• Text Length Dependency: Features optimized for long-form text; struggles 
  with short questions due to sparse TF-IDF representations

• Dataset Size: Limited to ~10K training samples; may benefit from larger, 
  more diverse datasets

• Intermediate Class Precision: Lower precision (53%) indicates model 
  sometimes misclassifies Beginner/Advanced content as Intermediate
```

This version is:
- ✅ Concise (5 bullet points)
- ✅ Specific (includes numbers)
- ✅ Professional
- ✅ Honest about challenges
- ✅ Appropriate for poster format

