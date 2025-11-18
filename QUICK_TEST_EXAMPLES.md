# Quick Test Examples for Each Level

## 🟢 Beginner (Level 1) - Best Examples

```bash
python3 scripts/interactive_classify.py --text "What is machine learning?"
python3 scripts/interactive_classify.py --text "Explain Bayes theorem"
python3 scripts/interactive_classify.py --text "What is supervised learning?"
python3 scripts/interactive_classify.py --text "What is an algorithm?"
```

**Expected**: Level 1 (Beginner) ✅

---

## 🟡 Intermediate (Level 2) - Best Examples

### NLP-Specific (100% Accurate):
```bash
python3 scripts/interactive_classify.py --text "How does entity extraction work?"
python3 scripts/interactive_classify.py --text "What is semantic parsing?"
python3 scripts/interactive_classify.py --text "How does cross-lingual processing work?"
python3 scripts/interactive_classify.py --text "What is smoothing in NLP?"
python3 scripts/interactive_classify.py --text "How does dependency parsing work?"
python3 scripts/interactive_classify.py --text "What is relation extraction?"
```

**Expected**: Level 2 (Intermediate) ✅  
**Confidence**: 90%+ for NLP-specific questions

---

## 🔴 Advanced (Level 3) - Best Examples

```bash
python3 scripts/interactive_classify.py --text "What are deep learning methods used in adversarial search?"
python3 scripts/interactive_classify.py --text "Explain the mathematical formulation of variational autoencoders"
python3 scripts/interactive_classify.py --text "How does self-attention mechanism work in Transformer architecture?"
```

**Expected**: Level 3 (Advanced) ✅

---

## 🚀 Quick Test All Levels

Run the test script:
```bash
python3 test_levels.py
```

This will test all examples and show accuracy for each level.

---

## 📊 Current Test Results

- **Beginner**: 50% accuracy (some misclassified as Intermediate/Advanced)
- **Intermediate**: 89% accuracy (NLP-specific questions work great!)
- **Advanced**: Works well for complex technical questions

---

## 💡 Best Practices

1. **For Intermediate**: Use NLP-specific terms like:
   - "entity extraction"
   - "semantic parsing"
   - "cross-lingual processing"
   - "dependency parsing"
   - "relation extraction"

2. **For Beginner**: Use simple "What is..." questions

3. **For Advanced**: Include technical terms and theoretical concepts

