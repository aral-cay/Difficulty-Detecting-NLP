# Interactive Difficulty Classification Interface

## Quick Start

Run the interactive interface:
```bash
python3 scripts/interactive_classify.py
```

Or classify a single text:
```bash
python3 scripts/interactive_classify.py --text "Your question here"
```

## Features

- **Dual Model Support**: Gets predictions from both Sklearn (TF-IDF + Logistic Regression) and HF Transformer (DistilBERT) models
- **Interactive Mode**: Keep entering questions without restarting
- **Visual Output**: Color-coded difficulty levels with probability bars
- **Graceful Fallback**: Works even if one model isn't available yet

## Example Usage

```bash
$ python3 scripts/interactive_classify.py

Enter text to classify: What are deep learning methods used in adversarial search?
Analyzing: What are deep learning methods used in adversarial search?
================================================================================
Sklearn (TF-IDF + Logistic Regression):
  Predicted Level: 3 (Advanced)
  Probabilities:
    Level 1 (Beginner    ): 0.0216 
    Level 2 (Intermediate): 0.0246 
    Level 3 (Advanced    ): 0.9538 ████████████████████████████

HF Transformer (DistilBERT):
  Predicted Level: 3 (Advanced)
  Probabilities:
    Level 1 (Beginner    ): 0.0234 
    Level 2 (Intermediate): 0.0156 
    Level 3 (Advanced    ): 0.9610 ████████████████████████████

Enter text to classify: Explain Bayes theorem
...
```

## Color Coding

- 🟢 Green: Beginner (Level 1)
- 🟡 Yellow: Intermediate (Level 2)
- 🔴 Red: Advanced (Level 3)

## Exit

Type `quit`, `exit`, or `q` to stop the interactive session.

