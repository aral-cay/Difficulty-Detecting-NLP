# Difficulty-Detecting-NLP

A machine learning system for classifying lecture content difficulty levels using NLP techniques.

## Overview

This project classifies educational content into three difficulty levels:
- **Level 1 (Introductory)**: Basic concepts and foundational knowledge
- **Level 2 (Intermediate)**: Applied concepts and moderate complexity
- **Level 3 (Advanced)**: Complex topics requiring deep understanding

## Features

- **TF-IDF + Logistic Regression** model for fast classification
- **DistilBERT** transformer model for advanced classification
- Interactive testing interface
- Threshold tuning for improved intermediate classification

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
# Or with conda
conda install joblib scikit-learn
```

### Interactive Testing

```bash
# Activate conda environment (if using conda)
conda activate base

# Run interactive test
python scripts/interactive_test.py
```

### Example Usage

```python
# The script will prompt for questions
Question: What is machine learning?
# Returns: Level 1 (Introductory) with probability distribution
```

## Project Structure

```
├── scripts/              # Python scripts
│   ├── interactive_test.py    # Interactive classification interface
│   ├── train_sklearn.py       # Train TF-IDF + Logistic Regression model
│   ├── train_hf_3levels.py    # Train DistilBERT model
│   └── ...
├── models/              # Trained models (not in git)
├── data/                # Dataset files (not in git)
├── results/             # Evaluation results
└── requirements.txt     # Python dependencies
```

## Documentation

- `QUICK_TEST_EXAMPLES.md` - Quick test examples for each level
- `DIFFICULTY_EXAMPLES.md` - Comprehensive examples and guidelines
- `INTERACTIVE_CLASSIFY.md` - Interactive classification guide
- `MODEL_PERFORMANCE.md` - Model performance metrics

## Models

The project uses two models:
1. **Sklearn (TF-IDF + Logistic Regression)**: Fast, interpretable baseline
2. **DistilBERT**: Transformer-based model for improved accuracy

Models are trained on the LectureBank dataset and saved in `models/` directory.

## License

See LICENSE file for details.

