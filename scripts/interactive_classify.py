#!/usr/bin/env python3
"""
Interactive interface for testing difficulty classification models.
Allows users to input questions/flashcard content and get predictions from both models.

Usage:
    python3 scripts/interactive_classify.py
    python3 scripts/interactive_classify.py --text "Your question here"
"""

import sys
import argparse
import os

# Add venv path if needed
venv_python = "/Users/aralcay/code/lecture-depth/venv/bin/python3"
if os.path.exists(venv_python) and not sys.executable.startswith(venv_python):
    print("⚠️  Warning: Not using virtual environment Python.")
    print(f"   Please use: {venv_python} scripts/interactive_classify.py")
    print("   Or activate venv first: source /Users/aralcay/code/lecture-depth/venv/bin/activate")
    print()

try:
    import joblib
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    print(f"❌ Error: Missing required package: {e}")
    print("\nPlease use the virtual environment Python:")
    print(f"  {venv_python} scripts/interactive_classify.py")
    print("\nOr activate the venv first:")
    print("  source /Users/aralcay/code/lecture-depth/venv/bin/activate")
    print("  python3 scripts/interactive_classify.py")
    sys.exit(1)

def predict_sklearn(text, vectorizer, classifier, use_threshold=True):
    """Predict difficulty using sklearn model with optional threshold tuning for Intermediate."""
    vec = vectorizer.transform([text])
    proba = classifier.predict_proba(vec)[0]
    
    if use_threshold:
        # Threshold tuning: If Intermediate probability is close to max, boost it
        max_prob = proba.max()
        intermediate_idx = 1  # Index 1 is Intermediate (0-indexed: 0=Beginner, 1=Intermediate, 2=Advanced)
        
        # If Intermediate is within 15% of max probability, prefer Intermediate
        if proba[intermediate_idx] >= max_prob * 0.85 and proba[intermediate_idx] >= 0.25:
            pred = intermediate_idx + 1  # Convert to 1-indexed
        else:
            pred = int(classifier.predict(vec)[0])
    else:
        pred = int(classifier.predict(vec)[0])
    
    return pred, proba

def predict_hf(text, tokenizer, model, max_length=256):
    """Predict difficulty using HF transformer model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()
        pred = pred_idx + 1  # Convert 0-2 to 1-3
        probabilities = torch.softmax(logits, dim=1)[0]
    return pred, probabilities.numpy()

def print_prediction(level, probabilities, model_name):
    """Print formatted prediction results."""
    level_names = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    level_colors = {1: "\033[92m", 2: "\033[93m", 3: "\033[91m"}  # Green, Yellow, Red
    reset = "\033[0m"
    
    print(f"\n{model_name}:")
    print(f"  Predicted Level: {level_colors[level]}{level} ({level_names[level]}){reset}")
    print(f"  Probabilities:")
    for i, prob in enumerate(probabilities):
        level_num = i + 1
        bar_length = int(prob * 30)
        bar = "█" * bar_length
        print(f"    Level {level_num} ({level_names[level_num]:12}): {prob:.4f} {bar}")

def main():
    ap = argparse.ArgumentParser(description="Interactive difficulty classification interface")
    ap.add_argument("--sklearn_model", default="models/tfidf_logreg_3level", 
                    help="Path to sklearn model directory")
    ap.add_argument("--hf_model", default="models/distilbert_depth3/best",
                    help="Path to HF model directory")
    ap.add_argument("--text", default=None, help="Text to classify (if not provided, interactive mode)")
    args = ap.parse_args()
    
    # Load sklearn model
    print("Loading sklearn model...")
    try:
        vectorizer = joblib.load(os.path.join(args.sklearn_model, 'vectorizer.joblib'))
        classifier = joblib.load(os.path.join(args.sklearn_model, 'classifier.joblib'))
        sklearn_loaded = True
        print("✓ Sklearn model loaded")
    except Exception as e:
        print(f"✗ Failed to load sklearn model: {e}")
        sklearn_loaded = False
    
    # Load HF model
    print("Loading HF transformer model...")
    try:
        if os.path.isdir(args.hf_model):
            tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
            model = AutoModelForSequenceClassification.from_pretrained(args.hf_model)
            hf_loaded = True
            print("✓ HF transformer model loaded")
        else:
            print(f"✗ HF model directory not found: {args.hf_model}")
            hf_loaded = False
    except Exception as e:
        print(f"✗ Failed to load HF model: {e}")
        hf_loaded = False
    
    if not sklearn_loaded and not hf_loaded:
        print("\nError: No models available!")
        sys.exit(1)
    
    # Interactive mode
    if args.text:
        # Single text mode
        text = args.text
        print("\n" + "="*80)
        print(f"Analyzing: {text[:100]}{'...' if len(text) > 100 else ''}")
        print("="*80)
        
        if sklearn_loaded:
            try:
                sk_pred, sk_proba = predict_sklearn(text, vectorizer, classifier)
                print_prediction(sk_pred, sk_proba, "Sklearn (TF-IDF + Logistic Regression)")
            except Exception as e:
                print(f"✗ Sklearn prediction failed: {e}")
        
        if hf_loaded:
            try:
                hf_pred, hf_proba = predict_hf(text, tokenizer, model)
                print_prediction(hf_pred, hf_proba, "HF Transformer (DistilBERT)")
            except Exception as e:
                print(f"✗ HF prediction failed: {e}")
        
        print("\n" + "="*80)
    else:
        # Interactive loop mode
        print("\n" + "="*80)
        print("Difficulty Classification Interface")
        print("="*80)
        print("Enter questions/flashcard content to classify difficulty level.")
        print("Type 'quit', 'exit', or 'q' to stop.\n")
        
        while True:
            try:
                text = input("Enter text to classify: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                if not text:
                    print("Please enter some text.")
                    continue
                
                # Classify immediately
                print("\n" + "="*80)
                print(f"Analyzing: {text[:100]}{'...' if len(text) > 100 else ''}")
                print("="*80)
                
                if sklearn_loaded:
                    try:
                        sk_pred, sk_proba = predict_sklearn(text, vectorizer, classifier)
                        print_prediction(sk_pred, sk_proba, "Sklearn (TF-IDF + Logistic Regression)")
                    except Exception as e:
                        print(f"✗ Sklearn prediction failed: {e}")
                
                if hf_loaded:
                    try:
                        hf_pred, hf_proba = predict_hf(text, tokenizer, model)
                        print_prediction(hf_pred, hf_proba, "HF Transformer (DistilBERT)")
                    except Exception as e:
                        print(f"✗ HF prediction failed: {e}")
                
                print("\n" + "="*80)
                print()  # Extra blank line for readability
                
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

if __name__ == "__main__":
    main()

