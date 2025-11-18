#!/usr/bin/env python3
"""Test both models with example questions."""
import sys
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_sklearn(text):
    """Predict using sklearn model."""
    vectorizer = joblib.load("models/tfidf_logreg/vectorizer.joblib")
    classifier = joblib.load("models/tfidf_logreg/classifier.joblib")
    vec = vectorizer.transform([text])
    pred = int(classifier.predict(vec)[0])
    return pred

def predict_hf(text):
    """Predict using HF transformer model."""
    tokenizer = AutoTokenizer.from_pretrained("models/distilbert_depth/best")
    model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_depth/best")
    model.eval()
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_level = predicted_class_idx + 1  # Convert 0-2 to 1-3
        probabilities = torch.softmax(logits, dim=1)[0]
    return predicted_level, probabilities

if __name__ == "__main__":
    questions = [
        "What are deep learning methods used in adversarial search?",
        "explain bayes theorem",
        "Introduction to machine learning basics",
        "Neural networks with attention mechanisms and transformers"
    ]
    
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    
    print("=" * 80)
    print("DIFFICULTY PREDICTION (3 Levels: 1=Introductory, 2=Intermediate, 3=Advanced)")
    print("=" * 80)
    
    for q in questions:
        print(f"\n📝 Question: {q}")
        print("-" * 80)
        
        # Sklearn prediction
        sk_pred = predict_sklearn(q)
        print(f"Sklearn Model: Level {sk_pred} ({'Introductory' if sk_pred==1 else 'Intermediate' if sk_pred==2 else 'Advanced'})")
        
        # HF prediction
        try:
            hf_pred, probs = predict_hf(q)
            print(f"HF Model:      Level {hf_pred} ({'Introductory' if hf_pred==1 else 'Intermediate' if hf_pred==2 else 'Advanced'})")
            print(f"  Probabilities: L1={probs[0]:.3f}, L2={probs[1]:.3f}, L3={probs[2]:.3f}")
        except Exception as e:
            print(f"HF Model:      Error - {e}")

