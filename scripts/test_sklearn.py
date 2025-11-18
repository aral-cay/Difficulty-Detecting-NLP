#!/usr/bin/env python3
"""Simple command-line interface for testing the sklearn model."""
import sys
import joblib

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/test_sklearn.py 'your question here'")
        print("\nExample:")
        print("  python3 scripts/test_sklearn.py 'explain bayes theorem'")
        sys.exit(1)
    
    text = " ".join(sys.argv[1:])
    
    # Load model
    vectorizer = joblib.load("models/tfidf_logreg/vectorizer.joblib")
    classifier = joblib.load("models/tfidf_logreg/classifier.joblib")
    
    # Predict
    vec = vectorizer.transform([text])
    pred = int(classifier.predict(vec)[0])
    probs = classifier.predict_proba(vec)[0]
    
    # Output
    level_names = {1: "Introductory", 2: "Intermediate", 3: "Advanced"}
    
    print(f"\nInput: {text}")
    print(f"\nPredicted Difficulty Level: {pred} ({level_names[pred]})")
    print("\nProbabilities:")
    for i, prob in enumerate(probs, 1):
        print(f"  Level {i} ({level_names[i]}): {prob*100:.1f}%")

if __name__ == "__main__":
    main()

