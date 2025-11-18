#!/usr/bin/env python3
"""Interactive script to test sklearn model with user input."""
import sys
import joblib
from pathlib import Path

def predict_difficulty(text, use_threshold=True):
    """Predict difficulty level using sklearn model with threshold tuning for Intermediate."""
    try:
        # Load model components
        vectorizer = joblib.load("models/tfidf_logreg_3level/vectorizer.joblib")
        classifier = joblib.load("models/tfidf_logreg_3level/classifier.joblib")
        
        # Predict
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
    except Exception as e:
        return None, None

def main():
    level_colors = {1: "\033[92m", 2: "\033[93m", 3: "\033[91m"}  # Green, Yellow, Red
    reset = "\033[0m"
    
    print("=" * 80)
    print("LECTURE DIFFICULTY PREDICTOR")
    print("=" * 80)
    print("\nThis model predicts difficulty on a 3-level scale:")
    print(f"  {level_colors[1]}Level 1: Introductory (basic concepts){reset}")
    print(f"  {level_colors[2]}Level 2: Intermediate (moderate complexity){reset}")
    print(f"  {level_colors[3]}Level 3: Advanced (complex topics){reset}")
    print("\n" + "=" * 80)
    
    # Check if model exists
    if not Path("models/tfidf_logreg_3level/classifier.joblib").exists():
        print("ERROR: Model not found!")
        print("Please train the model first: python3 scripts/train_sklearn.py")
        return
    
    # Check if stdin is interactive
    if not sys.stdin.isatty():
        print("ERROR: This script requires an interactive terminal.")
        print("Please run it directly (not piped or redirected).")
        return
    
    # Interactive loop
    print("\nEnter your questions (type 'quit' or 'exit' to stop):\n")
    
    while True:
        try:
            text = input("Question: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                print("Please enter a question.\n")
                continue
            
            pred, probs = predict_difficulty(text)
            
            if pred:
                level_names = {1: "Introductory", 2: "Intermediate", 3: "Advanced"}
                
                print("\n" + "=" * 80)
                print(f"Predicted Difficulty: {level_colors[pred]}Level {pred} ({level_names[pred]}){reset}")
                print("=" * 80)
                print("\nProbability Distribution:")
                for i, prob in enumerate(probs, 1):
                    level_name = level_names[i]
                    color = level_colors[i]
                    bar = "█" * int(prob * 50)
                    print(f"  {color}Level {i} ({level_name:12s}): {prob*100:5.1f}% {bar}{reset}")
                print()
            else:
                print("ERROR: Failed to predict. Please try again.\n")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nInput stream closed. Goodbye!")
            break

if __name__ == "__main__":
    main()

