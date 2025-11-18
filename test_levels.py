#!/usr/bin/env python3
"""
Quick test script with examples for each difficulty level.
Run: python3 test_levels.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.interactive_classify import predict_sklearn, print_prediction
import joblib

# Load model
print("Loading model...")
vectorizer = joblib.load("models/tfidf_logreg_3level/vectorizer.joblib")
classifier = joblib.load("models/tfidf_logreg_3level/classifier.joblib")

examples = {
    "Beginner (Level 1)": [
        "What is a variable in programming?",
        "What is machine learning?",
        "Explain Bayes theorem",
        "What is a neural network?",
        "What is natural language processing?",
        "What is supervised learning?",
        "What is an algorithm?",
        "What is recursion?"
    ],
    "Intermediate (Level 2)": [
        "How does entity extraction work?",
        "What is semantic parsing?",
        "How does cross-lingual processing work?",
        "What is smoothing in NLP?",
        "How does dependency parsing work?",
        "What is relation extraction?",
        "What is the difference between precision and recall?",
        "What is overfitting and how do you prevent it?",
        "How does cross-validation work?"
    ],
    "Advanced (Level 3)": [
        "What are deep learning methods used in adversarial search?",
        "Explain the theoretical foundations of biaffine attention mechanisms in neural parsing",
        "How do transformer models handle long-range dependencies?",
        "Explain the mathematical formulation of variational autoencoders",
        "How does self-attention mechanism work in Transformer architecture?",
        "What are the theoretical bounds for PAC learning?",
        "How does meta-learning work in few-shot learning?"
    ]
}

print("\n" + "=" * 80)
print("TESTING EXAMPLES FOR EACH DIFFICULTY LEVEL")
print("=" * 80)

for level_name, questions in examples.items():
    print(f"\n{'=' * 80}")
    print(f"{level_name}")
    print("=" * 80)
    
    correct = 0
    # Extract level number from level_name (e.g., "Beginner (Level 1)" -> 1)
    if "Level 1" in level_name:
        expected_level = 1
    elif "Level 2" in level_name:
        expected_level = 2
    elif "Level 3" in level_name:
        expected_level = 3
    else:
        expected_level = None
    
    for i, q in enumerate(questions, 1):
        pred, proba = predict_sklearn(q, vectorizer, classifier, use_threshold=True)
        
        # Check if correct
        is_correct = pred == expected_level
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        level_names = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
        
        print(f"\n{status} Example {i}: {q[:60]}{'...' if len(q) > 60 else ''}")
        print(f"   Predicted: Level {pred} ({level_names[pred]})")
        print(f"   Probabilities: B={proba[0]:.1%}, I={proba[1]:.1%}, A={proba[2]:.1%}")
    
    print(f"\n📊 Score: {correct}/{len(questions)} correct ({correct/len(questions)*100:.0f}%)")
    
    if level_name == "Intermediate":
        print(f"\n💡 Note: NLP-specific questions work best for Intermediate!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\n💡 Tips:")
print("   • Beginner: Use simple 'What is...' questions")
print("   • Intermediate: Use NLP-specific terms (parsing, extraction, semantic)")
print("   • Advanced: Include technical terms (biaffine, transformer, theoretical)")
print("=" * 80)

