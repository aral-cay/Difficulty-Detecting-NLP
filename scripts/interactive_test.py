#!/usr/bin/env python3
"""Interactive script to test sklearn model with user input."""
import sys
import joblib
from pathlib import Path
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

# Import TextComplexityFeatures class needed for unpickling the model
class TextComplexityFeatures(BaseEstimator, TransformerMixin):
    """Extract comprehensive text complexity features."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_str = str(text)
            
            # Basic statistics
            char_count = len(text_str)
            word_count = len(text_str.split())
            sentence_count = len(re.split(r'[.!?]+', text_str))
            
            # Average lengths
            words = text_str.split()
            avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Lexical diversity
            words_lower = text_str.lower().split()
            unique_words = len(set(words_lower))
            lexical_diversity = unique_words / word_count if word_count > 0 else 0
            
            # Question indicators
            question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'explain', 'describe']
            question_count = sum(1 for word in question_words if word in text_str.lower())
            has_question = 1 if text_str.strip().endswith('?') else 0
            
            # Technical term indicators
            technical_terms = ['algorithm', 'function', 'method', 'model', 'neural', 'network', 
                             'learning', 'processing', 'extraction', 'parsing', 'semantic',
                             'transformer', 'attention', 'embedding', 'vector', 'matrix',
                             'optimization', 'gradient', 'backpropagation', 'activation']
            technical_count = sum(1 for term in technical_terms if term in text_str.lower())
            technical_density = technical_count / word_count if word_count > 0 else 0
            
            # Complexity indicators
            has_comparison = 1 if any(word in text_str.lower() for word in ['difference', 'compare', 'versus', 'vs', 'between', 'versus']) else 0
            has_explanation = 1 if any(phrase in text_str.lower() for phrase in ['explain', 'describe', 'how does', 'how do', 'how can']) else 0
            has_definition = 1 if any(phrase in text_str.lower() for phrase in ['what is', 'what are', 'define', 'definition']) else 0
            
            # Advanced terms
            advanced_terms = ['theoretical', 'formulation', 'architecture', 'mechanism', 'adversarial',
                            'variational', 'autoencoder', 'transformer', 'biaffine', 'meta-learning',
                            'reinforcement', 'generative', 'discriminative', 'probabilistic']
            advanced_count = sum(1 for term in advanced_terms if term in text_str.lower())
            advanced_density = advanced_count / word_count if word_count > 0 else 0
            
            # Readability-like features
            long_words = sum(1 for w in words if len(w) > 6)
            long_word_ratio = long_words / word_count if word_count > 0 else 0
            
            # Question type features
            starts_with_what = 1 if text_str.lower().strip().startswith('what') else 0
            starts_with_how = 1 if text_str.lower().strip().startswith('how') else 0
            starts_with_explain = 1 if text_str.lower().strip().startswith('explain') else 0
            
            feature_vector = [
                char_count,
                word_count,
                sentence_count,
                avg_word_length,
                avg_sentence_length,
                lexical_diversity,
                question_count,
                has_question,
                technical_count,
                technical_density,
                has_comparison,
                has_explanation,
                has_definition,
                advanced_count,
                advanced_density,
                long_words,
                long_word_ratio,
                starts_with_what,
                starts_with_how,
                starts_with_explain
            ]
            features.append(feature_vector)
        
        return np.array(features)

def predict_difficulty(text, use_threshold=True, use_improved=True):
    """Predict difficulty level using sklearn model with threshold tuning for Intermediate."""
    try:
        # Make sure TextComplexityFeatures is available in __main__ for unpickling
        import __main__
        if not hasattr(__main__, 'TextComplexityFeatures'):
            __main__.TextComplexityFeatures = TextComplexityFeatures
        
        # Try maximum model first, then improved, then original
        if Path("models/tfidf_logreg_3level_max/feature_union.joblib").exists():
            # Load maximum model with feature engineering
            feature_union = joblib.load("models/tfidf_logreg_3level_max/feature_union.joblib")
            classifier = joblib.load("models/tfidf_logreg_3level_max/classifier.joblib")
            
            # Transform with feature union (TF-IDF + complexity features)
            features = feature_union.transform([text])
            proba = classifier.predict_proba(features)[0]
            vec = features  # For prediction
        elif use_improved and Path("models/tfidf_logreg_3level_improved/feature_union.joblib").exists():
            # Load improved model with feature engineering
            feature_union = joblib.load("models/tfidf_logreg_3level_improved/feature_union.joblib")
            classifier = joblib.load("models/tfidf_logreg_3level_improved/classifier.joblib")
            
            # Transform with feature union (TF-IDF + complexity features)
            features = feature_union.transform([text])
            proba = classifier.predict_proba(features)[0]
            vec = features  # For prediction
        else:
            # Load original model
            vectorizer = joblib.load("models/tfidf_logreg_3level/vectorizer.joblib")
            classifier = joblib.load("models/tfidf_logreg_3level/classifier.joblib")
            
            # Predict
            vec = vectorizer.transform([text])
            proba = classifier.predict_proba(vec)[0]
        
        # Get class order from classifier (should be [1, 2, 3] for Level 1, 2, 3)
        classes = classifier.classes_
        # Find index of Intermediate (Level 2) in the classes array
        intermediate_class = 2
        if intermediate_class in classes:
            intermediate_idx = list(classes).index(intermediate_class)
        else:
            intermediate_idx = 1  # Fallback to index 1
        
        if use_threshold:
            # Threshold tuning: If Intermediate probability is close to max, boost it
            max_prob = proba.max()
            
            # If Intermediate is within 15% of max probability, prefer Intermediate
            if proba[intermediate_idx] >= max_prob * 0.85 and proba[intermediate_idx] >= 0.25:
                pred = intermediate_class  # Use Level 2 directly
            else:
                pred = int(classifier.predict(vec)[0])
        else:
            pred = int(classifier.predict(vec)[0])
        
        return pred, proba
    except Exception as e:
        import traceback
        print(f"DEBUG: Error in predict_difficulty: {e}")
        traceback.print_exc()
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
    
    # Check if model exists (try maximum first, then improved, then original)
    model_path = None
    if Path("models/tfidf_logreg_3level_max/classifier.joblib").exists():
        model_path = "models/tfidf_logreg_3level_max"
        print("Using MAXIMUM model (best accuracy: 81.50%)")
    elif Path("models/tfidf_logreg_3level_improved/classifier.joblib").exists():
        model_path = "models/tfidf_logreg_3level_improved"
        print("Using IMPROVED model (with feature engineering)")
    elif Path("models/tfidf_logreg_3level/classifier.joblib").exists():
        model_path = "models/tfidf_logreg_3level"
        print("Using original model")
    else:
        print("ERROR: Model not found!")
        print("Please train the model first:")
        print("  python scripts/train_sklearn_max.py --use_smote")
        print("  or")
        print("  python scripts/train_sklearn_improved.py")
        print("  or")
        print("  python scripts/train_sklearn.py")
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

