#!/usr/bin/env python3
"""Evaluate model performance on ChatGPT-generated test questions."""
import sys
import joblib
from pathlib import Path
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd

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

def load_questions(file_path):
    """Load questions from a file, splitting by empty lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out empty lines and strip whitespace
        questions = [line.strip() for line in lines if line.strip()]
        
        return questions
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}")
        return []

def predict_difficulty(text, feature_union, classifier):
    """Predict difficulty level using sklearn model."""
    try:
        # Transform with feature union (TF-IDF + complexity features)
        features = feature_union.transform([text])
        proba = classifier.predict_proba(features)[0]
        pred = int(classifier.predict(features)[0])
        return pred, proba
    except Exception as e:
        print(f"Error predicting: {e}")
        return None, None

def main():
    print("=" * 80)
    print("EVALUATING MODEL ON CHATGPT TEST QUESTIONS")
    print("=" * 80)
    
    # Load model
    model_path = Path("models/tfidf_logreg_3level_max")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first: python scripts/train_sklearn_max.py --use_smote")
        sys.exit(1)
    
    print("\nLoading model...")
    import __main__
    if not hasattr(__main__, 'TextComplexityFeatures'):
        __main__.TextComplexityFeatures = TextComplexityFeatures
    
    feature_union = joblib.load(model_path / "feature_union.joblib")
    classifier = joblib.load(model_path / "classifier.joblib")
    print("✓ Model loaded successfully")
    
    # Load test questions
    test_dir = Path("chatGptTest")
    if not test_dir.exists():
        print(f"ERROR: Test directory not found at {test_dir}")
        sys.exit(1)
    
    print("\nLoading test questions...")
    easy_questions = load_questions(test_dir / "Easy")
    medium_questions = load_questions(test_dir / "Medium")
    hard_questions = load_questions(test_dir / "Hard")
    
    print(f"  Easy:   {len(easy_questions)} questions")
    print(f"  Medium: {len(medium_questions)} questions")
    print(f"  Hard:   {len(hard_questions)} questions")
    
    # Check if we have questions
    if len(easy_questions) == 0 and len(medium_questions) == 0 and len(hard_questions) == 0:
        print("\nERROR: No questions found in any test file!")
        print("Please make sure the Easy, Medium, and Hard files are saved and contain questions.")
        sys.exit(1)
    
    if len(easy_questions) == 0:
        print("\nWARNING: Easy file is empty. Evaluation will only include Medium and Hard.")
    if len(hard_questions) == 0:
        print("\nWARNING: Hard file is empty. Evaluation will only include Easy and Medium.")
    
    # Prepare data: Easy -> Level 1, Medium -> Level 2, Hard -> Level 3
    all_questions = easy_questions + medium_questions + hard_questions
    true_labels = [1] * len(easy_questions) + [2] * len(medium_questions) + [3] * len(hard_questions)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    probabilities = []
    
    for i, question in enumerate(all_questions):
        pred, proba = predict_difficulty(question, feature_union, classifier)
        if pred is not None:
            predictions.append(pred)
            probabilities.append(proba)
        else:
            print(f"Warning: Failed to predict question {i+1}")
            predictions.append(2)  # Default to Intermediate
            probabilities.append([0.33, 0.34, 0.33])
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(all_questions)} questions...")
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    
    # Overall accuracy
    overall_accuracy = accuracy_score(true_labels, predictions)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=[1, 2, 3], average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, labels=[1, 2, 3], average='weighted', zero_division=0
    )
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    print(f"{'Level 1 (Easy)':<15} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1[0]:<12.4f} {support[0]:<10}")
    print(f"{'Level 2 (Medium)':<15} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1[1]:<12.4f} {support[1]:<10}")
    print(f"{'Level 3 (Hard)':<15} {precision[2]:<12.4f} {recall[2]:<12.4f} {f1[2]:<12.4f} {support[2]:<10}")
    print("-" * 80)
    print(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {len(true_labels):<10}")
    print(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {len(true_labels):<10}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(true_labels, predictions, labels=[1, 2, 3])
    cm_df = pd.DataFrame(cm, 
                        index=['True Level 1', 'True Level 2', 'True Level 3'],
                        columns=['Pred Level 1', 'Pred Level 2', 'Pred Level 3'])
    print(cm_df)
    
    # Per-category accuracy
    print("\nPer-Category Accuracy:")
    print("-" * 80)
    easy_mask = true_labels == 1
    medium_mask = true_labels == 2
    hard_mask = true_labels == 3
    
    easy_acc = accuracy_score(true_labels[easy_mask], predictions[easy_mask])
    medium_acc = accuracy_score(true_labels[medium_mask], predictions[medium_mask])
    hard_acc = accuracy_score(true_labels[hard_mask], predictions[hard_mask])
    
    print(f"Easy (Level 1):   {easy_acc:.4f} ({easy_acc*100:.2f}%)")
    print(f"Medium (Level 2): {medium_acc:.4f} ({medium_acc*100:.2f}%)")
    print(f"Hard (Level 3):   {hard_acc:.4f} ({hard_acc*100:.2f}%)")
    
    # Detailed classification report
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(true_labels, predictions, 
                               target_names=['Level 1 (Easy)', 'Level 2 (Medium)', 'Level 3 (Hard)'],
                               labels=[1, 2, 3]))
    
    # Save results to file
    results_file = Path("results/chatgpt_test_evaluation.txt")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CHATGPT TEST EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Level 1 (Easy)':<15} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1[0]:<12.4f} {support[0]:<10}\n")
        f.write(f"{'Level 2 (Medium)':<15} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1[1]:<12.4f} {support[1]:<10}\n")
        f.write(f"{'Level 3 (Hard)':<15} {precision[2]:<12.4f} {recall[2]:<12.4f} {f1[2]:<12.4f} {support[2]:<10}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {len(true_labels):<10}\n")
        f.write(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {len(true_labels):<10}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 80 + "\n")
        f.write(str(cm_df) + "\n\n")
        f.write("Per-Category Accuracy:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Easy (Level 1):   {easy_acc:.4f} ({easy_acc*100:.2f}%)\n")
        f.write(f"Medium (Level 2): {medium_acc:.4f} ({medium_acc*100:.2f}%)\n")
        f.write(f"Hard (Level 3):   {hard_acc:.4f} ({hard_acc*100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(true_labels, predictions, 
                                     target_names=['Level 1 (Easy)', 'Level 2 (Medium)', 'Level 3 (Hard)'],
                                     labels=[1, 2, 3]))
    
    print(f"\n✓ Results saved to {results_file}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

