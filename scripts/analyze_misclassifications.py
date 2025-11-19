#!/usr/bin/env python3
"""Analyze misclassifications and provide explanations for confusion matrix off-diagonal values."""
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

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

def analyze_text_features(text):
    """Extract key features from text for analysis."""
    text_str = str(text)
    words = text_str.split()
    word_count = len(words)
    
    # Advanced terms
    advanced_terms = ['theoretical', 'formulation', 'architecture', 'mechanism', 'adversarial',
                     'variational', 'autoencoder', 'transformer', 'biaffine', 'meta-learning',
                     'reinforcement', 'generative', 'discriminative', 'probabilistic', 'gradient',
                     'optimization', 'backpropagation', 'neural', 'network']
    advanced_count = sum(1 for term in advanced_terms if term in text_str.lower())
    
    # Basic terms
    basic_terms = ['what', 'is', 'are', 'the', 'a', 'an', 'define', 'definition']
    basic_count = sum(1 for term in basic_terms if term in text_str.lower())
    
    # Technical complexity
    technical_terms = ['algorithm', 'function', 'method', 'model', 'processing', 'extraction']
    technical_count = sum(1 for term in technical_terms if term in text_str.lower())
    
    return {
        'word_count': word_count,
        'advanced_terms': advanced_count,
        'basic_terms': basic_count,
        'technical_terms': technical_count,
        'avg_word_length': np.mean([len(w) for w in words]) if word_count > 0 else 0,
        'has_question': 1 if text_str.strip().endswith('?') else 0,
        'starts_with_what': 1 if text_str.lower().strip().startswith('what') else 0,
        'starts_with_how': 1 if text_str.lower().strip().startswith('how') else 0,
    }

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Analyze misclassifications and explain confusion matrix")
    ap.add_argument("--test", default="data/processed/lecture_depth3_test.csv", 
                    help="Path to test CSV file")
    ap.add_argument("--model", default="models/tfidf_logreg_3level_max",
                    help="Path to model directory")
    ap.add_argument("--output", default="results/misclassification_analysis",
                    help="Output prefix for saving results")
    ap.add_argument("--num_examples", type=int, default=5,
                    help="Number of example misclassifications to show per category")
    args = ap.parse_args()
    
    print("=" * 80)
    print("ANALYZING MISCLASSIFICATIONS")
    print("=" * 80)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    print(f"\nLoading model from {model_path}...")
    import __main__
    if not hasattr(__main__, 'TextComplexityFeatures'):
        __main__.TextComplexityFeatures = TextComplexityFeatures
    
    feature_union = joblib.load(model_path / "feature_union.joblib")
    classifier = joblib.load(model_path / "classifier.joblib")
    print("✓ Model loaded successfully")
    
    # Load test set
    test_path = Path(args.test)
    if not test_path.exists():
        print(f"ERROR: Test file not found at {test_path}")
        sys.exit(1)
    
    print(f"\nLoading test set from {test_path}...")
    test_df = pd.read_csv(test_path)
    print(f"  Loaded {len(test_df)} samples")
    
    # Prepare data
    X_test = test_df['text'].fillna('').astype(str)
    y_true = test_df['depth_level'].astype(int)
    
    # Make predictions
    print("\nMaking predictions...")
    X_test_features = feature_union.transform(X_test)
    y_pred = classifier.predict(X_test_features)
    y_pred = y_pred.astype(int)
    
    # Get prediction probabilities
    y_proba = classifier.predict_proba(X_test_features)
    
    # Identify misclassifications
    misclassified = test_df.copy()
    misclassified['predicted'] = y_pred
    misclassified['true_label'] = y_true
    misclassified['is_correct'] = (y_pred == y_true)
    misclassified['confidence'] = np.max(y_proba, axis=1)
    misclassified['predicted_proba'] = [y_proba[i, y_pred[i]-1] for i in range(len(y_pred))]
    
    # Group misclassifications by type
    misclass_groups = defaultdict(list)
    
    for idx, row in misclassified.iterrows():
        if not row['is_correct']:
            true_lvl = int(row['true_label'])
            pred_lvl = int(row['predicted'])
            key = f"True_{true_lvl}_Pred_{pred_lvl}"
            misclass_groups[key].append({
                'text': row['text'],
                'true': true_lvl,
                'pred': pred_lvl,
                'confidence': row['confidence'],
                'proba': row['predicted_proba'],
                'proba_dist': y_proba[idx]
            })
    
    # Generate analysis report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    txt_file = output_path.with_suffix('.txt')
    
    level_names = {1: "Level 1 (Beginner)", 2: "Level 2 (Intermediate)", 3: "Level 3 (Advanced)"}
    
    with open(txt_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MISCLASSIFICATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Set: {args.test}\n")
        f.write(f"Total Misclassifications: {len(misclassified[~misclassified['is_correct']])}\n")
        f.write(f"Total Samples: {len(test_df)}\n")
        f.write(f"Accuracy: {(misclassified['is_correct'].sum() / len(misclassified) * 100):.2f}%\n\n")
        
        # Analyze each misclassification type
        misclass_types = [
            ("True_1_Pred_2", "Level 1 (Beginner) → Predicted as Level 2 (Intermediate)"),
            ("True_1_Pred_3", "Level 1 (Beginner) → Predicted as Level 3 (Advanced)"),
            ("True_2_Pred_1", "Level 2 (Intermediate) → Predicted as Level 1 (Beginner)"),
            ("True_2_Pred_3", "Level 2 (Intermediate) → Predicted as Level 3 (Advanced)"),
            ("True_3_Pred_1", "Level 3 (Advanced) → Predicted as Level 1 (Beginner)"),
            ("True_3_Pred_2", "Level 3 (Advanced) → Predicted as Level 2 (Intermediate)"),
        ]
        
        for key, description in misclass_types:
            if key in misclass_groups:
                examples = misclass_groups[key]
                examples.sort(key=lambda x: x['confidence'], reverse=True)  # Sort by confidence
                
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"{description}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Count: {len(examples)}\n\n")
                
                # Calculate average features for this misclassification type
                if examples:
                    avg_features = defaultdict(float)
                    for ex in examples:
                        features = analyze_text_features(ex['text'])
                        for feat_name, feat_value in features.items():
                            avg_features[feat_name] += feat_value
                    for feat_name in avg_features:
                        avg_features[feat_name] /= len(examples)
                    
                    f.write("Average Text Features:\n")
                    f.write(f"  Word Count: {avg_features['word_count']:.1f}\n")
                    f.write(f"  Advanced Terms: {avg_features['advanced_terms']:.2f}\n")
                    f.write(f"  Technical Terms: {avg_features['technical_terms']:.2f}\n")
                    f.write(f"  Average Word Length: {avg_features['avg_word_length']:.2f}\n")
                    f.write(f"  Starts with 'What': {avg_features['starts_with_what']:.2f}\n")
                    f.write(f"  Starts with 'How': {avg_features['starts_with_how']:.2f}\n\n")
                
                # Provide explanation
                true_lvl = int(key.split('_')[1])
                pred_lvl = int(key.split('_')[3])
                
                explanations = {
                    ("True_1_Pred_2",): "These Beginner texts were misclassified as Intermediate. "
                                      "Likely reasons: They contain intermediate-level terminology or concepts, "
                                      "or have complexity features (word count, technical terms) that push them toward Intermediate.",
                    ("True_1_Pred_3",): "These Beginner texts were misclassified as Advanced. "
                                      "Likely reasons: They mention advanced concepts or use technical terminology "
                                      "that the model associates with Advanced level, even if the explanation is basic.",
                    ("True_2_Pred_1",): "These Intermediate texts were misclassified as Beginner. "
                                      "Likely reasons: They use simpler language or basic terminology, "
                                      "or lack the complexity indicators the model expects for Intermediate content.",
                    ("True_2_Pred_3",): "These Intermediate texts were misclassified as Advanced. "
                                      "Likely reasons: They contain advanced terminology or concepts "
                                      "that push them toward Advanced, even if the overall complexity is intermediate.",
                    ("True_3_Pred_1",): "These Advanced texts were misclassified as Beginner. "
                                      "Likely reasons: They use simpler language structure or basic terminology "
                                      "despite covering advanced topics, or the model focuses on surface-level features.",
                    ("True_3_Pred_2",): "These Advanced texts were misclassified as Intermediate. "
                                      "Likely reasons: They lack the extreme complexity indicators (very long text, "
                                      "many advanced terms) that the model uses to identify Advanced content.",
                }
                
                for keys, explanation in explanations.items():
                    if key in keys:
                        f.write(f"Explanation: {explanation}\n\n")
                        break
                
                # Show examples
                f.write(f"Example Misclassifications (showing top {min(args.num_examples, len(examples))}):\n")
                f.write("-" * 80 + "\n")
                for i, ex in enumerate(examples[:args.num_examples], 1):
                    text_preview = ex['text'][:200] + "..." if len(ex['text']) > 200 else ex['text']
                    f.write(f"\nExample {i}:\n")
                    f.write(f"  Text: {text_preview}\n")
                    f.write(f"  Confidence: {ex['confidence']:.3f}\n")
                    f.write(f"  Probability Distribution: [Level 1: {ex['proba_dist'][0]:.3f}, "
                           f"Level 2: {ex['proba_dist'][1]:.3f}, Level 3: {ex['proba_dist'][2]:.3f}]\n")
                
                f.write("\n")
        
        # Summary statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        for key, description in misclass_types:
            if key in misclass_groups:
                examples = misclass_groups[key]
                avg_conf = np.mean([ex['confidence'] for ex in examples])
                f.write(f"{description}: {len(examples)} cases, avg confidence: {avg_conf:.3f}\n")
    
    print(f"\n✓ Analysis saved to {txt_file}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("MISCLASSIFICATION SUMMARY")
    print("=" * 80)
    for key, description in misclass_types:
        if key in misclass_groups:
            examples = misclass_groups[key]
            avg_conf = np.mean([ex['confidence'] for ex in examples])
            print(f"{description}: {len(examples)} cases (avg confidence: {avg_conf:.3f})")
    
    print("\n" + "=" * 80)
    print(f"✓ Detailed analysis saved to {txt_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()

