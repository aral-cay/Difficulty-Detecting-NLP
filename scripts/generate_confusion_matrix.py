#!/usr/bin/env python3
"""Generate confusion matrix for the model on the actual test set."""
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix
)
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

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate confusion matrix for model on test set")
    ap.add_argument("--test", default="data/processed/lecture_depth3_test.csv", 
                    help="Path to test CSV file")
    ap.add_argument("--model", default="models/tfidf_logreg_3level_max",
                    help="Path to model directory")
    ap.add_argument("--output", default="results/test_confusion_matrix",
                    help="Output prefix for saving results")
    ap.add_argument("--save_plot", action="store_true",
                    help="Save confusion matrix as PNG image")
    args = ap.parse_args()
    
    print("=" * 80)
    print("GENERATING CONFUSION MATRIX FOR TEST SET")
    print("=" * 80)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first: python scripts/train_sklearn_max.py --use_smote")
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
    
    print(f"\nTest set distribution:")
    print(f"  Level 1 (Beginner):   {(y_true == 1).sum()} samples")
    print(f"  Level 2 (Intermediate): {(y_true == 2).sum()} samples")
    print(f"  Level 3 (Advanced):   {(y_true == 3).sum()} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    X_test_features = feature_union.transform(X_test)
    y_pred = classifier.predict(X_test_features)
    y_pred = y_pred.astype(int)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 2, 3], average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 2, 3], average='weighted', zero_division=0
    )
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    print(f"{'Level 1 (Beginner)':<20} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1[0]:<12.4f} {support[0]:<10}")
    print(f"{'Level 2 (Intermediate)':<20} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1[1]:<12.4f} {support[1]:<10}")
    print(f"{'Level 3 (Advanced)':<20} {precision[2]:<12.4f} {recall[2]:<12.4f} {f1[2]:<12.4f} {support[2]:<10}")
    print("-" * 80)
    print(f"{'Macro Avg':<20} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {len(y_true):<10}")
    print(f"{'Weighted Avg':<20} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {len(y_true):<10}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    
    print("\nConfusion Matrix:")
    print("-" * 80)
    cm_df = pd.DataFrame(cm, 
                        index=['True Level 1', 'True Level 2', 'True Level 3'],
                        columns=['Pred Level 1', 'Pred Level 2', 'Pred Level 3'])
    print(cm_df)
    
    # Calculate percentages for better interpretation
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent_df = pd.DataFrame(cm_percent, 
                                 index=['True Level 1', 'True Level 2', 'True Level 3'],
                                 columns=['Pred Level 1', 'Pred Level 2', 'Pred Level 3'])
    
    print("\nConfusion Matrix (Percentages):")
    print("-" * 80)
    print(cm_percent_df.round(2))
    
    # Per-category accuracy
    print("\nPer-Category Accuracy:")
    print("-" * 80)
    for level in [1, 2, 3]:
        mask = y_true == level
        if mask.sum() > 0:
            level_acc = accuracy_score(y_true[mask], y_pred[mask])
            level_name = {1: "Level 1 (Beginner)", 2: "Level 2 (Intermediate)", 3: "Level 3 (Advanced)"}[level]
            print(f"{level_name}: {level_acc:.4f} ({level_acc*100:.2f}%)")
    
    # Detailed classification report
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_true, y_pred, 
                               target_names=['Level 1 (Beginner)', 'Level 2 (Intermediate)', 'Level 3 (Advanced)'],
                               labels=[1, 2, 3]))
    
    # Save results to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    txt_file = output_path.with_suffix('.txt')
    with open(txt_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST SET CONFUSION MATRIX RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Set: {args.test}\n")
        f.write(f"Test Samples: {len(test_df)}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Level 1 (Beginner)':<20} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1[0]:<12.4f} {support[0]:<10}\n")
        f.write(f"{'Level 2 (Intermediate)':<20} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1[1]:<12.4f} {support[1]:<10}\n")
        f.write(f"{'Level 3 (Advanced)':<20} {precision[2]:<12.4f} {recall[2]:<12.4f} {f1[2]:<12.4f} {support[2]:<10}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Macro Avg':<20} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {len(y_true):<10}\n")
        f.write(f"{'Weighted Avg':<20} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {len(y_true):<10}\n\n")
        f.write("Confusion Matrix (Counts):\n")
        f.write("-" * 80 + "\n")
        f.write(str(cm_df) + "\n\n")
        f.write("Confusion Matrix (Percentages):\n")
        f.write("-" * 80 + "\n")
        f.write(str(cm_percent_df.round(2)) + "\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(y_true, y_pred, 
                                     target_names=['Level 1 (Beginner)', 'Level 2 (Intermediate)', 'Level 3 (Advanced)'],
                                     labels=[1, 2, 3]))
    
    print(f"\n✓ Results saved to {txt_file}")
    
    # Save confusion matrix plot if requested
    if args.save_plot:
        plt.figure(figsize=(10, 8))
        # Create annotations with percentages and counts
        annot = []
        for i in range(len(cm)):
            row = []
            for j in range(len(cm[i])):
                percent = cm_percent[i, j]
                count = cm[i, j]
                row.append(f'{percent:.1f}%\n({count})')
            annot.append(row)
        
        # Use green tones colormap
        sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Greens', 
                   xticklabels=['Level 1', 'Level 2', 'Level 3'],
                   yticklabels=['Level 1', 'Level 2', 'Level 3'],
                   cbar_kws={'label': 'Percentage (%)'},
                   vmin=0, vmax=100)
        plt.title(f'Confusion Matrix (Percentages)\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        plot_file = output_path.with_suffix('.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix plot saved to {plot_file}")
        plt.close()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

