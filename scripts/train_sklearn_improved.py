#!/usr/bin/env python3
"""
Improved Sklearn training with hyperparameter tuning and feature engineering.
Expected improvement: +2-4% accuracy
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import re

class TextComplexityFeatures(BaseEstimator, TransformerMixin):
    """Extract text complexity features."""
    
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
            avg_word_length = np.mean([len(w) for w in text_str.split()]) if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Lexical diversity
            words = text_str.lower().split()
            unique_words = len(set(words))
            lexical_diversity = unique_words / word_count if word_count > 0 else 0
            
            # Question indicators
            question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'explain', 'describe']
            question_count = sum(1 for word in question_words if word in text_str.lower())
            has_question = 1 if any(text_str.strip().endswith('?') for _ in [1]) else 0
            
            # Technical term indicators
            technical_terms = ['algorithm', 'function', 'method', 'model', 'neural', 'network', 
                             'learning', 'processing', 'extraction', 'parsing', 'semantic',
                             'transformer', 'attention', 'embedding', 'vector', 'matrix']
            technical_count = sum(1 for term in technical_terms if term in text_str.lower())
            technical_density = technical_count / word_count if word_count > 0 else 0
            
            # Complexity indicators
            has_comparison = 1 if any(word in text_str.lower() for word in ['difference', 'compare', 'versus', 'vs', 'between']) else 0
            has_explanation = 1 if any(word in text_str.lower() for word in ['explain', 'describe', 'how does', 'how do']) else 0
            has_definition = 1 if any(word in text_str.lower() for word in ['what is', 'what are', 'define', 'definition']) else 0
            
            # Advanced terms
            advanced_terms = ['theoretical', 'formulation', 'architecture', 'mechanism', 'adversarial',
                            'variational', 'autoencoder', 'transformer', 'biaffine', 'meta-learning']
            advanced_count = sum(1 for term in advanced_terms if term in text_str.lower())
            
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
                advanced_count
            ]
            features.append(feature_vector)
        
        return np.array(features)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/lecture_depth3_train.csv")
    ap.add_argument("--val", default="data/processed/lecture_depth3_val.csv")
    ap.add_argument("--test", default="data/processed/lecture_depth3_test.csv")
    ap.add_argument("--output", default="models/tfidf_logreg_3level_improved")
    ap.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (slower)")
    args = ap.parse_args()

    print("=" * 80)
    print("IMPROVED SKLEARN TRAINING WITH FEATURE ENGINEERING")
    print("=" * 80)
    
    print("\nLoading datasets...")
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Prepare data
    X_train = train_df['text'].fillna('').astype(str)
    y_train = train_df['depth_level'].astype(int)
    
    X_val = val_df['text'].fillna('').astype(str)
    y_val = val_df['depth_level'].astype(int)
    
    X_test = test_df['text'].fillna('').astype(str)
    y_test = test_df['depth_level'].astype(int)
    
    # Enhanced oversampling: Give Intermediate class extra boost
    print("\nBalancing dataset with enhanced oversampling for Intermediate class...")
    from sklearn.utils import resample
    
    # Separate by class
    train_df_classes = {level: train_df[train_df['depth_level'] == level] for level in [1, 2, 3]}
    
    # Find max class size
    max_size = max(len(train_df_classes[level]) for level in [1, 2, 3])
    print(f"  Original sizes: Level 1={len(train_df_classes[1])}, Level 2={len(train_df_classes[2])}, Level 3={len(train_df_classes[3])}")
    
    # Special handling: Oversample Intermediate more aggressively (1.5x instead of 1x)
    intermediate_target = int(max_size * 1.5)  # Give Intermediate 50% more samples
    
    train_df_balanced = []
    for level in [1, 2, 3]:
        df_level = train_df_classes[level]
        if level == 2:  # Intermediate - extra oversampling
            if len(df_level) < intermediate_target:
                df_oversampled = resample(df_level, replace=True, n_samples=intermediate_target, random_state=42)
                train_df_balanced.append(df_oversampled)
                print(f"  Oversampled Level {level} (Intermediate) from {len(df_level)} to {len(df_oversampled)} (1.5x boost)")
            else:
                train_df_balanced.append(df_level)
        else:  # Beginner and Advanced - match max size
            if len(df_level) < max_size:
                df_oversampled = resample(df_level, replace=True, n_samples=max_size, random_state=42)
                train_df_balanced.append(df_oversampled)
                print(f"  Oversampled Level {level} from {len(df_level)} to {len(df_oversampled)}")
            else:
                train_df_balanced.append(df_level)
    
    train_df_balanced = pd.concat(train_df_balanced, ignore_index=True)
    train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"  Final balanced size: {len(train_df_balanced)} samples")
    print(f"  Final distribution: {train_df_balanced['depth_level'].value_counts().sort_index().to_dict()}")
    
    X_train_balanced = train_df_balanced['text'].fillna('').astype(str)
    y_train_balanced = train_df_balanced['depth_level'].astype(int)
    
    # Combine TF-IDF + Complexity Features
    print("\nCreating feature pipeline (TF-IDF + Text Complexity Features)...")
    
    # Feature union: TF-IDF + Complexity features
    feature_union = FeatureUnion([
        ('tfidf', TfidfVectorizer(
            max_features=12000,  # Slightly increased
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True  # Use log scale for term frequency
        )),
        ('complexity', Pipeline([
            ('complexity_features', TextComplexityFeatures()),
            ('scaler', StandardScaler())
        ]))
    ])
    
    # Transform training data
    print("  Transforming training data...")
    X_train_features = feature_union.fit_transform(X_train_balanced)
    X_val_features = feature_union.transform(X_val)
    X_test_features = feature_union.transform(X_test)
    
    print(f"  Feature dimensions: {X_train_features.shape[1]} (TF-IDF + 14 complexity features)")
    
    # Custom class weights: Boost Intermediate even more
    from sklearn.utils.class_weight import compute_class_weight
    base_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
    
    # Custom weights: give Intermediate 2x boost
    custom_weights = {}
    for i, level in enumerate(sorted(np.unique(y_train_balanced))):
        if level == 2:  # Intermediate
            custom_weights[level] = base_weights[i] * 2.0  # Double the weight for Intermediate
        else:
            custom_weights[level] = base_weights[i]
    
    print(f"\nCustom class weights (Intermediate boosted 2x): {custom_weights}")
    
    # Hyperparameter tuning or use best known parameters
    if args.tune:
        print("\nRunning hyperparameter tuning (this may take a while)...")
        param_grid = {
            'C': [0.5, 1.0, 2.0, 5.0],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [2000, 3000]
        }
        
        base_clf = LogisticRegression(
            random_state=42,
            multi_class='multinomial',
            class_weight=custom_weights
        )
        
        grid_search = GridSearchCV(
            base_clf,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_features, y_train_balanced)
        clf = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    else:
        print("\nTraining Logistic Regression with optimized parameters...")
        # Use best known parameters from analysis
        clf = LogisticRegression(
            max_iter=3000,  # Increased iterations
            random_state=42,
            multi_class='multinomial',
            class_weight=custom_weights,
            C=2.0,  # Slightly higher regularization (was 1.0)
            solver='lbfgs'
        )
        clf.fit(X_train_features, y_train_balanced)
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET RESULTS")
    print("=" * 80)
    y_val_pred = clf.predict(X_val_features)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Macro F1: {val_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    y_test_pred = clf.predict(X_test_features)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Save model components
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(feature_union, os.path.join(args.output, 'feature_union.joblib'))
    joblib.dump(clf, os.path.join(args.output, 'classifier.joblib'))
    
    # Also save separately for backward compatibility
    joblib.dump(feature_union.named_transformers['tfidf'], os.path.join(args.output, 'vectorizer.joblib'))
    
    print(f"\n{'=' * 80}")
    print(f"Model saved to {args.output}/")
    print(f"{'=' * 80}")
    print(f"\nImprovements made:")
    print(f"  ✅ Added 14 text complexity features")
    print(f"  ✅ Increased TF-IDF features to 12,000")
    print(f"  ✅ Added sublinear_tf scaling")
    print(f"  ✅ Optimized C parameter (2.0)")
    print(f"  ✅ Increased max_iter (3000)")
    print(f"\nExpected improvement: +2-4% accuracy")

if __name__ == "__main__":
    main()

