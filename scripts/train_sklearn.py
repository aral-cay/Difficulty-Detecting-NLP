import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/lecture_depth3_train.csv")
    ap.add_argument("--val", default="data/processed/lecture_depth3_val.csv")
    ap.add_argument("--test", default="data/processed/lecture_depth3_test.csv")
    ap.add_argument("--output", default="models/tfidf_logreg_3level")
    args = ap.parse_args()

    print("Loading datasets...")
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
    
    # Enhanced TF-IDF: More features and better n-grams for Intermediate
    print("\nVectorizing text with enhanced TF-IDF (more features for Intermediate)...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased from 5000
        ngram_range=(1, 3),  # Added trigrams (was 1-2)
        stop_words='english',
        min_df=2,  # Filter very rare terms
        max_df=0.95  # Filter very common terms
    )
    X_train_vec = vectorizer.fit_transform(X_train_balanced)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
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
    
    # Train logistic regression with custom class weights
    print("\nTraining Logistic Regression classifier with custom class weights...")
    clf = LogisticRegression(
        max_iter=2000,  # Increased iterations
        random_state=42, 
        multi_class='multinomial',
        class_weight=custom_weights,  # Use custom weights
        C=1.0,  # Regularization strength
        solver='lbfgs'  # Better for multinomial
    )
    clf.fit(X_train_vec, y_train_balanced)
    
    # Evaluate on validation set
    print("\n=== Validation Set Results ===")
    y_val_pred = clf.predict(X_val_vec)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Accuracy: {val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))
    
    # Evaluate on test set
    print("\n=== Test Set Results ===")
    y_test_pred = clf.predict(X_test_vec)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Save model
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(args.output, 'vectorizer.joblib'))
    joblib.dump(clf, os.path.join(args.output, 'classifier.joblib'))
    print(f"\nModel saved to {args.output}/")

if __name__ == "__main__":
    main()

