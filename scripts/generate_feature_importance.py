#!/usr/bin/env python3
"""Generate feature importance chart for poster."""
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Import TextComplexityFeatures for unpickling
class TextComplexityFeatures:
    """Placeholder for unpickling."""
    pass

def main():
    model_path = Path("models/tfidf_logreg_3level_max")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    print("Loading model...")
    import __main__
    if not hasattr(__main__, 'TextComplexityFeatures'):
        __main__.TextComplexityFeatures = TextComplexityFeatures
    
    feature_union = joblib.load(model_path / "feature_union.joblib")
    classifier = joblib.load(model_path / "classifier.joblib")
    
    print("Extracting feature importance...")
    
    # Get feature names
    feature_names = []
    
    # TF-IDF features (first part of feature union)
    tfidf_transformer = feature_union.transformer_list[0][1]
    
    # Check if it's a Pipeline or direct TfidfVectorizer
    if hasattr(tfidf_transformer, 'named_steps'):
        tfidf_vectorizer = tfidf_transformer.named_steps['tfidf']
    else:
        tfidf_vectorizer = tfidf_transformer
    
    # Get TF-IDF feature names (limit to reasonable number)
    try:
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        # Get top 10 TF-IDF features by average coefficient
        feature_names.extend([f"TF-IDF: {name}" for name in tfidf_feature_names[:1000]])  # Store more for analysis
    except:
        # If we can't get names, use placeholder
        num_tfidf_features = tfidf_vectorizer.max_features if hasattr(tfidf_vectorizer, 'max_features') else 10000
        feature_names.extend([f"TF-IDF Feature {i}" for i in range(min(1000, num_tfidf_features))])
    
    # Complexity features (second part)
    complexity_feature_names = [
        "Word Count", "Sentence Count", "Avg Word Length", "Avg Sentence Length",
        "Lexical Diversity", "Question Count", "Has Question", "Technical Count",
        "Technical Density", "Has Comparison", "Has Explanation", "Has Definition",
        "Advanced Count", "Advanced Density", "Long Words", "Long Word Ratio",
        "Starts with What", "Starts with How", "Starts with Explain"
    ]
    feature_names.extend([f"Complexity: {name}" for name in complexity_feature_names])
    
    # Get coefficients for each class
    # Logistic regression coefficients shape: (n_classes, n_features)
    coefficients = classifier.coef_
    
    # Average absolute coefficients across classes for overall importance
    avg_importance = np.abs(coefficients).mean(axis=0)
    
    # Get top 15 features
    top_indices = np.argsort(avg_importance)[-15:][::-1]
    
    # Create feature names for top indices
    top_features = []
    for idx in top_indices:
        if idx < len(feature_names):
            top_features.append(feature_names[idx])
        elif idx < len(feature_names) + len(complexity_feature_names):
            comp_idx = idx - len(feature_names)
            top_features.append(f"Complexity: {complexity_feature_names[comp_idx]}")
        else:
            top_features.append(f"Feature {idx}")
    
    top_importance = avg_importance[top_indices]
    
    # Create chart
    print("Generating feature importance chart...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_importance, color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel('Average Absolute Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Important Features\n(TF-IDF Model)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_importance)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Invert y-axis to show highest at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    output_path = Path("results/poster_charts/feature_importance.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Feature importance chart saved to {output_path}")
    print(f"\nTop 5 features:")
    for i, (feat, imp) in enumerate(zip(top_features[:5], top_importance[:5]), 1):
        print(f"  {i}. {feat}: {imp:.3f}")

if __name__ == "__main__":
    main()

