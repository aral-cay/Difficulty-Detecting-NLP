#!/usr/bin/env python3
"""Generate charts for the research poster."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# Create output directory
output_dir = Path("results/poster_charts")
output_dir.mkdir(parents=True, exist_ok=True)

# Data
tfidf_scores = {
    'Overall': 45.09,
    'Easy': 44.50,
    'Medium': 60.50,
    'Hard': 30.35
}

distilbert_scores = {
    'Overall': 48.59,
    'Easy': 63.50,
    'Medium': 1.50,
    'Hard': 80.60
}

# 1. Model Comparison Bar Chart
print("Generating Model Comparison Bar Chart...")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(tfidf_scores))
width = 0.35

bars1 = ax.bar(x - width/2, list(tfidf_scores.values()), width, label='TF-IDF', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, list(distilbert_scores.values()), width, label='DistilBERT', color='#A23B72', alpha=0.8)

ax.set_xlabel('Category', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('TF-IDF vs DistilBERT Performance on ChatGPT Test Set', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tfidf_scores.keys())
ax.legend(fontsize=12)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_chatgpt.png', bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to {output_dir / 'model_comparison_chatgpt.png'}")

# 2. Test Set Performance Comparison
print("Generating Test Set Performance Chart...")
test_set_tfidf = {
    'Overall': 80.12,
    'Easy': 79.61,
    'Medium': 78.74,
    'Hard': 81.23
}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(test_set_tfidf))
width = 0.6

bars = ax.bar(x, list(test_set_tfidf.values()), width, color='#06A77D', alpha=0.8)

ax.set_xlabel('Category', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('TF-IDF Model Performance on LectureBank Test Set', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(test_set_tfidf.keys())
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'test_set_performance.png', bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to {output_dir / 'test_set_performance.png'}")

# 3. Radar Chart for Per-Category Performance
print("Generating Radar Chart...")
from math import pi

categories = ['Easy', 'Medium', 'Hard']
tfidf_values = [44.50, 60.50, 30.35]
distilbert_values = [63.50, 1.50, 80.60]

# Number of variables
N = len(categories)

# Compute angle for each category
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Add values
tfidf_values += tfidf_values[:1]
distilbert_values += distilbert_values[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Plot
ax.plot(angles, tfidf_values, 'o-', linewidth=2, label='TF-IDF', color='#2E86AB')
ax.fill(angles, tfidf_values, alpha=0.25, color='#2E86AB')
ax.plot(angles, distilbert_values, 'o-', linewidth=2, label='DistilBERT', color='#A23B72')
ax.fill(angles, distilbert_values, alpha=0.25, color='#A23B72')

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
ax.grid(True)

ax.set_title('Per-Category Performance Comparison\n(ChatGPT Test Set)', 
             size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'radar_chart.png', bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to {output_dir / 'radar_chart.png'}")

# 4. Domain Mismatch Comparison
print("Generating Domain Mismatch Chart...")
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Original\nTest Set', 'ChatGPT\nTest Set']
tfidf_acc = [80.12, 45.09]
distilbert_acc = [78.79, 48.59]  # From your DistilBERT test results

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, tfidf_acc, width, label='TF-IDF', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, distilbert_acc, width, label='DistilBERT', color='#A23B72', alpha=0.8)

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Domain Mismatch: LectureBank vs ChatGPT Questions', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=12)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'domain_mismatch.png', bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to {output_dir / 'domain_mismatch.png'}")

# 5. Per-Class Metrics Bar Chart
print("Generating Per-Class Metrics Chart...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['Precision', 'Recall', 'F1-Score']
levels = ['Level 1\n(Beginner)', 'Level 2\n(Intermediate)', 'Level 3\n(Advanced)']

# Data from test set results
precision = [0.8522, 0.5297, 0.8741]
recall = [0.7961, 0.7874, 0.8123]
f1 = [0.8232, 0.6333, 0.8420]

data = [precision, recall, f1]
colors = ['#06A77D', '#F18F01', '#C73E1D']

for idx, (metric, values, color) in enumerate(zip(metrics, data, colors)):
    ax = axes[idx]
    bars = ax.bar(levels, values, color=color, alpha=0.8)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.suptitle('Per-Class Performance Metrics (TF-IDF on Test Set)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'per_class_metrics.png', bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to {output_dir / 'per_class_metrics.png'}")

print(f"\n✓ All charts generated in {output_dir}/")
print("\nGenerated charts:")
print("  1. model_comparison_chatgpt.png - TF-IDF vs DistilBERT on ChatGPT test")
print("  2. test_set_performance.png - TF-IDF performance on LectureBank test")
print("  3. radar_chart.png - Radar chart comparing per-category performance")
print("  4. domain_mismatch.png - Comparison of original vs ChatGPT test sets")
print("  5. per_class_metrics.png - Precision, Recall, F1 for each class")

