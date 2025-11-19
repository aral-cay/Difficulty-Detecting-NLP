#!/usr/bin/env python3
"""Generate baseline comparison chart for poster."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# Data - Only 4 items as requested
# Random Guess: 1/3 = 33.33% for 3 classes
# Majority Class (Beginner): 2,246 / 4,527 = 49.6%
# TF-IDF Max: 80.12% (from test_confusion_matrix.txt)
# DistilBERT: 78.79% (evaluated on test set)

methods = ['Random\nGuess', 'Majority\nClass\n(Beginner)', 'TF-IDF', 'DistilBERT']
accuracies = [33.33, 49.6, 80.12, 78.79]  # DistilBERT evaluated on test set
colors = ['#E63946', '#F77F00', '#06A77D', '#A23B72']

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight our models (TF-IDF and DistilBERT)
bars[-2].set_edgecolor('green')
bars[-2].set_linewidth(3)
bars[-1].set_edgecolor('purple')
bars[-1].set_linewidth(3)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotations
# Show improvement from majority class to TF-IDF
ax.annotate('+30.5%', xy=(2, 80.12), xytext=(2, 88),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, fontweight='bold', color='green',
            ha='center')

# Customize
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Baseline Comparison: Model Performance', fontsize=16, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add horizontal line at 50% for reference
ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Add legend for our models
from matplotlib.patches import Rectangle
legend_elements = [
    Rectangle((0, 0), 1, 1, facecolor='#06A77D', edgecolor='green', linewidth=3, label='Our Models'),
    Rectangle((0, 0), 1, 1, facecolor='#E63946', label='Baselines')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('results/poster_charts/baseline_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

print("✓ Baseline comparison chart saved to results/poster_charts/baseline_comparison.png")
print(f"\nValues used:")
print(f"  Random Guess: 33.33%")
print(f"  Majority Class (Beginner): 49.6% (2,246/4,527)")
print(f"  TF-IDF: 80.12%")
print(f"  DistilBERT: 78.79% (evaluated on test set)")
