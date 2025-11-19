#!/usr/bin/env python3
"""Generate misclassification analysis chart for poster."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Data from confusion matrix (percentages)
misclassifications = {
    'Level 1 → Level 2': 12.6,
    'Level 3 → Level 1': 12.6,
    'Level 1 → Level 3': 7.8,
    'Level 3 → Level 2': 6.1,
    'Level 2 → Level 1': 16.6,
    'Level 2 → Level 3': 4.7
}

# Sort by percentage
sorted_misclass = dict(sorted(misclassifications.items(), key=lambda x: x[1], reverse=True))

# Create chart
fig, ax = plt.subplots(figsize=(10, 6))

# Colors - use red tones for misclassifications
colors = ['#E63946', '#E63946', '#F77F00', '#FCBF49', '#F77F00', '#FCBF49']

bars = ax.barh(list(sorted_misclass.keys()), list(sorted_misclass.values()), 
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Customize
ax.set_xlabel('Percentage of Misclassifications (%)', fontsize=12, fontweight='bold')
ax.set_title('Most Common Misclassification Patterns\n(TF-IDF Model on Test Set)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars, sorted_misclass.values()):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%',
            ha='left', va='center', fontsize=10, fontweight='bold')

# Add annotation
ax.text(0.5, 0.02, 'Note: Diagonal values (correct predictions) not shown',
        transform=ax.transAxes, fontsize=9, style='italic', ha='center')

plt.tight_layout()
output_path = "results/poster_charts/misclassification_analysis.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Misclassification analysis chart saved to {output_path}")

