#!/usr/bin/env python3
"""Generate class distribution chart for dataset section."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# Data from test set
levels = ['Level 1\n(Beginner)', 'Level 2\n(Intermediate)', 'Level 3\n(Advanced)']
counts = [2246, 555, 1726]
percentages = [49.6, 12.3, 38.1]
colors = ['#06A77D', '#F18F01', '#C73E1D']  # Green, Orange, Red

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Pie Chart
wedges, texts, autotexts = ax1.pie(counts, labels=levels, colors=colors, autopct='%1.1f%%',
                                    startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})

# Make percentage text white and bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax1.set_title('Class Distribution (Test Set)', fontsize=14, fontweight='bold', pad=20)

# 2. Bar Chart
bars = ax2.bar(levels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax2.set_title('Class Distribution by Count', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, count, pct in zip(bars, counts, percentages):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Dataset Class Distribution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/poster_charts/dataset_class_distribution.png', bbox_inches='tight', dpi=300)
plt.close()

# Also create a simpler single chart version
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(levels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
ax.set_title('Dataset Class Distribution (Test Set)', fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, count, pct in zip(bars, counts, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/poster_charts/dataset_class_distribution_simple.png', bbox_inches='tight', dpi=300)
plt.close()

print("✓ Class distribution charts saved:")
print("  - results/poster_charts/dataset_class_distribution.png (dual chart)")
print("  - results/poster_charts/dataset_class_distribution_simple.png (single chart)")

