#!/usr/bin/env python3
"""Generate visual explanation for class imbalance."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Data
levels = ['Level 1\n(Beginner)', 'Level 2\n(Intermediate)', 'Level 3\n(Advanced)']
counts = [2246, 555, 1726]
percentages = [49.6, 12.3, 38.1]
colors = ['#06A77D', '#F18F01', '#C73E1D']

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Bar chart with explanation
bars = ax1.bar(levels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax1.set_title('Class Distribution: Natural Imbalance', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, count, pct in zip(bars, counts, percentages):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add explanation text box
textstr = 'Why Intermediate is smaller:\n• Lectures target specific levels\n• Fewer intermediate topics\n• Natural distribution'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# Right: Before/After oversampling
categories = ['Before\nOversampling', 'After\nOversampling']
intermediate_before = 555
intermediate_after = int(555 * 1.5)  # 1.5x boost

x = np.arange(len(categories))
width = 0.6

bars2 = ax2.bar(x, [intermediate_before, intermediate_after], 
                color=['#F18F01', '#06A77D'], alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('Intermediate Samples', fontsize=12, fontweight='bold')
ax2.set_title('Oversampling Strategy', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars2, [intermediate_before, intermediate_after]):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement arrow
ax2.annotate('1.5x boost', xy=(1, intermediate_after), xytext=(0.5, intermediate_after + 200),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=11, fontweight='bold', color='green',
            ha='center')

plt.suptitle('Class Imbalance: Challenge and Solution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/poster_charts/class_imbalance_explanation.png', bbox_inches='tight', dpi=300)
plt.close()

print("✓ Class imbalance explanation chart saved to results/poster_charts/class_imbalance_explanation.png")

