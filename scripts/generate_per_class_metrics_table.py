#!/usr/bin/env python3
"""Generate a per-class metrics table visualization for the poster."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Create output directory
output_dir = "results/poster_charts"
import os
os.makedirs(output_dir, exist_ok=True)

# Data from test_confusion_matrix.txt
data = {
    'Level 1 (Beginner)': {'Precision': 0.8522, 'Recall': 0.7961, 'F1': 0.8232, 'Support': 2246},
    'Level 2 (Intermediate)': {'Precision': 0.5297, 'Recall': 0.7874, 'F1': 0.6333, 'Support': 555},
    'Level 3 (Advanced)': {'Precision': 0.8741, 'Recall': 0.8123, 'F1': 0.8420, 'Support': 1726},
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Table data
rows = ['Level 1\n(Beginner)', 'Level 2\n(Intermediate)', 'Level 3\n(Advanced)']
columns = ['Precision', 'Recall', 'F1-Score', 'Support']
cell_text = []

for level in rows:
    level_key = level.replace('\n', ' ')
    row = [
        f"{data[level_key]['Precision']:.4f}",
        f"{data[level_key]['Recall']:.4f}",
        f"{data[level_key]['F1']:.4f}",
        f"{data[level_key]['Support']:,}"
    ]
    cell_text.append(row)

# Create table
table = ax.table(cellText=cell_text,
                rowLabels=rows,
                colLabels=columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(rows) + 1):
    for j in range(len(columns)):
        cell = table[(i, j)]
        if i == 0:  # Header row
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
        else:
            # Color code based on values
            if j < 3:  # Precision, Recall, F1 columns
                value = float(cell_text[i-1][j])
                if value >= 0.80:
                    cell.set_facecolor('#D4EDDA')  # Light green
                elif value >= 0.65:
                    cell.set_facecolor('#FFF3CD')  # Light yellow
                else:
                    cell.set_facecolor('#F8D7DA')  # Light red
            else:  # Support column
                cell.set_facecolor('#F0F0F0')  # Light gray
            cell.set_text_props(weight='normal', color='black')

# Add title
ax.set_title('Per-Class Performance Metrics\nTF-IDF Model on Test Set (n=4,527)', 
             fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/per_class_metrics_table.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Generated per-class metrics table: {output_dir}/per_class_metrics_table.png")

