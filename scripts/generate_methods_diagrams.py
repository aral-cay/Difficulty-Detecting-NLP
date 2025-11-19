#!/usr/bin/env python3
"""Generate visual diagrams for Methods section."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = "results/poster_charts"
import os
os.makedirs(output_dir, exist_ok=True)

# 1. Data Pipeline Diagram
print("Generating Data Pipeline Diagram...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Boxes
boxes = [
    (1, 8, "Raw Files\n(PDF/PPTX)", '#E63946'),
    (1, 6, "Text\nExtraction", '#F77F00'),
    (1, 4, "Depth\nComputation\n(ConceptBank)", '#FCBF49'),
    (1, 2, "Text\nChunking\n(512 words)", '#06A77D'),
    (5, 2, "Relabeling\n(5→3 levels)", '#2A9D8F'),
    (5, 4, "Train/Val/Test\nSplit\n(70/15/15)", '#264653'),
]

# Draw boxes
for x, y, text, color in boxes:
    box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Arrows
arrows = [
    ((1, 7.4), (1, 6.6)),  # Raw → Extraction
    ((1, 5.4), (1, 4.6)),  # Extraction → Depth
    ((1, 3.4), (1, 2.6)),  # Depth → Chunking
    ((1.8, 2), (4.2, 2)),  # Chunking → Relabeling
    ((5, 2.6), (5, 3.4)),  # Relabeling → Split
]

for (x1, y1), (x2, y2) in arrows:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

ax.set_title('Data Preprocessing Pipeline', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/data_pipeline_diagram.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to {output_dir}/data_pipeline_diagram.png")

# 2. Model Architecture Diagram
print("Generating Model Architecture Diagram...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Input
input_box = FancyBboxPatch((3.5, 8.5), 3, 0.8,
                          boxstyle="round,pad=0.1",
                          facecolor='#E63946', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(input_box)
ax.text(5, 8.9, "Input Text", ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Feature extraction boxes
tfidf_box = FancyBboxPatch((1, 6), 2.5, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor='#2E86AB', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(tfidf_box)
ax.text(2.25, 7, "TF-IDF\nVectorizer", ha='center', va='center', fontsize=9, fontweight='bold', color='white')
ax.text(2.25, 6.4, "• 10,000 features\n• N-grams (1-3)", ha='center', va='center', fontsize=8, color='white')

complexity_box = FancyBboxPatch((6.5, 6), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#A23B72', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(complexity_box)
ax.text(7.75, 7, "Complexity\nFeatures", ha='center', va='center', fontsize=9, fontweight='bold', color='white')
ax.text(7.75, 6.4, "• 20 features\n• Lexical diversity", ha='center', va='center', fontsize=8, color='white')

# Feature Union
union_box = FancyBboxPatch((3.5, 4), 3, 1,
                          boxstyle="round,pad=0.1",
                          facecolor='#F18F01', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(union_box)
ax.text(5, 4.5, "Feature Union", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Classifier
classifier_box = FancyBboxPatch((3.5, 2), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor='#06A77D', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(classifier_box)
ax.text(5, 2.5, "Logistic Regression\nC=3.0, Class Weights", ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Output
output_box = FancyBboxPatch((3.5, 0.2), 3, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor='#C73E1D', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(output_box)
ax.text(5, 0.6, "Output: Level 1, 2, or 3", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Arrows
arrows = [
    ((5, 8.5), (2.25, 7.5)),  # Input → TF-IDF
    ((5, 8.5), (7.75, 7.5)),  # Input → Complexity
    ((2.25, 6), (4.5, 4.5)),  # TF-IDF → Union
    ((7.75, 6), (5.5, 4.5)),  # Complexity → Union
    ((5, 4), (5, 3)),         # Union → Classifier
    ((5, 2), (5, 1)),         # Classifier → Output
]

for (x1, y1), (x2, y2) in arrows:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

ax.set_title('Model Architecture: TF-IDF + Logistic Regression', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/model_architecture_diagram.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to {output_dir}/model_architecture_diagram.png")

# 3. Feature Engineering Diagram
print("Generating Feature Engineering Diagram...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Input
ax.text(5, 5.5, "Input Text", ha='center', va='center', fontsize=12, fontweight='bold')

# Feature boxes
tfidf_features = [
    "TF-IDF Features (10,000)",
    "• Unigrams, Bigrams, Trigrams",
    "• English stopwords removed",
    "• Sublinear TF scaling"
]

complexity_features = [
    "Complexity Features (20)",
    "• Word/Sentence count",
    "• Lexical diversity",
    "• Technical term density",
    "• Advanced term count"
]

# Left side - TF-IDF
tfidf_box = FancyBboxPatch((0.5, 2.5), 4, 2,
                          boxstyle="round,pad=0.2",
                          facecolor='#2E86AB', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(tfidf_box)
y_pos = 4
for line in tfidf_features:
    ax.text(2.5, y_pos, line, ha='center', va='center', fontsize=9, color='white', fontweight='bold' if 'TF-IDF' in line else 'normal')
    y_pos -= 0.4

# Right side - Complexity
comp_box = FancyBboxPatch((5.5, 2), 4, 2.5,
                         boxstyle="round,pad=0.2",
                         facecolor='#A23B72', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(comp_box)
y_pos = 4
for line in complexity_features:
    ax.text(7.5, y_pos, line, ha='center', va='center', fontsize=9, color='white', fontweight='bold' if 'Complexity' in line else 'normal')
    y_pos -= 0.4

# Feature Union
union_box = FancyBboxPatch((3.5, 0.5), 3, 1,
                          boxstyle="round,pad=0.2",
                          facecolor='#06A77D', edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(union_box)
ax.text(5, 1, "Feature Union\n10,020 Total Features", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Arrows
arrows = [
    ((5, 5.3), (2.5, 4.5)),  # Input → TF-IDF
    ((5, 5.3), (7.5, 4.5)),  # Input → Complexity
    ((2.5, 2.5), (4.5, 1.5)),  # TF-IDF → Union
    ((7.5, 2), (5.5, 1.5)),  # Complexity → Union
]

for (x1, y1), (x2, y2) in arrows:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

ax.set_title('Feature Engineering Pipeline', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/feature_engineering_diagram.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"  ✓ Saved to {output_dir}/feature_engineering_diagram.png")

print(f"\n✓ All Methods diagrams generated in {output_dir}/")
print("\nGenerated diagrams:")
print("  1. data_pipeline_diagram.png - Preprocessing pipeline")
print("  2. model_architecture_diagram.png - Model architecture")
print("  3. feature_engineering_diagram.png - Feature engineering flow")

