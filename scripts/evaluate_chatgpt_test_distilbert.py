#!/usr/bin/env python3
"""Evaluate DistilBERT model performance on ChatGPT-generated test questions."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def load_questions(file_path):
    """Load questions from a file, splitting by empty lines."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out empty lines and strip whitespace
        questions = [line.strip() for line in lines if line.strip()]
        
        return questions
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}")
        return []

def main():
    print("=" * 80)
    print("EVALUATING DISTILBERT MODEL ON CHATGPT TEST QUESTIONS")
    print("=" * 80)
    
    # Load model
    model_path = Path("models/distilbert_depth3_fast/best")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first: python scripts/train_hf_3levels_fast.py")
        sys.exit(1)
    
    print(f"\nLoading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load test questions
    test_dir = Path("chatGptTest")
    if not test_dir.exists():
        print(f"ERROR: Test directory not found at {test_dir}")
        sys.exit(1)
    
    print("\nLoading test questions...")
    easy_questions = load_questions(test_dir / "Easy")
    medium_questions = load_questions(test_dir / "Medium")
    hard_questions = load_questions(test_dir / "Hard")
    
    print(f"  Easy:   {len(easy_questions)} questions")
    print(f"  Medium: {len(medium_questions)} questions")
    print(f"  Hard:   {len(hard_questions)} questions")
    
    # Check if we have questions
    if len(easy_questions) == 0 and len(medium_questions) == 0 and len(hard_questions) == 0:
        print("\nERROR: No questions found in any test file!")
        print("Please make sure the Easy, Medium, and Hard files are saved and contain questions.")
        sys.exit(1)
    
    if len(easy_questions) == 0:
        print("\nWARNING: Easy file is empty. Evaluation will only include Medium and Hard.")
    if len(hard_questions) == 0:
        print("\nWARNING: Hard file is empty. Evaluation will only include Easy and Medium.")
    
    # Prepare data: Easy -> Level 1, Medium -> Level 2, Hard -> Level 3
    all_questions = easy_questions + medium_questions + hard_questions
    true_labels = [1] * len(easy_questions) + [2] * len(medium_questions) + [3] * len(hard_questions)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(all_questions), batch_size):
            batch_questions = all_questions[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_questions,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Convert from 0-2 to 1-3
            batch_preds = batch_preds + 1
            predictions.extend(batch_preds.tolist())
            
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(all_questions):
                print(f"  Processed {min(i + batch_size, len(all_questions))}/{len(all_questions)} questions...")
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    
    # Overall accuracy
    overall_accuracy = accuracy_score(true_labels, predictions)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=[1, 2, 3], average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, labels=[1, 2, 3], average='weighted', zero_division=0
    )
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    print(f"{'Level 1 (Easy)':<15} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1[0]:<12.4f} {support[0]:<10}")
    print(f"{'Level 2 (Medium)':<15} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1[1]:<12.4f} {support[1]:<10}")
    print(f"{'Level 3 (Hard)':<15} {precision[2]:<12.4f} {recall[2]:<12.4f} {f1[2]:<12.4f} {support[2]:<10}")
    print("-" * 80)
    print(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {len(true_labels):<10}")
    print(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {len(true_labels):<10}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(true_labels, predictions, labels=[1, 2, 3])
    cm_df = pd.DataFrame(cm, 
                        index=['True Level 1', 'True Level 2', 'True Level 3'],
                        columns=['Pred Level 1', 'Pred Level 2', 'Pred Level 3'])
    print(cm_df)
    
    # Calculate percentages for better interpretation
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent_df = pd.DataFrame(cm_percent, 
                                 index=['True Level 1', 'True Level 2', 'True Level 3'],
                                 columns=['Pred Level 1', 'Pred Level 2', 'Pred Level 3'])
    
    print("\nConfusion Matrix (Percentages):")
    print("-" * 80)
    print(cm_percent_df.round(2))
    
    # Per-category accuracy
    print("\nPer-Category Accuracy:")
    print("-" * 80)
    easy_mask = true_labels == 1
    medium_mask = true_labels == 2
    hard_mask = true_labels == 3
    
    easy_acc = accuracy_score(true_labels[easy_mask], predictions[easy_mask]) if easy_mask.sum() > 0 else 0
    medium_acc = accuracy_score(true_labels[medium_mask], predictions[medium_mask]) if medium_mask.sum() > 0 else 0
    hard_acc = accuracy_score(true_labels[hard_mask], predictions[hard_mask]) if hard_mask.sum() > 0 else 0
    
    print(f"Easy (Level 1):   {easy_acc:.4f} ({easy_acc*100:.2f}%)")
    print(f"Medium (Level 2): {medium_acc:.4f} ({medium_acc*100:.2f}%)")
    print(f"Hard (Level 3):   {hard_acc:.4f} ({hard_acc*100:.2f}%)")
    
    # Detailed classification report
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(true_labels, predictions, 
                               target_names=['Level 1 (Easy)', 'Level 2 (Medium)', 'Level 3 (Hard)'],
                               labels=[1, 2, 3]))
    
    # Save results to file
    results_file = Path("results/chatgpt_test_evaluation_distilbert.txt")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CHATGPT TEST EVALUATION RESULTS - DISTILBERT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Level 1 (Easy)':<15} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1[0]:<12.4f} {support[0]:<10}\n")
        f.write(f"{'Level 2 (Medium)':<15} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1[1]:<12.4f} {support[1]:<10}\n")
        f.write(f"{'Level 3 (Hard)':<15} {precision[2]:<12.4f} {recall[2]:<12.4f} {f1[2]:<12.4f} {support[2]:<10}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {len(true_labels):<10}\n")
        f.write(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {len(true_labels):<10}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 80 + "\n")
        f.write(str(cm_df) + "\n\n")
        f.write("Confusion Matrix (Percentages):\n")
        f.write("-" * 80 + "\n")
        f.write(str(cm_percent_df.round(2)) + "\n\n")
        f.write("Per-Category Accuracy:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Easy (Level 1):   {easy_acc:.4f} ({easy_acc*100:.2f}%)\n")
        f.write(f"Medium (Level 2): {medium_acc:.4f} ({medium_acc*100:.2f}%)\n")
        f.write(f"Hard (Level 3):   {hard_acc:.4f} ({hard_acc*100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(true_labels, predictions, 
                                     target_names=['Level 1 (Easy)', 'Level 2 (Medium)', 'Level 3 (Hard)'],
                                     labels=[1, 2, 3]))
    
    print(f"\n✓ Results saved to {results_file}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

