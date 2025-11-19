#!/usr/bin/env python3
"""
Fast DistilBERT training optimized for ~1.5 hour completion:
- Reduced epochs (3-4 with early stopping)
- Larger batch size (32)
- Shorter context (256 tokens)
- Optimized for speed while maintaining good accuracy
"""
import argparse
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np
import os
import time

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/lecture_depth3_train.csv")
    ap.add_argument("--val", default="data/processed/lecture_depth3_val.csv")
    ap.add_argument("--test", default="data/processed/lecture_depth3_test.csv")
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--output", default="models/distilbert_depth3_fast")
    ap.add_argument("--batch_size", type=int, default=32)  # Larger batch for speed
    ap.add_argument("--epochs", type=int, default=4)  # Fewer epochs
    ap.add_argument("--learning_rate", type=float, default=3e-5)  # Slightly higher LR for faster convergence
    ap.add_argument("--max_length", type=int, default=256)  # Shorter context for speed
    args = ap.parse_args()

    start_time = time.time()
    
    print("=" * 80)
    print("FAST DISTILBERT TRAINING (Target: ~1.5 hours)")
    print("=" * 80)
    
    print("\nLoading datasets...")
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Balanced oversampling (simpler version for speed)
    print("\nBalancing dataset...")
    from sklearn.utils import resample
    
    train_df_classes = {level: train_df[train_df['depth_level'] == level] for level in [1, 2, 3]}
    max_size = max(len(train_df_classes[level]) for level in [1, 2, 3])
    print(f"  Original sizes: Level 1={len(train_df_classes[1])}, Level 2={len(train_df_classes[2])}, Level 3={len(train_df_classes[3])}")
    
    train_df_balanced = []
    for level in [1, 2, 3]:
        df_level = train_df_classes[level]
        if len(df_level) < max_size:
            df_oversampled = resample(df_level, replace=True, n_samples=max_size, random_state=42)
            train_df_balanced.append(df_oversampled)
            print(f"  Oversampled Level {level} from {len(df_level)} to {len(df_oversampled)}")
        else:
            train_df_balanced.append(df_level)
    
    train_df_balanced = pd.concat(train_df_balanced, ignore_index=True)
    train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Final balanced size: {len(train_df_balanced)} samples")
    
    # Prepare data
    train_texts = train_df_balanced['text'].fillna('').astype(str).tolist()
    train_labels = (train_df_balanced['depth_level'].astype(int) - 1).tolist()
    
    val_texts = val_df['text'].fillna('').astype(str).tolist()
    val_labels = (val_df['depth_level'].astype(int) - 1).tolist()
    
    test_texts = test_df['text'].fillna('').astype(str).tolist()
    test_labels = (test_df['depth_level'].astype(int) - 1).tolist()
    
    # Class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"\nClass weights: {dict(zip(range(len(class_weights)), class_weights))}")
    
    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    num_labels = 3
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    
    # Tokenize
    print(f"Tokenizing texts (max_length={args.max_length})...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=args.max_length,
            return_attention_mask=True
        )
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})
    
    print("  Tokenizing train set...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1000)
    print("  Tokenizing val set...")
    val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=1000)
    print("  Tokenizing test set...")
    test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=1000)
    
    # Check device
    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    use_fp16 = device_type == "cuda"
    print(f"  Device: {device_type}, fp16: {use_fp16}")
    
    # Fast training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,  # Keep fewer checkpoints
        logging_steps=100,  # Less frequent logging
        logging_dir=os.path.join(args.output, 'logs'),
        report_to="none",
        fp16=use_fp16,
        dataloader_num_workers=0 if device_type == "mps" else 2,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        remove_unused_columns=False,
    )
    
    # Custom loss function with class weights
    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            if class_weights is not None:
                self.class_weights = class_weights.to(self.model.device)
            else:
                self.class_weights = None
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    # Trainer with early stopping
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Less patience for speed
        class_weights=class_weight_tensor
    )
    
    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"  Epochs: {args.epochs} (with early stopping patience=2)")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  Estimated time: ~1-1.5 hours")
    print("=" * 80 + "\n")
    
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
    print(f"Test F1 Macro: {test_results['eval_f1_macro']:.4f}")
    
    # Generate predictions for detailed report
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1) + 1
    y_true = np.array(test_labels) + 1
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Save best model
    best_model_path = os.path.join(args.output, 'best')
    os.makedirs(best_model_path, exist_ok=True)
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'=' * 80}")
    print(f"Training completed!")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Training time: {int(train_time // 60)}m {int(train_time % 60)}s")
    print(f"Best model saved to {best_model_path}/")
    print(f"{'=' * 80}")
    print(f"\nFast training optimizations:")
    print(f"  ✅ Reduced epochs (4 with early stopping)")
    print(f"  ✅ Larger batch size (32)")
    print(f"  ✅ Shorter context (256 tokens)")
    print(f"  ✅ Optimized tokenization (batched)")
    print(f"  ✅ Early stopping patience=2")

if __name__ == "__main__":
    main()

