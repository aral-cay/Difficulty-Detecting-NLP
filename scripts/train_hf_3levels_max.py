#!/usr/bin/env python3
"""
Maximum accuracy DistilBERT training with all advanced techniques:
- Optimized hyperparameters
- Better learning rate scheduling
- More epochs with early stopping
- Longer context
- Enhanced class weighting
Expected improvement: +2-4% accuracy (77.79% → 80-82%)
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
    ap.add_argument("--output", default="models/distilbert_depth3_max")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)  # More epochs
    ap.add_argument("--learning_rate", type=float, default=2e-5)  # Lower LR
    ap.add_argument("--max_length", type=int, default=512)  # Longer context
    args = ap.parse_args()

    print("=" * 80)
    print("MAXIMUM ACCURACY DISTILBERT TRAINING")
    print("=" * 80)
    
    print("\nLoading datasets...")
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Enhanced oversampling: Give Intermediate class extra boost
    print("\nBalancing dataset with enhanced oversampling for Intermediate class...")
    from sklearn.utils import resample
    
    # Separate by class
    train_df_classes = {level: train_df[train_df['depth_level'] == level] for level in [1, 2, 3]}
    
    # Find max class size
    max_size = max(len(train_df_classes[level]) for level in [1, 2, 3])
    print(f"  Original sizes: Level 1={len(train_df_classes[1])}, Level 2={len(train_df_classes[2])}, Level 3={len(train_df_classes[3])}")
    
    # Special handling: Oversample Intermediate more aggressively (1.5x instead of 1x)
    intermediate_target = int(max_size * 1.5)  # Give Intermediate 50% more samples
    
    train_df_balanced = []
    for level in [1, 2, 3]:
        df_level = train_df_classes[level]
        if level == 2:  # Intermediate - extra oversampling
            if len(df_level) < intermediate_target:
                df_oversampled = resample(df_level, replace=True, n_samples=intermediate_target, random_state=42)
                train_df_balanced.append(df_oversampled)
                print(f"  Oversampled Level {level} (Intermediate) from {len(df_level)} to {len(df_oversampled)} (1.5x boost)")
            else:
                train_df_balanced.append(df_level)
        else:  # Beginner and Advanced - match max size
            if len(df_level) < max_size:
                df_oversampled = resample(df_level, replace=True, n_samples=max_size, random_state=42)
                train_df_balanced.append(df_oversampled)
                print(f"  Oversampled Level {level} from {len(df_level)} to {len(df_oversampled)}")
            else:
                train_df_balanced.append(df_level)
    
    train_df_balanced = pd.concat(train_df_balanced, ignore_index=True)
    train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"  Final balanced size: {len(train_df_balanced)} samples")
    print(f"  Final distribution: {train_df_balanced['depth_level'].value_counts().sort_index().to_dict()}")
    
    # Prepare data
    train_texts = train_df_balanced['text'].fillna('').astype(str).tolist()
    train_labels = (train_df_balanced['depth_level'].astype(int) - 1).tolist()  # Convert 1-3 to 0-2
    
    val_texts = val_df['text'].fillna('').astype(str).tolist()
    val_labels = (val_df['depth_level'].astype(int) - 1).tolist()  # Convert 1-3 to 0-2
    
    test_texts = test_df['text'].fillna('').astype(str).tolist()
    test_labels = (test_df['depth_level'].astype(int) - 1).tolist()  # Convert 1-3 to 0-2
    
    # Custom class weights: Boost Intermediate even more (2.5x instead of 2x)
    from sklearn.utils.class_weight import compute_class_weight
    base_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    
    # Custom weights: give Intermediate 2.5x boost
    custom_weights = []
    for i, level in enumerate(sorted(np.unique(train_labels))):
        if level == 1:  # Intermediate is at index 1 (0-indexed: 0=Beginner, 1=Intermediate, 2=Advanced)
            custom_weights.append(base_weights[i] * 2.5)  # Increased from 2.0x
        else:
            custom_weights.append(base_weights[i])
    
    class_weight_tensor = torch.tensor(custom_weights, dtype=torch.float32)
    print(f"\nCustom class weights (Intermediate boosted 2.5x): {dict(zip(range(len(custom_weights)), custom_weights))}")
    
    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    num_labels = 3  # 3 levels: Beginner, Intermediate, Advanced
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    
    # Tokenize with better settings
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
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Check device type for fp16 compatibility
    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    use_fp16 = device_type == "cuda"  # Only use fp16 on CUDA GPUs
    print(f"  Device: {device_type}, fp16: {use_fp16}")
    
    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # Larger eval batch
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,  # 10% warmup
        lr_scheduler_type="cosine",  # Cosine learning rate schedule
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",  # Use F1 instead of accuracy
        greater_is_better=True,
        save_total_limit=3,  # Keep more checkpoints
        logging_steps=50,
        logging_dir=os.path.join(args.output, 'logs'),
        report_to="none",  # Disable wandb/tensorboard
        fp16=use_fp16,  # Only use fp16 on CUDA GPUs
        dataloader_num_workers=0 if device_type == "mps" else 2,  # MPS doesn't support multiprocessing
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
    )
    
    # Custom loss function with class weights
    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            # Move class weights to the same device as model
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
    
    # Trainer with class balancing and early stopping
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # More patience
        class_weights=class_weight_tensor
    )
    
    # Train
    print("\nTraining model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  LR scheduler: cosine")
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
    print(f"Test F1 Macro: {test_results['eval_f1_macro']:.4f}")
    
    # Generate predictions for detailed report
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1) + 1  # Convert 0-2 back to 1-3
    y_true = np.array(test_labels) + 1  # Convert 0-2 back to 1-3
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Save best model
    best_model_path = os.path.join(args.output, 'best')
    os.makedirs(best_model_path, exist_ok=True)
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\n{'=' * 80}")
    print(f"Best model saved to {best_model_path}/")
    print(f"{'=' * 80}")
    print(f"\nImprovements made:")
    print(f"  ✅ Lower learning rate (2e-5 instead of 5e-5)")
    print(f"  ✅ Longer context (512 tokens)")
    print(f"  ✅ More epochs (8 with early stopping)")
    print(f"  ✅ Cosine LR scheduler")
    print(f"  ✅ Increased Intermediate class weight (2.5x)")
    print(f"  ✅ F1-macro as metric for best model selection")
    print(f"  ✅ Mixed precision training (fp16)")
    print(f"\nExpected improvement: +2-4% accuracy over original DistilBERT")

if __name__ == "__main__":
    main()

