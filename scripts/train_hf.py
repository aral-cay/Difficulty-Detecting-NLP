import argparse
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/lecture_depth_train.csv")
    ap.add_argument("--val", default="data/processed/lecture_depth_val.csv")
    ap.add_argument("--test", default="data/processed/lecture_depth_test.csv")
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--output", default="models/distilbert_depth")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    args = ap.parse_args()

    print("Loading datasets...")
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Prepare data
    train_texts = train_df['text'].fillna('').astype(str).tolist()
    train_labels = (train_df['depth_level'].astype(int) - 1).tolist()  # Convert 1-3 to 0-2
    
    val_texts = val_df['text'].fillna('').astype(str).tolist()
    val_labels = (val_df['depth_level'].astype(int) - 1).tolist()  # Convert 1-3 to 0-2
    
    test_texts = test_df['text'].fillna('').astype(str).tolist()
    test_labels = (test_df['depth_level'].astype(int) - 1).tolist()  # Convert 1-3 to 0-2
    
    # Compute class weights for balancing
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"\nClass weights for balancing: {dict(zip(range(len(class_weights)), class_weights))}")
    
    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    num_labels = 3  # Fixed to 3 levels
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels
    )
    
    # Tokenize
    print("Tokenizing texts...")
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    
    train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        logging_steps=10,
        warmup_steps=100,
    )
    
    # Custom loss function with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    # Trainer with class balancing
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\nTraining model...")
    trainer.train()
    
    # Evaluate on test set
    print("\n=== Test Set Results ===")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
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
    print(f"\nBest model saved to {best_model_path}/")

if __name__ == "__main__":
    main()

