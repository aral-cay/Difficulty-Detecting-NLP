import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/distilbert_depth3/best")
    ap.add_argument("--text", required=True, help="Text to predict difficulty for")
    args = ap.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_level = predicted_class_idx + 1  # Convert 0-2 to 1-3
        probabilities = torch.softmax(logits, dim=1)[0]
    
    level_names = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    
    print(f"\nText: {args.text}")
    print(f"Predicted difficulty level: {predicted_level} ({level_names[predicted_level]})")
    print(f"\nProbabilities:")
    for i, prob in enumerate(probabilities):
        level = i + 1
        print(f"  Level {level} ({level_names[level]}): {prob.item():.4f}")

if __name__ == "__main__":
    main()

