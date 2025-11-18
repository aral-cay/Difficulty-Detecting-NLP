import sys, joblib

from pathlib import Path

# Load sklearn model (vectorizer + classifier)
vectorizer = joblib.load("models/tfidf_logreg_3level/vectorizer.joblib")
classifier = joblib.load("models/tfidf_logreg_3level/classifier.joblib")

text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "This lecture introduces linear models and gradient descent."

# Predict
vec = vectorizer.transform([text])
pred = int(classifier.predict(vec)[0])  # already 1..3

level_names = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
print(f"Sklearn predicted difficulty level: {pred} ({level_names[pred]})")
print(f"Text: {text[:100]}...")

