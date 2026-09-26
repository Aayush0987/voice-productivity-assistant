"""
Load the fine-tuned intent classifier and predict on arbitrary text.
Usage:
  python src/classifier/predict.py                # interactive REPL
  python src/classifier/predict.py "some text"     # single prediction
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "intent_classifier"


def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    label = model.config.id2label[pred_id]
    return label, {model.config.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}


def main():
    tokenizer, model = load()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, probs = predict(text, tokenizer, model)
        print(f"{label}  {probs}")
        return

    print("Intent classifier REPL. Type a phrase, or 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        label, probs = predict(text, tokenizer, model)
        print(f"  -> {label}  {probs}")


if __name__ == "__main__":
    main()
