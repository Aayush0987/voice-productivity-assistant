import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "classifier"

GREETINGS = [
    "hey what's up",
    "hi there",
    "hello, how are you",
    "good morning",
    "what's new",
    "how's it going",
    "hey, you there",
    "yo",
    "sup",
    "good evening, how are things",
    "hiya",
    "anything interesting happening",
    "how are you doing today",
    "what's going on",
    "hey, got a minute",
    "hey buddy",
    "howdy",
    "what's happening",
    "hey, can we chat",
    "good to see you",
]

all_examples = json.load(open(DATA_DIR / "all_examples.json"))
existing = set(p.lower() for p in all_examples["qna"])
added = 0
for g in GREETINGS:
    if g.lower() not in existing:
        all_examples["qna"].append(g)
        added += 1

json.dump(all_examples, open(DATA_DIR / "all_examples.json", "w"), indent=2)
print(f"Added {added} greeting examples to qna (new total: {len(all_examples['qna'])})")

train, test = [], []
for intent, phrases in all_examples.items():
    phrases = list(phrases)
    random.shuffle(phrases)
    split = int(len(phrases) * 0.85)
    for p in phrases[:split]:
        train.append({"text": p, "label": intent})
    for p in phrases[split:]:
        test.append({"text": p, "label": intent})

random.shuffle(train)
random.shuffle(test)

json.dump(train, open(DATA_DIR / "train.json", "w"), indent=2)
json.dump(test, open(DATA_DIR / "test.json", "w"), indent=2)
print(f"train={len(train)} test={len(test)}")
