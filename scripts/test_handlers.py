"""Quick manual smoke test for all three Phase 2 handlers, run independently."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.handlers import qna, reminder, weather

TESTS = [
    ("qna", "what's the tallest mountain in the world"),
    ("reminder", "remind me to take my medication tomorrow at 9am"),
    ("weather", "what's the weather like in London"),
]

HANDLERS = {"qna": qna.handle, "reminder": reminder.handle, "weather": weather.handle}

if __name__ == "__main__":
    for intent, text in TESTS:
        print(f"[{intent}] {text}")
        result = HANDLERS[intent](text)
        print(f"  -> {result.message}\n")
