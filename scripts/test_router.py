import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router.router import Router

TEST_PHRASES = [
    # clear-cut cases
    "what's the capital of Japan",
    "remind me to pick up the dry cleaning tomorrow",
    "what's the weather in Berlin",
    # ambiguous / edge cases
    "should I bring an umbrella today",
    "remind me to check the weather before I leave",
    "what's a good reminder app",
    "hey what's up",
    "set a timer for 10 minutes",
    "will it be sunny for my picnic on Saturday",
    "add buy groceries to my list",
]

if __name__ == "__main__":
    router = Router()
    for text in TEST_PHRASES:
        r = router.route(text)
        print(f"'{text}'")
        print(f"  -> [{r.intent} @ {r.confidence:.2f}, {r.latency_ms:.0f}ms] {r.message}\n")
