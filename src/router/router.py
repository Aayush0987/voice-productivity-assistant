"""
Router: text input -> intent classifier -> correct handler -> response text.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classifier.predict import load as load_classifier
from src.classifier.predict import predict as classify
from src.handlers import qna, reminder, weather

HANDLERS = {
    "qna": qna.handle,
    "reminder": reminder.handle,
    "weather": weather.handle,
}


@dataclass
class RouteResult:
    intent: str
    confidence: float
    message: str
    latency_ms: float


class Router:
    def __init__(self):
        self.tokenizer, self.model = load_classifier()

    def route(self, text: str) -> RouteResult:
        start = time.perf_counter()
        intent, probs = classify(text, self.tokenizer, self.model)
        confidence = probs[intent]

        result = HANDLERS[intent](text)

        latency_ms = (time.perf_counter() - start) * 1000
        return RouteResult(
            intent=intent,
            confidence=confidence,
            message=result.message,
            latency_ms=latency_ms,
        )


if __name__ == "__main__":
    router = Router()
    print("Router REPL. Type a phrase, or 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        r = router.route(text)
        print(f"  [{r.intent} @ {r.confidence:.2f}, {r.latency_ms:.0f}ms] {r.message}\n")
