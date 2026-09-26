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
    classify_ms: float = 0.0
    handler_ms: float = 0.0


class Router:
    def __init__(self):
        self.tokenizer, self.model = load_classifier()

    def route(self, text: str) -> RouteResult:
        start = time.perf_counter()
        intent, probs = classify(text, self.tokenizer, self.model)
        confidence = probs[intent]
        classified = time.perf_counter()

        result = HANDLERS[intent](text)
        done = time.perf_counter()

        return RouteResult(
            intent=intent,
            confidence=confidence,
            message=result.message,
            latency_ms=(done - start) * 1000,
            classify_ms=(classified - start) * 1000,
            handler_ms=(done - classified) * 1000,
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
