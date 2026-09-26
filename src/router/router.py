"""
Router: text input -> intent classifier -> correct handler -> response text.
"""
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classifier.predict import load as load_classifier
from src.classifier.predict import predict as classify
from src.handlers import qna, reminder, weather

MAX_HISTORY_MESSAGES = 8  # last 4 exchanges: enough for follow-ups, keeps the LLM prompt short
MAX_HISTORY_CHARS = 400
_LOCATION_LEAD = re.compile(r"^(?:it'?s|its|in|for|the city is|city of|i said|i meant)\s+", re.IGNORECASE)


@dataclass
class RouteResult:
    intent: str
    confidence: float
    message: str
    latency_ms: float
    classify_ms: float = 0.0
    handler_ms: float = 0.0
    location: str | None = None
    needs_location: bool = False


class Router:
    """Routes text to a handler and carries a little session memory:
    chat history for Q&A follow-ups, the last weather city, and whether the
    assistant just asked "which city?" (so a bare "Mumbai" is the answer).

    route() only READS that state. Call remember() once the user has actually
    heard the response, so an answer discarded by a barge-in never becomes
    part of the conversation."""

    def __init__(self):
        self.tokenizer, self.model = load_classifier()
        self.history: list[dict] = []
        self.last_location: str | None = None
        self.awaiting_location = False

    @staticmethod
    def _location_answer(text: str) -> str | None:
        cleaned = _LOCATION_LEAD.sub("", text.strip()).strip(" .,!?")
        words = cleaned.split()
        return " ".join(w.capitalize() for w in words) if 1 <= len(words) <= 4 else None

    def route(self, text: str) -> RouteResult:
        start = time.perf_counter()

        answer = self._location_answer(text) if self.awaiting_location else None
        if answer:
            intent, confidence = "weather", 1.0
        else:
            intent, probs = classify(text, self.tokenizer, self.model)
            confidence = probs[intent]
        classified = time.perf_counter()

        if intent == "qna":
            result = qna.handle(text, history=self.history)
        elif intent == "weather":
            result = weather.handle(text, default_location=answer or self.last_location)
        else:
            result = reminder.handle(text)
        done = time.perf_counter()

        return RouteResult(
            intent=intent,
            confidence=confidence,
            message=result.message,
            latency_ms=(done - start) * 1000,
            classify_ms=(classified - start) * 1000,
            handler_ms=(done - classified) * 1000,
            location=getattr(result, "location", None),
            needs_location=getattr(result, "needs_location", False),
        )

    def remember(self, text: str, r: RouteResult) -> None:
        self.history += [
            {"role": "user", "content": text[:MAX_HISTORY_CHARS]},
            {"role": "assistant", "content": r.message[:MAX_HISTORY_CHARS]},
        ]
        self.history = self.history[-MAX_HISTORY_MESSAGES:]
        if r.intent == "weather" and r.location and not r.needs_location:
            self.last_location = r.location
        self.awaiting_location = r.intent == "weather" and r.needs_location


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
        router.remember(text, r)
        print(f"  [{r.intent} @ {r.confidence:.2f}, {r.latency_ms:.0f}ms] {r.message}\n")
