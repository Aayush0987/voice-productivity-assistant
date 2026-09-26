"""
Router: text input -> intent classifier -> correct handler -> response text.
"""
import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classifier.predict import load as load_classifier
from src.classifier.predict import predict as classify
from src.handlers import qna, reminder, weather
from src.utils.sentences import split_sentences

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


@dataclass
class StreamedRoute:
    """A routing decision whose reply arrives sentence by sentence. Q&A yields
    sentences while the LLM is still writing; reminder/weather have already run
    their (fast) handler and just yield their sentences."""

    intent: str
    confidence: float
    classify_ms: float
    sentences: Callable[[], Iterator[str]]
    cancel: Callable[[], None]
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

    def _pick(self, text: str) -> tuple[str, float, str | None]:
        """(intent, confidence, city if this reply answers a 'which city?' prompt)."""
        answer = self._location_answer(text) if self.awaiting_location else None
        if answer:
            return "weather", 1.0, answer
        intent, probs = classify(text, self.tokenizer, self.model)
        return intent, probs[intent], None

    def route_stream(self, text: str) -> StreamedRoute:
        start = time.perf_counter()
        intent, confidence, answer = self._pick(text)
        classify_ms = (time.perf_counter() - start) * 1000

        if intent == "qna":
            stream = qna.LLMStream(text, self.history)
            return StreamedRoute(intent, confidence, classify_ms, stream.sentences, stream.cancel)

        if intent == "weather":
            result = weather.handle(text, default_location=answer or self.last_location)
        else:
            result = reminder.handle(text)
        sentences = split_sentences(result.message)
        return StreamedRoute(
            intent, confidence, classify_ms, lambda: iter(sentences), lambda: None,
            location=getattr(result, "location", None),
            needs_location=getattr(result, "needs_location", False),
        )

    def route(self, text: str) -> RouteResult:
        start = time.perf_counter()

        intent, confidence, answer = self._pick(text)
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
        self._remember(text, r.intent, r.message, r.location, r.needs_location)

    def remember_spoken(self, text: str, route: StreamedRoute, spoken: str) -> None:
        """Remember only what the user actually heard (a reply interrupted after
        two sentences is remembered as those two sentences)."""
        if spoken.strip():
            self._remember(text, route.intent, spoken, route.location, route.needs_location)

    def _remember(self, text, intent, message, location, needs_location) -> None:
        self.history += [
            {"role": "user", "content": text[:MAX_HISTORY_CHARS]},
            {"role": "assistant", "content": message[:MAX_HISTORY_CHARS]},
        ]
        self.history = self.history[-MAX_HISTORY_MESSAGES:]
        if intent == "weather" and location and not needs_location:
            self.last_location = location
        self.awaiting_location = intent == "weather" and needs_location


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
