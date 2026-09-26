"""
Q&A handler: routes an open-ended question to a local LLM served by Ollama.

Kept independent from any other project's fine-tuned model — this is a
separate, general-purpose local model used only for open-ended Q&A.
"""
import json
from dataclasses import dataclass

import requests

from src.utils.sentences import SentenceChunker

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
KEEP_ALIVE = "30m"  # default is 5m: an idle pause would unload the model and cost ~6s on the next question

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Answer concisely and conversationally, "
    "as if speaking out loud. Avoid markdown, bullet points, or long lists — "
    "give a short, natural spoken-style answer, typically 1-3 sentences unless "
    "more detail is clearly needed."
)


@dataclass
class QnAResult:
    ok: bool
    message: str


def _messages(text: str, history: list[dict] | None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), {"role": "user", "content": text}]


class LLMStream:
    """Streams an answer from Ollama and yields it sentence by sentence as soon
    as each is complete, so TTS can start speaking while the model is still
    writing. cancel() closes the HTTP connection, which makes Ollama stop
    generating (a non-streamed request cannot be cancelled once sent)."""

    def __init__(self, text: str, history: list[dict] | None = None):
        self.text, self.history = text, history
        self.full_text = ""
        self._resp: requests.Response | None = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:  # noqa: BLE001 - closing a half-read socket may raise
                pass

    def _deltas(self):
        self._resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": _messages(self.text, self.history),
                  "stream": True, "keep_alive": KEEP_ALIVE},
            stream=True,
            timeout=(5, 60),
        )
        self._resp.raise_for_status()
        for line in self._resp.iter_lines():
            if self._cancelled:
                return
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                self.full_text += piece
                yield piece
            if chunk.get("done"):
                return

    def sentences(self):
        chunker = SentenceChunker()
        try:
            for piece in self._deltas():
                yield from chunker.feed(piece)
            yield from chunker.flush()
        except (requests.RequestException, ValueError):
            if not self._cancelled:
                yield "I couldn't reach the local language model."


def warm_up() -> bool:
    """Load the model into memory now, so the first real question is not a cold start."""
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                  "stream": False, "keep_alive": KEEP_ALIVE, "options": {"num_predict": 1}},
            timeout=120,
        ).raise_for_status()
        return True
    except requests.RequestException:
        return False


def handle(text: str, history: list[dict] | None = None) -> QnAResult:
    """`history` is prior chat turns ({'role','content'} dicts) so follow-ups
    like 'tell me that in short' have something to refer to."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": _messages(text, history),
                "stream": False,
                "keep_alive": KEEP_ALIVE,
            },
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return QnAResult(ok=False, message=f"I couldn't reach the local LLM: {e}")

    data = resp.json()
    answer = data.get("message", {}).get("content", "").strip()
    if not answer:
        return QnAResult(ok=False, message="I didn't get a response from the model.")

    return QnAResult(ok=True, message=answer)


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "what's the capital of France"
    result = handle(text)
    print(result.message)
