"""
Q&A handler: routes an open-ended question to a local LLM served by Ollama.

Kept independent from any other project's fine-tuned model — this is a
separate, general-purpose local model used only for open-ended Q&A.
"""
from dataclasses import dataclass

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"

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


def handle(text: str) -> QnAResult:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
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
