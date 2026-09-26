"""Unit checks for the streaming sentence chunker, incl. token-by-token feeding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.sentences import SentenceChunker, split_sentences

CASES = [
    ("Hello there. How are you? I am fine!", ["Hello there.", "How are you?", "I am fine!"]),
    ("It is 3.5 degrees today. Bring a coat.", ["It is 3.5 degrees today.", "Bring a coat."]),
    ("Dr. Smith met Mr. Jones at 5 p.m. yesterday. They talked.",
     ["Dr. Smith met Mr. Jones at 5 p.m. yesterday.", "They talked."]),
    ("The U.S. is large. Really large.", ["The U.S. is large.", "Really large."]),
    ("Wait... what? Yes!", ["Wait... what?", "Yes!"]),
    ('He said "stop." Then he left.', ['He said "stop."', "Then he left."]),
    ("No punctuation at the end", ["No punctuation at the end"]),
    ("Line one\nLine two\nLine three", ["Line one", "Line two", "Line three"]),
    ("Got it, I'll remind you to call mom on Monday, September 28 at 6:00 PM.",
     ["Got it, I'll remind you to call mom on Monday, September 28 at 6:00 PM."]),
    ("", []),
]

failures = 0
for text, expected in CASES:
    whole = split_sentences(text)

    # same result when the text arrives a few characters at a time, as from a stream
    c, streamed = SentenceChunker(), []
    for i in range(0, len(text), 3):
        streamed += c.feed(text[i : i + 3])
    streamed += c.flush()

    ok = whole == expected and streamed == expected
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {text[:60]!r}")
    if not ok:
        print(f"   expected {expected}\n   whole    {whole}\n   streamed {streamed}")

# a long run-on with commas must still be split so speech can start
long_text = "This is a very long sentence, " * 12
chunks = split_sentences(long_text.strip())
ok = len(chunks) > 1 and all(len(x) <= 260 for x in chunks)
failures += not ok
print(f"[{'PASS' if ok else 'FAIL'}] long run-on split into {len(chunks)} chunks")

sys.exit(1 if failures else 0)
