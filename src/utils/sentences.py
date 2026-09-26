"""Incremental sentence splitting for streaming LLM output into TTS."""
import re

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx", "no",
}
_END = re.compile(r"([.!?]+[\"')\]]*)(\s+|$)")
MAX_CHUNK_CHARS = 200  # a run-on with no sentence end is split at a comma so speech can start


def _is_abbreviation(text_before_end: str) -> bool:
    last = re.split(r"\s+", text_before_end.strip())[-1].lower().rstrip(".")
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())  # "J. Smith", "U.S."


class SentenceChunker:
    """feed() text deltas as they stream in; complete sentences come back as soon
    as they are known to be complete. Call flush() at the end for the remainder."""

    def __init__(self):
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            sentence = self._next()
            if sentence is None:
                break
            out.append(sentence)
        return out

    def _next(self) -> str | None:
        buf = self._buf
        nl = buf.find("\n")
        for m in _END.finditer(buf):
            # a terminator at the very end of the buffer might be followed by more
            # text ("3." then "5"), so only trust it once whitespace confirms it
            if m.end() == len(buf) and not m.group(2):
                break
            if nl != -1 and nl < m.start():
                break
            if m.group(1).startswith("."):
                tail = buf[m.end():]
                if not tail:
                    break  # cannot tell yet: "p.m." then "yesterday" vs "done." then "Next"
                if tail[0].islower():
                    continue  # "5 p.m. yesterday", "U.S. is", "Wait... what": not a sentence end
                if _is_abbreviation(buf[: m.start()]):
                    continue  # "Dr. Smith", "J. Smith"
            return self._take(m.end())
        if nl != -1:
            return self._take(nl + 1)
        if len(buf) > MAX_CHUNK_CHARS:
            cut = buf.rfind(", ", 0, MAX_CHUNK_CHARS)
            if cut > 40:
                return self._take(cut + 2)
        return None

    def _take(self, n: int) -> str | None:
        sentence, self._buf = self._buf[:n].strip(), self._buf[n:]
        return sentence or self._next()

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


def split_sentences(text: str) -> list[str]:
    chunker = SentenceChunker()
    return chunker.feed(text + " ") + chunker.flush()
