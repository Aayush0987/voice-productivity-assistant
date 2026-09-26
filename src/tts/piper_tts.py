"""
TTS via Piper (fully local, no key/cost). Synthesizes text to audio and
plays it through the default output device. Supports streaming per-sentence
synthesis so playback can start before the full response text is ready.
"""
import re
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "tts" / "en_US-lessac-medium.onnx"

_voice = None

# Split on sentence-ending punctuation to allow incremental streaming to TTS.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(MODEL_PATH)
    return _voice


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Synthesize text to a float32 mono array. Returns (audio, sample_rate)."""
    voice = get_voice()
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.array([], dtype="float32"), 22050
    audio = np.concatenate([c.audio_float_array for c in chunks])
    return audio, chunks[0].sample_rate


def speak(text: str) -> None:
    """Synthesize and play the full text as one clip (blocking)."""
    audio, sample_rate = synthesize(text)
    if audio.size == 0:
        return
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def speak_streaming(text: str) -> None:
    """Synthesize sentence-by-sentence, playing each as soon as it's ready
    instead of waiting for the entire response to be synthesized first."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return
    for sentence in sentences:
        speak(sentence)


def speak_streaming_duplex(text: str, player) -> bool:
    """Synthesizes sentence-by-sentence and plays each through the given
    DuplexBargeInPlayer (src/vad/duplex_player.py), which monitors for
    barge-in via a single duplex audio stream instead of separate concurrent
    input/output streams. Returns True if interrupted."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    for sentence in sentences:
        if player.interrupted.is_set():
            return True
        audio, sample_rate = synthesize(sentence)
        if audio.size == 0:
            continue
        if player.play_with_monitoring(audio, sample_rate):
            return True
    return False


def speak_streaming_interruptible(text: str, interrupt_event: threading.Event) -> bool:
    """DEPRECATED for live use: opens a separate output stream while a
    separate MicMonitor input stream runs concurrently, which caused audible
    crackling and a segfault in live testing (see src/vad/duplex_player.py
    docstring). Kept only for reference; use speak_streaming_duplex instead.
    Same as speak_streaming, but checks `interrupt_event` between and
    during sentences and immediately stops playback if it's set (barge-in).
    Returns True if playback was cut short by an interruption."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    for sentence in sentences:
        if interrupt_event.is_set():
            return True

        audio, sample_rate = synthesize(sentence)
        if audio.size == 0:
            continue

        sd.play(audio, samplerate=sample_rate)
        stream = sd.get_stream()
        while stream is not None and stream.active:
            if interrupt_event.is_set():
                sd.stop()
                return True
            time.sleep(0.03)

    return False


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "Hello! This is a test of the Piper text to speech system."
    speak_streaming(text)
