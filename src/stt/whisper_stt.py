"""
STT via faster-whisper. Two entry points:
  - transcribe_file(path): transcribe an existing audio file (any format ffmpeg/
    faster-whisper can read) — used for testing without live mic input.
  - record_and_transcribe(seconds): record from the default microphone and
    transcribe it.
"""
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = "base.en"
SAMPLE_RATE = 16000

_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_file(path: str | Path) -> str:
    model = get_model()
    segments, _ = model.transcribe(str(path), language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe_array(audio: np.ndarray) -> str:
    """Transcribe a float32 mono array at SAMPLE_RATE. Empty input -> empty string,
    skipping Whisper entirely (it can hallucinate filler text on silence)."""
    if audio.size == 0:
        return ""
    model = get_model()
    segments, _ = model.transcribe(audio, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def record_and_transcribe(seconds: float = 5.0) -> str:
    """Fixed-duration recording (Phase 4 baseline). Superseded by VAD-based
    turn-taking (src/vad/turn_taking.py, Phase 5) which avoids cutting
    sentences off mid-word and skips hallucination-prone silent captures."""
    import sounddevice as sd

    print(f"Recording for {seconds}s... speak now.")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return transcribe_array(audio.flatten())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(transcribe_file(sys.argv[1]))
    else:
        print(record_and_transcribe())
