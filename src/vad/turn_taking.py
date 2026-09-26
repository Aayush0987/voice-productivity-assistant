"""
VAD-based turn-taking: record from the mic until the user actually stops
talking (detected via Silero VAD), instead of a fixed-duration window.

Replaces the Phase 4 fixed-5s recording, which was shown (via live testing)
to both cut sentences off mid-word and hallucinate transcripts from silence.
"""
import collections

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, load_silero_vad

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # ~32ms per chunk, required by the Silero VAD model
PRE_SPEECH_PAD_CHUNKS = 10  # ~320ms of audio kept from just before speech onset

_vad_model = None


def get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def record_utterance(
    max_seconds: float = 15.0,
    silence_timeout_s: float = 0.7,
    vad_threshold: float = 0.5,
) -> np.ndarray:
    """
    Blocks until the user finishes speaking (VAD detects `silence_timeout_s`
    of silence after speech) or `max_seconds` elapses. Returns the recorded
    speech as a float32 mono array at SAMPLE_RATE — empty if no speech was
    ever detected.
    """
    model = get_vad_model()
    vad_iterator = VADIterator(
        model,
        threshold=vad_threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=int(silence_timeout_s * 1000),
    )

    pre_buffer = collections.deque(maxlen=PRE_SPEECH_PAD_CHUNKS)
    collected = []
    speech_started = False
    max_chunks = int(max_seconds * SAMPLE_RATE / CHUNK_SAMPLES)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES
    ) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk = chunk.flatten()

            if not speech_started:
                pre_buffer.append(chunk)

            event = vad_iterator(torch.from_numpy(chunk), return_seconds=False)

            if event and "start" in event and not speech_started:
                speech_started = True
                collected.extend(pre_buffer)
                pre_buffer.clear()

            if speech_started:
                collected.append(chunk)

            if event and "end" in event and speech_started:
                break

    vad_iterator.reset_states()

    if not collected:
        return np.array([], dtype="float32")
    return np.concatenate(collected)


if __name__ == "__main__":
    print("Speak when ready (recording stops automatically after you pause)...")
    audio = record_utterance()
    print(f"Captured {len(audio) / SAMPLE_RATE:.2f}s of speech.")
