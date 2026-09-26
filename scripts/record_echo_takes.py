"""
Record real speaker->mic echo on THIS machine (stay silent while it runs) so
the barge-in detector can be tuned offline against actual hardware echo
instead of guesses. Saves (mic, playback) pairs to an .npz per take.

usage: python scripts/record_echo_takes.py OUT_DIR [N_TAKES]
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as sig
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tts.piper_tts import synthesize
from src.vad.turn_taking import CHUNK_SAMPLES, SAMPLE_RATE

TEXTS = [
    "A rainbow forms when sunlight passes through water droplets in the air at just the right angle. The light is refracted and separated into its different colors, which is why we see those bands of color in the sky after a rain shower.",
    "Right now in Mumbai it is twenty seven degrees with overcast skies. Today's high will be thirty one and the low twenty six degrees. Wind is around five kilometers per hour and humidity is eighty four percent.",
    "The capital of France is Paris. It has been the capital for many centuries and is famous for the Eiffel Tower, the Louvre museum, and its cafes. Millions of tourists visit every year.",
    "Got it, I will remind you to call mom on Friday at six in the evening. I have saved that reminder and I will let you know when the time comes so that you do not forget.",
    "Photosynthesis is how plants turn sunlight, water and carbon dioxide into food and oxygen. It happens mostly in the leaves, inside tiny structures called chloroplasts that contain chlorophyll.",
]


def record(text: str):
    """Record the way the real pipeline plays: one fresh duplex stream per
    sentence (no tail), a short gap between them, chunks concatenated."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    mics, refs = [], []
    for sentence in sentences:
        m, r = record_one(sentence)
        mics.append(m)
        refs.append(r)
        time.sleep(0.15)
    return np.concatenate(mics), np.concatenate(refs)


def record_one(text: str):
    audio, sr = synthesize(text)
    ref = sig.resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype("float32")
    total = len(ref)
    mics, refs, n = [], [], 0

    def cb(indata, outdata, frames, t, status):
        nonlocal n
        outdata[:] = 0
        take = max(0, min(frames, len(ref) - n))
        if take:
            outdata[:take, 0] = ref[n : n + take]
        n += frames
        mics.append(indata[:, 0].copy())
        refs.append(outdata[:, 0].copy())
        if n >= total:
            raise sd.CallbackStop()

    with sd.Stream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES, callback=cb) as s:
        while s.active:
            sd.sleep(50)
    return np.concatenate(mics), np.concatenate(refs)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    k = int(sys.argv[2]) if len(sys.argv) > 2 else len(TEXTS)
    for i, text in enumerate(TEXTS[:k]):
        mic, ref = record(text)
        np.savez(out / f"take{i}.npz", mic=mic, ref=ref)
        print(f"take{i}: {len(mic)/SAMPLE_RATE:.1f}s, mic rms {np.sqrt(np.mean(mic**2)):.3f}, ref rms {np.sqrt(np.mean(ref**2)):.3f}")
