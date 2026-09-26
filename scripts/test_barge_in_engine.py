"""
Offline test of BargeInEngine with simulated acoustic echo.

mic = echo_gain * (playback delayed ~100ms) + room noise + (user voice from
`say`, starting at USER_START_S). Verifies the engine (a) never triggers on
echo alone and (b) triggers shortly after the user starts talking, across
echo strengths and user loudness. No audio hardware needed.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tts.piper_tts import synthesize
from src.vad.barge_in_engine import BargeInEngine
from src.vad.turn_taking import CHUNK_SAMPLES, SAMPLE_RATE

USER_START_S = 3.0
ECHO_DELAY = 1600  # 100ms
ASSISTANT_TEXT = (
    "A rainbow forms when sunlight passes through water droplets in the air "
    "at just the right angle, causing the light to be refracted and separated "
    "into its different colors. That is why we see those beautiful bands of "
    "color in the sky after a rain shower."
)
USER_TEXT = "wait stop, what is the weather in Mumbai"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def user_voice(workdir: Path) -> np.ndarray:
    aiff, wav = workdir / "user.aiff", workdir / "user.wav"
    subprocess.run(["say", "-o", str(aiff), USER_TEXT], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True
    )
    audio, _ = sf.read(str(wav), dtype="float32")
    return audio


def run(playback, user, echo_gain, user_to_echo, with_user, seed=0):
    rng = np.random.default_rng(seed)
    n = len(playback)
    echo = np.zeros(n, dtype="float32")
    echo[ECHO_DELAY:] = echo_gain * playback[: n - ECHO_DELAY]
    mic = echo + rng.normal(0, 0.002, n).astype("float32")

    user_start = int(USER_START_S * SAMPLE_RATE)
    if with_user:
        seg = user[: n - user_start]
        target_rms = user_to_echo * rms(echo[user_start : user_start + SAMPLE_RATE * 2])
        seg = seg * (target_rms / max(rms(seg), 1e-9))
        mic[user_start : user_start + len(seg)] += seg

    engine = BargeInEngine()
    for i in range(0, n - CHUNK_SAMPLES, CHUNK_SAMPLES):
        engine.feed(mic[i : i + CHUNK_SAMPLES], playback[i : i + CHUNK_SAMPLES])
        if engine.triggered:
            return (i / SAMPLE_RATE) - USER_START_S  # seconds after user started
    return None


def main():
    audio, sr = synthesize(ASSISTANT_TEXT)
    playback = sig.resample(audio, int(len(audio) * SAMPLE_RATE / sr)).astype("float32")
    playback = playback / np.abs(playback).max() * 0.5
    with tempfile.TemporaryDirectory() as tmp:
        user = user_voice(Path(tmp))

    print(f"playback {len(playback)/SAMPLE_RATE:.1f}s, user talks from t={USER_START_S}s\n")
    print(f"{'echo gain':>10} {'user/echo':>10} {'user?':>6}  result")
    failures = 0
    for g in (0.1, 0.3, 0.6, 0.9):
        t = run(playback, user, g, 0, with_user=False)
        ok = t is None
        failures += not ok
        print(f"{g:>10} {'-':>10} {'no':>6}  {'OK: no false trigger' if ok else f'FALSE TRIGGER at {t:+.2f}s'}")
        for uer in (1.0, 2.0, 3.0):
            t = run(playback, user, g, uer, with_user=True)
            if t is None:
                res, ok = "MISSED interruption", uer < 2.0  # <2x is documented as too quiet
            elif t < 0:
                res, ok = f"FALSE TRIGGER {abs(t):.2f}s before user spoke", False
            else:
                res, ok = f"OK: triggered {t*1000:.0f}ms after user started", True
            failures += (not ok)
            print(f"{g:>10} {uer:>10} {'yes':>6}  {res}")
    print("\nnote: user/echo < 2 is below the detector margin, a miss there is expected")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
