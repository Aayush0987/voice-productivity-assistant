"""
Tune DoubleTalkDetector against REAL recorded echo (scripts/record_echo_takes.py)
with a simulated user voice mixed in at chosen loudness.

Objective: zero false triggers on echo-only takes, then maximize how many
(take, user-loudness) cases are caught, then minimize detection delay.
The VAD gate is omitted: the assistant's own voice scores p~1.0 anyway (seen
in live debug logs), so it never helps discriminate; only the energy test does.

usage: python scripts/tune_barge_in.py TAKES_DIR
"""
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vad.barge_in_engine import DoubleTalkDetector
from src.vad.turn_taking import CHUNK_SAMPLES, SAMPLE_RATE

USER_START_S = 3.0
USER_RMS_LEVELS = (0.06, 0.10, 0.15)  # active-speech RMS at the mic, vs ~0.04 echo RMS
USER_TEXTS = ["wait stop, what is the weather in Mumbai", "no no, remind me to call mom instead"]


def user_voices(workdir: Path):
    voices = []
    for i, text in enumerate(USER_TEXTS):
        aiff, wav = workdir / f"u{i}.aiff", workdir / f"u{i}.wav"
        subprocess.run(["say", "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
        audio, _ = sf.read(str(wav), dtype="float32")
        voices.append(audio)
    return voices


def active_rms(x):
    a = x[np.abs(x) > 0.1 * np.abs(x).max()]
    return float(np.sqrt(np.mean(a**2)))


def first_trigger(mic, ref, cfg):
    det = DoubleTalkDetector(**cfg["det"])
    run = 0
    for i in range(0, len(mic) - CHUNK_SAMPLES, CHUNK_SAMPLES):
        near = det.process(mic[i : i + CHUNK_SAMPLES], ref[i : i + CHUNK_SAMPLES])
        run = run + 1 if near else 0
        if run >= cfg["consec"]:
            return i / SAMPLE_RATE
    return None


def with_clicks(mic, seed):
    """Add single-chunk bursts (clicks/knocks, RMS ~0.3-0.4) to echo-only audio."""
    rng = np.random.default_rng(seed)
    out = mic.copy()
    for _ in range(4):
        pos = int(rng.integers(SAMPLE_RATE, len(mic) - CHUNK_SAMPLES * 2))
        burst = rng.normal(0, rng.uniform(0.3, 0.45), CHUNK_SAMPLES).astype("float32")
        out[pos : pos + CHUNK_SAMPLES] += burst
    return out


def evaluate(cfg, takes, voices):
    echo_only = list(takes) + [(with_clicks(m, i), r) for i, (m, r) in enumerate(takes)]
    false_triggers = sum(first_trigger(m, r, cfg) is not None for m, r in echo_only)
    caught = total = 0
    delays = []
    early = 0
    for (m, r), v in itertools.product(takes, voices):
        for level in USER_RMS_LEVELS:
            mixed = m.copy()
            start = int(USER_START_S * SAMPLE_RATE)
            seg = v[: len(m) - start]
            seg = seg * (level / active_rms(seg))
            mixed[start : start + len(seg)] += seg
            t = first_trigger(mixed, r, cfg)
            total += 1
            if t is None:
                continue
            if t < USER_START_S:
                early += 1  # would have false-triggered anyway
                continue
            caught += 1
            delays.append((t - USER_START_S) * 1000)
    return false_triggers, caught, total, (float(np.median(delays)) if delays else None)


def main():
    takes = []
    for f in sorted(Path(sys.argv[1]).glob("take*.npz")):
        d = np.load(f)
        takes.append((d["mic"], d["ref"]))
    with tempfile.TemporaryDirectory() as tmp:
        voices = user_voices(Path(tmp))

    grid = {
        "margin": (1.3, 1.6, 2.0),
        "quantile": (0.8, 0.9),
        "smooth": (1,),
        "ref_lag": ((0, 12), (2, 8)),
        "consec": (5, 7, 9),
    }
    results = []
    for margin, q, sm, lag, consec in itertools.product(*grid.values()):
        cfg = {"det": dict(margin=margin, quantile=q, smooth=sm, ref_lag=lag), "consec": consec}
        fa, caught, total, delay = evaluate(cfg, takes, voices)
        results.append((fa, -caught, delay or 1e9, margin, q, sm, lag, consec, total))

    results.sort()
    print(f"{len(takes)} real echo takes (+{len(takes)} with injected clicks); {results[0][-1]} user-mix cases per config\n")
    print(f"{'false':>5} {'caught':>7} {'delay_ms':>9}  margin  quant  smooth  ref_lag   consec")
    for fa, negc, delay, margin, q, sm, lag, consec, total in results[:12]:
        d = f"{delay:.0f}" if delay < 1e8 else "-"
        print(f"{fa:>5} {-negc:>3}/{total:<3} {d:>9}  {margin:>6} {q:>6} {sm:>7}  {str(lag):<9} {consec}")


if __name__ == "__main__":
    main()
