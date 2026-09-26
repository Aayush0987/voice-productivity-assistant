"""
Offline per-stage latency benchmark (no live mic needed).

Speech is synthesized with macOS `say`, then pushed through
STT -> classify -> handler -> TTS synthesis. Reports median ms per stage
per intent. Excludes the fixed VAD end-of-speech wait (silence timeout)
and audio playback time, which are constants, not compute.
"""
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

from src.router.router import Router
from src.stt.whisper_stt import transcribe_array
from src.tts.piper_tts import synthesize
from src.utils.latency import log_turn

PHRASES = {
    "qna": [
        "what is the capital of Australia",
        "how does a rainbow form",
        "who wrote Pride and Prejudice",
        "why is the sky blue",
    ],
    "reminder": [
        "remind me to call mom tomorrow at 6 pm",
        "remind me to submit the report on Friday at noon",
        "remind me to water the plants tomorrow at 9 am",
        "remind me to book the dentist next Monday at 10 am",
    ],
    "weather": [
        "what is the weather in Mumbai",
        "will it rain in London tomorrow",
        "how is the weather in Tokyo right now",
        "what is the temperature in Berlin",
    ],
}


def speak_to_array(text: str, workdir: Path) -> np.ndarray:
    aiff, wav = workdir / "u.aiff", workdir / "u.wav"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
        check=True,
    )
    audio, _ = sf.read(str(wav), dtype="float32")
    return audio


def main():
    router = Router()
    rows = {intent: {"stt": [], "classify": [], "handler": [], "tts": []} for intent in PHRASES}

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        # warm-up so model load/first-call costs don't pollute results
        transcribe_array(speak_to_array("hello there", workdir))
        router.route("what is two plus two")
        synthesize("warm up.")

        for intent, phrases in PHRASES.items():
            for phrase in phrases:
                audio = speak_to_array(phrase, workdir)

                t = time.perf_counter()
                transcript = transcribe_array(audio)
                stt_ms = (time.perf_counter() - t) * 1000

                r = router.route(transcript)

                first_sentence = r.message.split(". ")[0]
                t = time.perf_counter()
                synthesize(first_sentence)
                tts_ms = (time.perf_counter() - t) * 1000

                rec = {
                    "expected": intent,
                    "routed": r.intent,
                    "transcript": transcript,
                    "stt_ms": stt_ms,
                    "classify_ms": r.classify_ms,
                    "handler_ms": r.handler_ms,
                    "tts_first_sentence_ms": tts_ms,
                }
                log_turn(rec)
                bucket = rows[intent]
                bucket["stt"].append(stt_ms)
                bucket["classify"].append(r.classify_ms)
                bucket["handler"].append(r.handler_ms)
                bucket["tts"].append(tts_ms)
                flag = "" if r.intent == intent else f"  (routed to {r.intent}!)"
                print(f"[{intent}] {transcript!r}{flag}")

    print("\nMedian latency per stage (ms), n=%d per intent" % len(PHRASES["qna"]))
    print(f"{'intent':<10}{'STT':>8}{'classify':>10}{'handler':>10}{'TTS(1st sent)':>15}{'total':>9}")
    for intent, b in rows.items():
        med = {k: statistics.median(v) for k, v in b.items()}
        total = sum(med.values())
        print(
            f"{intent:<10}{med['stt']:>8.0f}{med['classify']:>10.0f}"
            f"{med['handler']:>10.0f}{med['tts']:>15.0f}{total:>9.0f}"
        )


if __name__ == "__main__":
    main()
