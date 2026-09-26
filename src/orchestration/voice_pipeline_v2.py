"""
Phase 5: microphone -> Silero VAD (end-of-speech detection) -> Whisper ->
transcript -> router. Replaces the Phase 4 fixed-duration recording with
real turn-taking: it waits for you to actually stop talking.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.router.router import Router
from src.stt.whisper_stt import transcribe_array
from src.vad.turn_taking import SAMPLE_RATE, record_utterance

if __name__ == "__main__":
    router = Router()
    print("Voice pipeline (Phase 5, VAD turn-taking). Press Ctrl+C to exit.\n")
    while True:
        try:
            print("Listening... (speak, then pause when you're done)")
            audio = record_utterance()
        except KeyboardInterrupt:
            break

        if audio.size == 0:
            print("  (no speech detected, try again)\n")
            continue

        duration = len(audio) / SAMPLE_RATE
        transcript = transcribe_array(audio)
        if not transcript:
            print(f"  (captured {duration:.1f}s but got an empty transcript, try again)\n")
            continue

        print(f'  Heard ({duration:.1f}s): "{transcript}"')
        r = router.route(transcript)
        print(f"  [{r.intent} @ {r.confidence:.2f}, {r.latency_ms:.0f}ms] {r.message}\n")
