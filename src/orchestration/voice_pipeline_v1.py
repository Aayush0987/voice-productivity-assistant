"""
Phase 4: microphone -> Whisper -> transcript -> router (no VAD/TTS/barge-in yet).
Fixed-duration recording per turn; VAD-based turn-taking comes in Phase 5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.router.router import Router
from src.stt.whisper_stt import record_and_transcribe

RECORD_SECONDS = 5.0

if __name__ == "__main__":
    router = Router()
    print("Voice pipeline (Phase 4). Press Ctrl+C to exit.\n")
    while True:
        try:
            transcript = record_and_transcribe(RECORD_SECONDS)
        except KeyboardInterrupt:
            break
        if not transcript:
            print("  (heard nothing, try again)\n")
            continue
        print(f"  Heard: \"{transcript}\"")
        r = router.route(transcript)
        print(f"  [{r.intent} @ {r.confidence:.2f}, {r.latency_ms:.0f}ms] {r.message}\n")
