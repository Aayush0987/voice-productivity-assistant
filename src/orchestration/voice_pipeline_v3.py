"""
Phase 6: microphone -> Silero VAD -> Whisper -> router -> Piper TTS -> speaker.
Full voice loop, still without barge-in (Phase 7).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.router.router import Router
from src.stt.whisper_stt import transcribe_array
from src.tts.piper_tts import speak_streaming
from src.vad.turn_taking import SAMPLE_RATE, record_utterance

if __name__ == "__main__":
    router = Router()
    print("Voice pipeline (Phase 6, full voice loop). Press Ctrl+C to exit.\n")
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
        speak_streaming(r.message)
