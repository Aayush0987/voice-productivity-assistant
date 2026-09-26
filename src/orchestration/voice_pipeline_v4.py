"""
Phase 7: full voice loop + barge-in. While the assistant is "thinking"
(handler/LLM call) or speaking (TTS), a background MicMonitor listens for the
user starting to talk. If detected, the in-flight response is discarded (or
playback is stopped immediately) and the newly-captured utterance is routed
next — no need to record it twice.
"""
import concurrent.futures
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.router.router import Router
from src.stt.whisper_stt import transcribe_array
from src.tts.piper_tts import speak_streaming_duplex
from src.vad.barge_in import MicMonitor
from src.vad.duplex_player import DuplexBargeInPlayer
from src.vad.turn_taking import SAMPLE_RATE, record_utterance

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def think_with_barge_in(router: Router, transcript: str):
    """Runs router.route() in the background while a MicMonitor listens for
    an interruption. Returns ('interrupted', audio) or ('done', RouteResult)."""
    monitor = MicMonitor()
    monitor.start()
    future = executor.submit(router.route, transcript)

    while not future.done():
        if monitor.interrupted.is_set():
            audio = monitor.wait_for_capture(timeout=20.0)
            print("  [barge-in] interrupted while thinking — discarding in-flight response")
            return "interrupted", audio
        time.sleep(0.02)

    monitor.stop()
    return "done", future.result()


def speak_with_barge_in(message: str):
    """Plays the response with barge-in monitoring via a single duplex audio
    stream (src/vad/duplex_player.py). Returns ('interrupted', audio) or
    ('done', None)."""
    player = DuplexBargeInPlayer()
    was_interrupted = speak_streaming_duplex(message, player)

    if was_interrupted:
        audio = player.wait_for_capture(timeout=20.0)
        print("  [barge-in] interrupted mid-speech — stopping playback")
        return "interrupted", audio

    return "done", None


def main():
    router = Router()
    print("Voice pipeline (Phase 7, barge-in enabled). Press Ctrl+C to exit.\n")

    pending_audio = None
    while True:
        try:
            if pending_audio is not None and pending_audio.size > 0:
                audio = pending_audio
            else:
                print("Listening... (speak, then pause when you're done)")
                audio = record_utterance()
            pending_audio = None
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

        status, result = think_with_barge_in(router, transcript)
        if status == "interrupted":
            pending_audio = result
            print()
            continue

        r = result
        print(f"  [{r.intent} @ {r.confidence:.2f}, {r.latency_ms:.0f}ms] {r.message}")

        status, result = speak_with_barge_in(r.message)
        router.remember(transcript, r)  # the user heard it (fully or partly)
        if status == "interrupted":
            pending_audio = result
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    # PortAudio/torch threads can abort during normal interpreter teardown
    # (seen as a segfault / 'recursive_mutex lock failed'); skip it.
    print("\nBye.", flush=True)
    os._exit(0)
