"""
Phase 7 + streaming: like voice_pipeline_v4 (barge-in, memory) but the reply is
STREAMED. The LLM writes sentence 1 while sentence 0 is already being spoken,
so time-to-first-word no longer scales with answer length. Barge-in listens the
whole time, including the gaps while waiting for the next sentence, and an
interruption cancels the LLM request so Ollama stops generating (a discarded
non-streamed answer keeps running and blocks the next request; measured 57 s
vs 1.1 s).

Threads: a producer thread streams the LLM, splits it into sentences and
synthesizes each with Piper; the main thread plays them one after another.
"""
import os
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.handlers import qna
from src.router.router import Router
from src.stt.whisper_stt import transcribe_array
from src.tts.piper_tts import synthesize
from src.utils.latency import log_turn
from src.vad.barge_in import MicMonitor
from src.vad.duplex_player import DuplexBargeInPlayer
from src.vad.turn_taking import SAMPLE_RATE, record_utterance


def start_producer(router, transcript, cancelled, out_q, holder):
    """Route, stream sentences, synthesize each, push (sentence, audio, rate).
    Always ends by pushing None."""

    def run():
        try:
            route = router.route_stream(transcript)
            holder["route"] = route
            if cancelled.is_set():
                route.cancel()
                return
            for sentence in route.sentences():
                if cancelled.is_set():
                    route.cancel()
                    return
                audio, rate = synthesize(sentence)
                out_q.put((sentence, audio, rate))
        except Exception as exc:  # noqa: BLE001 - never leave the consumer waiting
            if not cancelled.is_set():
                print(f"  [error] {type(exc).__name__}: {exc}")
        finally:
            out_q.put(None)

    threading.Thread(target=run, daemon=True).start()


def next_item(out_q):
    """Next (sentence, audio, rate) or None when the reply is finished, while
    listening for the user. Returns (item, None) or (None, captured_audio_if_interrupted)."""
    try:
        return out_q.get_nowait(), None
    except queue.Empty:
        pass

    monitor = MicMonitor()
    monitor.start()
    try:
        while True:
            try:
                item = out_q.get(timeout=0.05)
            except queue.Empty:
                if monitor.interrupted.is_set():
                    return None, monitor.wait_for_capture(timeout=20.0)
                continue
            if monitor.interrupted.is_set():  # both at once: the user wins
                return None, monitor.wait_for_capture(timeout=20.0)
            return item, None
    finally:
        monitor.stop()


def respond(router, transcript, t0, stt_ms):
    """Speak the reply to `transcript`. Returns ('done'|'interrupted', audio|None)."""
    cancelled = threading.Event()
    out_q: queue.Queue = queue.Queue()
    holder: dict = {}
    start_producer(router, transcript, cancelled, out_q, holder)

    player = DuplexBargeInPlayer()
    spoken: list[str] = []
    first = True

    def finish(status, audio=None):
        cancelled.set()
        route = holder.get("route")
        if route is not None:
            route.cancel()
            router.remember_spoken(transcript, route, " ".join(spoken))
        return status, audio

    while True:
        item, barge_audio = next_item(out_q)
        if barge_audio is not None:
            print("  [barge-in] interrupted while waiting for the reply; cancelled it")
            return finish("interrupted", barge_audio)
        if item is None:
            return finish("done")

        sentence, audio, rate = item
        if first:
            first = False
            route = holder["route"]
            first_ms = (time.perf_counter() - t0) * 1000
            print(f"  [{route.intent} @ {route.confidence:.2f}] first audio ready {first_ms:.0f} ms after speech ended (plus the ~0.7 s silence wait)")
            log_turn({"live": True, "intent": route.intent, "stt_ms": stt_ms,
                      "classify_ms": route.classify_ms, "first_audio_ms": first_ms})

        spoken.append(sentence)
        print(f"  > {sentence}")
        if audio.size and player.play_with_monitoring(audio, rate):
            print("  [barge-in] interrupted mid-speech; stopped playback and cancelled the rest")
            return finish("interrupted", player.wait_for_capture(timeout=20.0))


def main():
    router = Router()
    print("Loading the language model...", flush=True)
    if not qna.warm_up():
        print("  (could not reach Ollama; Q&A will fail until `ollama serve` is running)")
    print("Voice pipeline (streaming replies + barge-in + memory). Press Ctrl+C to exit.\n")

    pending_audio = None
    while True:
        if pending_audio is not None and pending_audio.size > 0:
            audio = pending_audio
        else:
            print("Listening... (speak, then pause when you're done)")
            audio = record_utterance()
        pending_audio = None

        if audio.size == 0:
            print("  (no speech detected, try again)\n")
            continue

        duration = len(audio) / SAMPLE_RATE
        t = time.perf_counter()
        transcript = transcribe_array(audio)
        stt_ms = (time.perf_counter() - t) * 1000
        if not transcript:
            print(f"  (captured {duration:.1f}s but got an empty transcript, try again)\n")
            continue

        print(f'  Heard ({duration:.1f}s): "{transcript}"')
        status, captured = respond(router, transcript, time.perf_counter() - stt_ms / 1000, stt_ms)
        if status == "interrupted":
            pending_audio = captured
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    # PortAudio/torch threads can abort during normal interpreter teardown; skip it.
    print("\nBye.", flush=True)
    os._exit(0)
