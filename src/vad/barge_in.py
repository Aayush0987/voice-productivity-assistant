"""
MicMonitor: listens for the user during the "thinking" phase (handler / LLM
call), when nothing is playing. Input-only stream, so none of the duplex
problems apply, and there is no echo (no playback reference is passed).
Uses the same BargeInEngine as the speaking phase.
"""
import threading

import numpy as np
import sounddevice as sd

from src.vad.barge_in_engine import BargeInEngine
from src.vad.turn_taking import CHUNK_SAMPLES, SAMPLE_RATE


class MicMonitor:
    def __init__(self, max_seconds: float = 15.0, silence_timeout_s: float = 0.7):
        self.max_seconds = max_seconds
        self.silence_timeout_s = silence_timeout_s
        self.interrupted = threading.Event()
        self.captured_audio: np.ndarray | None = None
        self._stop_requested = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop if no interruption occurred (no-op once one has)."""
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def wait_for_capture(self, timeout: float | None = None) -> np.ndarray:
        self._done.wait(timeout=timeout)
        if self.captured_audio is None:
            return np.array([], dtype="float32")
        return self.captured_audio

    def _run(self):
        engine = BargeInEngine(
            silence_timeout_s=self.silence_timeout_s, max_capture_s=self.max_seconds
        )
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES
            ) as stream:
                while not engine.done:
                    if not engine.triggered and self._stop_requested.is_set():
                        break
                    chunk, _ = stream.read(CHUNK_SAMPLES)
                    engine.feed(chunk.flatten())
                    if engine.triggered and not self.interrupted.is_set():
                        self.interrupted.set()
        except Exception:
            pass
        finally:
            if engine.triggered:
                self.captured_audio = engine.captured()
            self._done.set()
