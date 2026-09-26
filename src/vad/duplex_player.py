"""
Barge-in-aware playback using a SINGLE duplex audio stream.

History: the first Phase 7 attempt opened a separate input stream (mic
monitor) and a separate output stream (playback) concurrently, which caused
audible crackling and a segfault on shutdown. One sd.Stream drives both
directions through a single realtime callback instead. That callback does
only cheap work (numpy copies + queue.put); Silero inference runs on an
ordinary thread fed by a queue.

On interrupt the callback outputs silence immediately but the stream KEEPS
RUNNING so the mic keeps capturing until the user finishes their sentence.
The decision logic lives in barge_in_engine.py (echo-aware, testable offline).
"""
import os
import queue
import threading

import numpy as np
import sounddevice as sd

from src.vad.barge_in_engine import BargeInEngine, DoubleTalkDetector
from src.vad.turn_taking import CHUNK_SAMPLES, SAMPLE_RATE

_DEBUG = os.environ.get("VAD_DEBUG") in ("1", "2")
_DEBUG_EVERY = 1 if os.environ.get("VAD_DEBUG") == "2" else 8


class DuplexBargeInPlayer:
    """Create one per spoken message; call play_with_monitoring once per
    sentence. The echo-gain estimate is shared across sentences."""

    def __init__(self, silence_timeout_s: float = 0.7, max_capture_s: float = 15.0):
        self.silence_timeout_s = silence_timeout_s
        self.max_capture_s = max_capture_s
        self.detector = DoubleTalkDetector()

        self.interrupted = threading.Event()
        self.captured_audio: np.ndarray | None = None
        self._q: queue.Queue = queue.Queue()
        self._stop_worker = threading.Event()
        self._capture_done = threading.Event()
        self._worker: threading.Thread | None = None

    def _vad_worker(self):
        engine = BargeInEngine(
            silence_timeout_s=self.silence_timeout_s,
            max_capture_s=self.max_capture_s,
            detector=self.detector,
        )
        n = 0
        while not engine.done:
            if not engine.triggered and self._stop_worker.is_set():
                break
            try:
                mic, ref = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stop_worker.is_set():
                    break  # stream closed, no more audio is coming: never wait forever
                continue
            engine.feed(mic, ref)
            if engine.triggered and not self.interrupted.is_set():
                self.interrupted.set()
            if _DEBUG:
                n += 1
                if n % _DEBUG_EVERY == 0 or (engine.triggered and n % _DEBUG_EVERY == 1):
                    d = self.detector
                    print(
                        f"    [vad-debug] p={engine.last_prob:.2f} mic_rms={np.sqrt(np.mean(mic**2)):.3f} "
                        f"ratio={d.last_ratio:.2f} echo_gain={d.last_gain:.2f} triggered={engine.triggered}",
                        flush=True,
                    )
        if engine.triggered:
            self.captured_audio = engine.captured()
        self._capture_done.set()

    def play_with_monitoring(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Plays `audio`; returns True if the user barged in. When True, the
        user's full utterance has already been captured (self.captured_audio)."""
        if audio.size == 0:
            return False

        if sample_rate != SAMPLE_RATE:
            import scipy.signal as sig

            audio = sig.resample(audio, int(len(audio) * SAMPLE_RATE / sample_rate))
        audio = np.ascontiguousarray(audio, dtype="float32")
        total = len(audio)

        self.interrupted.clear()
        self._capture_done.clear()
        self._stop_worker.clear()
        self.captured_audio = None
        self._q = queue.Queue()
        if _DEBUG:
            print('    [vad-debug] ---- new stream / sentence ----', flush=True)
        self._worker = threading.Thread(target=self._vad_worker, daemon=True)
        self._worker.start()

        pos = 0

        def callback(indata, outdata, frames, time_info, status):
            nonlocal pos
            mic = indata[:, 0].copy()
            outdata[:] = 0
            finished = False

            if not self.interrupted.is_set():
                take = min(frames, total - pos)
                if take > 0:
                    outdata[:take, 0] = audio[pos : pos + take]
                pos += max(take, 0)
                finished = pos >= total

            self._q.put((mic, outdata[:, 0].copy()))
            if finished:
                raise sd.CallbackStop()

        with sd.Stream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        ) as stream:
            while stream.active:
                if self.interrupted.is_set() and self._capture_done.is_set():
                    break
                sd.sleep(20)

        was_interrupted = self.interrupted.is_set()
        self._stop_worker.set()
        self._worker.join(timeout=2.0)
        return was_interrupted

    def wait_for_capture(self, timeout: float = 20.0) -> np.ndarray:
        self._capture_done.wait(timeout=timeout)
        if self.captured_audio is None:
            return np.array([], dtype="float32")
        return self.captured_audio
