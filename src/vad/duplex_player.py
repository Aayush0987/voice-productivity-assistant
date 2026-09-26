"""
Barge-in-aware playback using a SINGLE duplex audio stream.

The earlier approach (Phase 7 first pass) opened a separate input stream for
mic monitoring and a separate output stream for TTS playback, each running
concurrently. That caused audible crackling and a hard segfault on shutdown
in live testing — running two independent PortAudio streams against the same
device, with Silero VAD neural-net inference competing for CPU time inside
the realtime audio path, is fundamentally unstable.

Fix: one sd.Stream (duplex) drives both directions through a single
realtime callback. That callback does only cheap work (numpy slicing +
queue.put) — the actual VAD inference runs on a separate ordinary thread,
decoupled from the realtime audio path, fed by a queue.
"""
import collections
import os
import queue
import threading

_DEBUG = os.environ.get("VAD_DEBUG") == "1"

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator

from src.vad.turn_taking import (
    CHUNK_SAMPLES,
    PRE_SPEECH_PAD_CHUNKS,
    SAMPLE_RATE,
    get_vad_model,
)


class DuplexBargeInPlayer:
    """Plays a pre-synthesized audio buffer while monitoring the mic for
    barge-in through a single duplex stream. See module docstring for why."""

    def __init__(
        self,
        vad_threshold: float = 0.6,
        consecutive_speech_chunks: int = 3,
        silence_timeout_s: float = 0.7,
        max_capture_s: float = 15.0,
    ):
        self.vad_threshold = vad_threshold
        self.consecutive_speech_chunks = consecutive_speech_chunks
        self.silence_timeout_s = silence_timeout_s
        self.max_capture_s = max_capture_s

        self.interrupted = threading.Event()
        self.captured_audio: np.ndarray | None = None
        self._audio_q: queue.Queue = queue.Queue()
        self._stop_vad = threading.Event()
        self._vad_thread: threading.Thread | None = None

    def _vad_worker(self):
        model = get_vad_model()
        vad_iterator = VADIterator(
            model,
            threshold=self.vad_threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=int(self.silence_timeout_s * 1000),
        )

        pre_buffer = collections.deque(maxlen=PRE_SPEECH_PAD_CHUNKS)
        collected = []
        speech_started = False
        consecutive_speech = 0
        max_chunks = int(self.max_capture_s * SAMPLE_RATE / CHUNK_SAMPLES)
        chunks_done = 0

        while chunks_done < max_chunks:
            if not speech_started and self._stop_vad.is_set():
                break
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            chunks_done += 1

            if not speech_started:
                pre_buffer.append(chunk)

            event = vad_iterator(torch.from_numpy(chunk), return_seconds=False)
            is_speech = event is not None and "start" in event

            if _DEBUG and event:
                print(f"    [vad-debug] event: {event}", flush=True)

            if is_speech:
                consecutive_speech += 1
            elif not speech_started:
                consecutive_speech = 0

            if not speech_started and consecutive_speech >= self.consecutive_speech_chunks:
                speech_started = True
                self.interrupted.set()
                collected.extend(pre_buffer)
                pre_buffer.clear()

            if speech_started:
                collected.append(chunk)

            if event and "end" in event and speech_started:
                break

        vad_iterator.reset_states()
        if speech_started:
            self.captured_audio = (
                np.concatenate(collected) if collected else np.array([], dtype="float32")
            )

    def play_with_monitoring(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Plays `audio` while monitoring for barge-in. Returns True if
        playback was cut short by an interruption."""
        if audio.size == 0:
            return False

        if sample_rate != SAMPLE_RATE:
            import scipy.signal as sig

            num_samples = int(len(audio) * SAMPLE_RATE / sample_rate)
            audio = sig.resample(audio, num_samples).astype("float32")
        audio = np.ascontiguousarray(audio, dtype="float32")

        self.interrupted.clear()
        self.captured_audio = None
        self._stop_vad.clear()
        self._audio_q = queue.Queue()

        self._vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self._vad_thread.start()

        state = {"play_idx": 0, "chunk_count": 0}
        n = len(audio)

        def callback(indata, outdata, frames, time_info, status):
            mic_chunk = indata[:, 0].copy()
            self._audio_q.put(mic_chunk)

            if _DEBUG:
                state["chunk_count"] += 1
                if state["chunk_count"] % 15 == 0:  # ~every 480ms
                    peak = float(np.abs(mic_chunk).max())
                    print(f"    [vad-debug] mic peak amplitude: {peak:.4f}", flush=True)

            if self.interrupted.is_set():
                outdata[:] = 0
                raise sd.CallbackStop()

            play_idx = state["play_idx"]
            remaining = n - play_idx
            if remaining <= 0:
                outdata[:] = 0
                raise sd.CallbackStop()

            take = min(frames, remaining)
            outdata[:take, 0] = audio[play_idx : play_idx + take]
            if take < frames:
                outdata[take:, 0] = 0
            state["play_idx"] = play_idx + take

        try:
            with sd.Stream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                callback=callback,
            ) as stream:
                while stream.active and not self.interrupted.is_set():
                    sd.sleep(20)
        except sd.CallbackStop:
            pass

        was_interrupted = self.interrupted.is_set()
        self._stop_vad.set()
        return was_interrupted

    def wait_for_capture(self, timeout: float = 20.0) -> np.ndarray:
        if self._vad_thread is not None:
            self._vad_thread.join(timeout=timeout)
        return self.captured_audio if self.captured_audio is not None else np.array([], dtype="float32")
