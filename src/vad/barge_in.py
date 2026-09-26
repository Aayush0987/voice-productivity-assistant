"""
Barge-in: a background mic monitor that runs while the assistant is
"thinking" (handler/LLM generation) or speaking (TTS playback). The moment
it detects the user starting to talk, it signals an interrupt and captures
their new utterance itself — so the caller doesn't need to record twice.

No AEC (acoustic echo cancellation): with no local/free AEC in this stack,
the mic can pick up the assistant's own voice bleeding from the speakers
during playback and falsely trigger a "barge-in". Mitigated by requiring
`consecutive_speech_chunks` consistent speech detections (not just one
noisy blip) before treating it as a real interruption — tune this or use
headphones if false-triggering is observed in practice.
"""
import collections
import threading

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


class MicMonitor:
    """Runs a mic-reading + VAD loop on a background thread from start() until
    stop() or until it detects and fully captures an interrupting utterance."""

    def __init__(
        self,
        max_seconds: float = 15.0,
        silence_timeout_s: float = 0.7,
        vad_threshold: float = 0.6,
        consecutive_speech_chunks: int = 3,
    ):
        self.max_seconds = max_seconds
        self.silence_timeout_s = silence_timeout_s
        self.vad_threshold = vad_threshold
        self.consecutive_speech_chunks = consecutive_speech_chunks

        self.interrupted = threading.Event()
        self.captured_audio: np.ndarray | None = None
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring if no interruption occurred; safe to call even if
        an interruption already happened (no-op in that case)."""
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def wait_for_capture(self, timeout: float | None = None) -> np.ndarray:
        """Block until the interrupting utterance has been fully captured
        (VAD end-of-speech or max_seconds). Only meaningful after
        `interrupted` is set."""
        self._done.wait(timeout=timeout)
        return self.captured_audio if self.captured_audio is not None else np.array([], dtype="float32")

    def _run(self):
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
        max_chunks = int(self.max_seconds * SAMPLE_RATE / CHUNK_SAMPLES)
        chunks_done = 0

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES
            ) as stream:
                while chunks_done < max_chunks:
                    if not speech_started and self._stop_requested.is_set():
                        break

                    chunk, _ = stream.read(CHUNK_SAMPLES)
                    chunk = chunk.flatten()
                    chunks_done += 1

                    if not speech_started:
                        pre_buffer.append(chunk)

                    event = vad_iterator(torch.from_numpy(chunk), return_seconds=False)
                    speech_prob_flag = event is not None and "start" in event

                    if speech_prob_flag:
                        consecutive_speech += 1
                    elif not speech_started:
                        consecutive_speech = 0

                    if (
                        not speech_started
                        and consecutive_speech >= self.consecutive_speech_chunks
                    ):
                        speech_started = True
                        self.interrupted.set()
                        collected.extend(pre_buffer)
                        pre_buffer.clear()

                    if speech_started:
                        collected.append(chunk)

                    if event and "end" in event and speech_started:
                        break
        except Exception:
            pass
        finally:
            vad_iterator.reset_states()
            if speech_started:
                self.captured_audio = (
                    np.concatenate(collected) if collected else np.array([], dtype="float32")
                )
            self._done.set()
