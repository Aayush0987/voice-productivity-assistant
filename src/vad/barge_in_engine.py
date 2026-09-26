"""
Barge-in decision logic, free of any audio I/O so it can be tested offline.

Two things had to be fixed from the first Phase 7 attempt:

1. Silero's VADIterator emits a single {'start': ...} event at speech onset
   and then nothing until {'end': ...}. Counting "consecutive start events"
   could never reach 3, so barge-in could never fire. This engine reads the
   raw per-chunk speech probability instead.

2. With no acoustic echo cancellation, the assistant's own voice leaking into
   the mic looks exactly like speech to a VAD (the debug log showed a 'start'
   ~0.2s into every playback, before the user said anything). So VAD alone
   cannot decide. DoubleTalkDetector (a Geigel-style detector) compares mic
   energy to the energy the assistant is currently playing and only reports
   near-end speech when the mic is well above the echo level.
"""
import collections

import numpy as np
import torch

from src.vad.turn_taking import CHUNK_SAMPLES, PRE_SPEECH_PAD_CHUNKS, SAMPLE_RATE, get_vad_model


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


class DoubleTalkDetector:
    """True when mic energy exceeds what echo of the playback alone explains.

    The echo path gain (mic level / playback level) is unknown and varies with
    volume and hardware, so it is estimated on the fly. Echo-only ratios are
    bounded above by the true gain (playback level is a max over a window) but
    swing widely below it with speech dynamics, so a HIGH quantile of recent
    ratios tracks the echo ceiling; a low quantile underestimates it and
    false-triggers on the assistant's own voice.
    """

    def __init__(
        self,
        ref_lag: tuple[int, int] = (2, 8),  # playback chunks (age lo..hi-1, ~32ms each) that may be echoing now
        smooth: int = 1,
        margin: float = 1.3,  # mic/playback ratio must exceed the echo ceiling by this factor
        floor: float = 0.01,  # ignore mic RMS below this (noise)
        warmup: int = 8,
        startup_skip: int = 16,  # ~0.5s: echo takes a while to reach the mic (measured on the M4 Mac); ratios are bogus until then
        ratio_window: int = 100,
        quantile: float = 0.8,
    ):
        self.ref_lag = ref_lag
        self.ref_hist = collections.deque(maxlen=ref_lag[1])
        self.ratios = collections.deque(maxlen=ratio_window)
        self.margin = margin
        self.floor = floor
        self.warmup = warmup
        self.startup_skip = startup_skip
        self.quantile = quantile
        self._playing_chunks = 0
        self._recent = collections.deque(maxlen=smooth)
        self.last_ratio = 0.0
        self.last_gain = 0.0

    def process(self, mic: np.ndarray, ref: np.ndarray) -> bool:
        mic_rms = _rms(mic)
        self.ref_hist.append(_rms(ref))
        h = list(self.ref_hist)
        lo, hi = self.ref_lag
        window = h[max(0, len(h) - hi) : len(h) - lo] if len(h) > lo else []
        ref_level = max(window) if window else 0.0

        if ref_level < 1e-3:  # nothing playing: any audible mic input is the user
            return mic_rms > self.floor

        self._playing_chunks += 1
        if self._playing_chunks <= self.startup_skip:
            return False

        ratio = mic_rms / ref_level
        self.ratios.append(ratio)
        self.last_ratio = ratio
        if len(self.ratios) < self.warmup:
            return False
        gain = float(np.quantile(self.ratios, self.quantile))
        self.last_gain = gain
        # Chunk-level echo ratios spike (measured 0.03..0.31 around a typical 0.1
        # on real hardware) and stream start/stop or a knock can produce a single
        # 10x burst. Judge the recent MEDIAN: one outlier chunk is ignored (a mean
        # let one click carry the next few chunks over the threshold in live
        # testing), sustained user speech still lifts most of the window.
        self._recent.append(ratio)
        smoothed = float(np.median(self._recent))
        return mic_rms > self.floor and smoothed > self.margin * gain


class BargeInEngine:
    """Consumes (mic_chunk, playback_chunk) pairs; decides when the user has
    barged in, then keeps capturing their utterance until they pause."""

    def __init__(
        self,
        vad_threshold: float = 0.5,
        consecutive_chunks: int = 7,
        silence_timeout_s: float = 0.7,
        max_capture_s: float = 15.0,
        detector: DoubleTalkDetector | None = None,
    ):
        self.model = get_vad_model()
        self.model.reset_states()
        self.detector = detector or DoubleTalkDetector()
        self.vad_threshold = vad_threshold
        self.consecutive_chunks = consecutive_chunks
        self.silence_chunks = int(silence_timeout_s * SAMPLE_RATE / CHUNK_SAMPLES)
        self.max_chunks = int(max_capture_s * SAMPLE_RATE / CHUNK_SAMPLES)

        self._pre = collections.deque(maxlen=PRE_SPEECH_PAD_CHUNKS)
        self._collected: list[np.ndarray] = []
        self._consecutive = 0
        self._silence = 0
        self.triggered = False
        self.done = False
        self.last_prob = 0.0

    def feed(self, mic: np.ndarray, ref: np.ndarray | None = None) -> None:
        if self.done:
            return
        if ref is None:
            ref = np.zeros_like(mic)

        with torch.no_grad():
            self.last_prob = float(self.model(torch.from_numpy(mic), SAMPLE_RATE).item())
        is_speech = self.last_prob >= self.vad_threshold

        if not self.triggered:
            self._pre.append(mic)
            near_end = self.detector.process(mic, ref)
            self._consecutive = self._consecutive + 1 if (is_speech and near_end) else 0
            if self._consecutive >= self.consecutive_chunks:
                self.triggered = True
                self._collected.extend(self._pre)
                self._pre.clear()
            return

        self._collected.append(mic)
        self._silence = 0 if is_speech else self._silence + 1
        if self._silence >= self.silence_chunks or len(self._collected) >= self.max_chunks:
            self.done = True

    def captured(self) -> np.ndarray:
        if not self._collected:
            return np.array([], dtype="float32")
        return np.concatenate(self._collected)
