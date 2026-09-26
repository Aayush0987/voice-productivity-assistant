# Progress Tracker — Voice Productivity Assistant

Local-only tracker. No git commits/pushes happen until explicitly requested.

Legend: `[ ]` not started · `[~]` in progress · `[x]` done

## Phase 0 — Study
- [ ] STT/TTS/VAD fundamentals, audio sampling basics
- [ ] Barge-in / interruption handling patterns
- [ ] Text classification basics (encoder-only models)
- [ ] Function-calling / tool-routing patterns in voice assistants
- [ ] Structured extraction (task + datetime from free text)
- [ ] Pipecat pipeline/frame architecture

## Phase 1 — Intent Classifier
- [x] Generate synthetic labeled dataset (~150-300 phrases/intent) via Groq free tier
      (220/intent, 660 total; train=561 test=99; script: src/classifier/generate_dataset.py,
      model used: openai/gpt-oss-120b via Groq)
- [x] Fine-tune small classifier (DistilBERT) on dataset
      (script: src/classifier/train.py, 6 epochs, MPS device, ~39s train time,
      model saved to models/intent_classifier/)
- [x] Evaluate on held-out test set, record accuracy
      **Accuracy: 100% (99/99)** — precision/recall/f1 all 1.00 across qna/reminder/weather.
      Caveat for README "known limitations": synthetic LLM-generated phrases per intent
      are quite distinct from each other, so this is likely an optimistic ceiling vs.
      real messy speech — re-stress-test with ambiguous phrasings in Phase 3.
- [x] Validate with typed text input
      (script: src/classifier/predict.py — REPL or single-shot CLI)
      Manually tested 6 phrases incl. ambiguous ones ("what should I wear today" -> weather,
      "add milk to my shopping list" -> reminder, "hey remind me about the weather forecast
      for my trip" -> reminder w/ 0.25 weather confidence) — all sensible, confidences
      appropriately lower on genuinely ambiguous phrasing.

## Phase 2 — Individual Functionality Handlers (text-only)
- [x] Q&A handler (local LLM) — src/handlers/qna.py, uses Ollama HTTP API
      (localhost:11434) with model llama3.1:8b (installed via homebrew,
      model pulled ~4.9GB). Tested with multiple questions, concise
      spoken-style answers as instructed by system prompt.
- [x] Reminder handler (extract task+datetime, store to SQLite)
      src/handlers/reminder.py — uses `dateparser.search_dates` (must restrict
      `languages=["en"]`, otherwise it false-matches short words like "me"/"to"
      as dates in other locales — found and fixed). Also found and fixed a
      search_dates bug where compound phrases like "tomorrow at 9am" kept the
      right date but wrong time (defaulted to current time instead of 9am) —
      fixed by re-parsing the isolated matched phrase with dateparser.parse()
      instead of trusting search_dates' own merged datetime.
      Stores to data/reminders/reminders.db. Tested on 5 phrasings, all correct
      except a cosmetic issue: "next Monday morning" only extracts "Monday" as
      the date phrase, leaving a stray "next ... morning" in the task text —
      dateparser's search granularity limitation, noted for README limitations.
- [x] Weather handler (extract location, call Open-Meteo, format response)
      src/handlers/weather.py — regex extracts location after in/for/at/near
      (must be consecutive capitalized words only, else it swallows trailing
      lowercase words like "tomorrow" into the place name — found and fixed).
      Falls back to asking "Which city would you like the weather for?" when
      no location given (per user's choice: no hardcoded default). Tested with
      Mumbai, Tokyo, New York City, London, and no-location input — all correct.
- [x] Test all three independently via text input
      (script: scripts/test_handlers.py — quick smoke test across all 3)

## Phase 3 — Router Integration
- [x] Wire text input → classifier → handler → response
      src/router/router.py — Router class loads classifier once, routes text
      to the matching handler, records per-request latency. REPL mode via
      `python src/router/router.py`.
- [x] Test ambiguous/edge-case phrasings
      (script: scripts/test_router.py, 10 phrases: clear-cut + ambiguous)
      Findings for README "known limitations":
        - Confidence appropriately drops on genuinely ambiguous input
          ("should I bring an umbrella today" -> weather @ 0.74, vs. 0.99+ on
          clear cases) — classifier is reasonably well-calibrated.
        - Correctly generalizes past keyword overlap: "what's a good reminder
          app" -> qna @ 0.98 despite containing the word "reminder".
        - Genuine misclassification found: "hey what's up" -> weather @ 0.97
          (should be qna). Casual non-question greetings weren't represented
          in the synthetic qna training data. FIXED: added 20 curated greeting
          phrases to qna (scripts/augment_qna_greetings.py, dataset now 240/
          intent qna, 660+20 total), retrained — held-out accuracy still 100%
          (102 samples), and "hey what's up"/"how's it going" now correctly
          classify as qna @ ~0.996 with no regression on reminder/weather.
        - Cosmetic: "set a timer for 10 minutes" -> reminder, task text left
          as "set a timer for" (trailing preposition) after the "10 minutes"
          date phrase is stripped — same dateparser extraction limitation as
          Phase 2.

## Phase 4 — STT Integration
- [x] Microphone → Whisper (faster-whisper) → transcript → router
      src/stt/whisper_stt.py — model `base.en`, CPU, int8 quantized (fast/free,
      no GPU needed on M4 Pro). transcribe_file() for testing, record_and_transcribe()
      for live mic via sounddevice. src/orchestration/voice_pipeline_v1.py wires
      mic -> STT -> Router (fixed 5s recording per turn; VAD-based turn-taking
      is Phase 5).
      IMPORTANT: this agent session has no live microphone access, so end-to-end
      testing used synthesized test audio (macOS `say` command -> .aiff files)
      instead of live speech. Transcription tested on 4 phrases (weather/qna/
      greeting/reminder) — all transcribed correctly (e.g. "Remind me to call
      Mom tomorrow at 6 p.m." verbatim).
      Live-mic testing DONE by user on real hardware, 6 turns:
        - qna ("what is the capital of France") -> perfect transcript, correct
          route, correct answer.
        - weather ("what is the weather like in Mumbai") -> perfect transcript
          (across 2 merged utterances), correct route, correct answer.
        - reminder ("remind me to call mom tomorrow at 6pm") -> MANGLED:
          transcript came out as "on tomorrow at 6 p.m. Remind me to call
          Mont-", word order scrambled and "Mom" truncated to "Mont-". Root
          cause: the fixed 5s recording window cut the sentence off mid-word;
          Whisper's decoder reorders/hallucinates when fed truncated audio.
        - 3 silent turns -> Whisper hallucinated filler text ("... ... ...",
          ". . . . .") instead of returning empty string for true silence; the
          garbage transcript was then routed to qna and answered nonsensically
          by the LLM.
      Conclusion: fixed-duration recording is not viable for real use — both
      failure modes (mid-sentence cutoff, silence hallucination) are exactly
      what VAD-based turn-taking (Phase 5) is meant to fix. Proceeding to
      Phase 5 as the direct next step, motivated by this real evidence.
      Bug found + fixed during this integration: Whisper renders "6pm" as
      "6 p.m." (with periods), which broke dateparser's tokenization in the
      reminder handler (split "p.m." into a stray "m" fragment, silently fell
      back to current time instead of 6pm). Fixed in src/handlers/reminder.py
      by normalizing "a.m."/"p.m." -> "am"/"pm" before date extraction — this
      is exactly the kind of STT-specific formatting quirk that wouldn't show
      up in text-only testing (Phase 2/3), which is why it wasn't caught until
      this integration point.

## Phase 5 — VAD + Turn-Taking
- [x] Silero VAD end-of-speech detection
      src/vad/turn_taking.py — `silero-vad` pip package, streaming VADIterator
      API, 512-sample (32ms) chunks at 16kHz. Motivated directly by the Phase 4
      live-mic failures (mid-sentence cutoff, silence hallucination).
      Note: silero-vad's own `read_audio()` pulls in torchaudio+torchcodec,
      which isn't installed/needed — used `soundfile` directly instead for
      any offline audio loading in this project.
- [x] Basic turn-taking loop
      record_utterance() in src/vad/turn_taking.py: opens a continuous mic
      stream, keeps a ~320ms pre-speech pad buffer (so the moment right
      before VAD triggers isn't lost), starts collecting on VAD 'start'
      event, stops on 'end' event (0.7s of trailing silence) or a 15s safety
      cap. Returns empty array on pure silence -> whisper_stt.transcribe_array()
      short-circuits on empty input instead of calling Whisper at all, which
      directly fixes the Phase 4 silence-hallucination bug (no more "... ... ..."
      transcripts from dead air).
      src/orchestration/voice_pipeline_v2.py wires mic -> VAD -> Whisper -> router.
      Offline validation (agent has no live mic access): simulated a mic stream
      by feeding a synthesized "remind me to call mom tomorrow at six pm" wav
      (with added trailing silence) through the same chunk-by-chunk VAD loop
      used in record_utterance() — correctly detected end-of-speech, captured
      the full untruncated sentence, transcribed it perfectly. This directly
      addresses the mid-sentence truncation seen in the Phase 4 live test.
      Live-mic re-test DONE by user on voice_pipeline_v2.py (real hardware):
        - weather ("what is the weather in Mumbai") -> correct, 3.0s capture.
        - qna ("what is the capital of France") -> correct, 3.0s capture.
        - reminder ("Can you remind me to call mom tomorrow at 6 p.m. in the
          evening?") -> transcript now FULLY CAPTURED, no truncation (fixed
          window would have cut this off) — confirms the core Phase 5 fix works.
      New bug surfaced by this transcript (task-extraction, not VAD): task text
      came out as "Can you remind me to call mom evening?" — two problems:
        1. "Can you remind me to..." (polite phrasing) wasn't stripped — the
           leading-filler regex only matched an unprefixed "remind me to".
        2. search_dates matched only "tomorrow at 6 pm in the", leaving a
           dangling "evening?" in the task even though the time was already
           correctly captured from "6 pm".
      Both FIXED in src/handlers/reminder.py: _LEADING_FILLERS now accepts an
      optional "can/could/would you" + "please" prefix; added
      _DANGLING_TIME_QUALIFIER to drop a stray trailing "in the
      morning/afternoon/evening/night" once a datetime was already found.
      Re-tested the exact failing phrase plus 3 regression cases (all correct)
      and re-ran the full Phase 3 edge-case suite (scripts/test_router.py) —
      no regressions; one classifier flip on an already-borderline case
      ("should I bring an umbrella today": weather@0.74 -> reminder@0.78 after
      the greeting retrain) — still low-confidence either way, not concerning.

## Phase 6 — TTS Integration
- [x] Response text → Piper → audio output
      src/tts/piper_tts.py — `piper-tts` pip package, PiperVoice class, voice
      model `en_US-lessac-medium` downloaded via `python -m piper.download_voices`
      (~63MB, free, no key) into models/tts/. speak() synthesizes + plays via
      sounddevice.
      Quality check without human ears: synthesized a weather-style sentence,
      resampled 22050Hz -> 16kHz (scipy.signal.resample), fed it back through
      the Phase 4 Whisper STT as a round-trip sanity check — transcript matched
      the original text exactly (only cosmetic number-formatting differences,
      e.g. "twenty nine" -> "29"). Confirms clean, intelligible speech output.
- [x] Incremental streaming to TTS
      speak_streaming() splits response text on sentence boundaries and
      synthesizes+plays each sentence as soon as it's ready, instead of
      waiting for the whole response to be synthesized first — reduces
      perceived latency, especially for longer Q&A answers.
      src/orchestration/voice_pipeline_v3.py — full loop: mic -> VAD -> Whisper
      -> router -> Piper TTS -> speaker. Still no barge-in (Phase 7 next).
      Live-playback test DONE by user on real hardware, 5 full voice turns:
      greeting, qna, reminder (the exact previously-buggy polite phrasing —
      now extracts cleanly: "Can you remind me to call my mom tomorrow at 6
      in the evening?" -> correct task + time), weather, and a closing
      chit-chat turn. All 5 transcribed, routed, and answered correctly with
      no errors. User confirmed audio played back clearly and naturally
      through speakers with no glitches between streamed sentences.

## Phase 7 — Barge-In / Interruption Handling (core differentiator)
- [x] Detect user speech during agent playback
      src/vad/barge_in.py — MicMonitor class runs a background thread that
      continuously reads mic chunks + Silero VAD while the assistant is
      "thinking" (handler/LLM call) or speaking (TTS). Requires
      `consecutive_speech_chunks` (3, ~96ms) of sustained VAD-positive frames
      before treating it as a real interrupt, not just one noisy blip — this
      is a deliberate mitigation for a fundamental limitation: this stack has
      NO acoustic echo cancellation (AEC), so the mic can pick up the
      assistant's own voice bleeding from the speakers and falsely trigger a
      self-interruption. This is a known, unresolved risk — flagging clearly
      for the README. Real AEC (e.g. WebRTC APM) would be the proper fix but
      was out of scope for the time budgeted here.
- [x] Stop TTS + discard in-flight response + start listening on interrupt
      src/tts/piper_tts.py — speak_streaming_interruptible() plays sentence-
      by-sentence via non-blocking sd.play(), polling an interrupt Event and
      calling sd.stop() the instant it's set (playback cut immediately, not
      at the next sentence boundary).
      src/orchestration/voice_pipeline_v4.py — full loop: router.route() runs
      on a background thread (concurrent.futures) while a MicMonitor watches
      for barge-in during that "thinking" phase too, not just during TTS —
      covers the Q&A path where LLM generation can take a few seconds. If
      interrupted at any point, the in-flight response/handler result is
      simply discarded (never spoken), and the utterance MicMonitor already
      captured during the interruption is fed directly into the next loop
      iteration — no need to re-record.
      Caveat: "discard in-flight response" during the thinking phase discards
      the RESULT once the background thread completes, but doesn't literally
      cancel the underlying Ollama HTTP request already in flight (Python
      threads/requests aren't cleanly cancellable mid-call) — the wasted
      generation still runs to completion in the background, it's just never
      spoken or used. Acceptable for reminder/weather (near-instant) but a
      real inefficiency for Q&A specifically; noted as a known limitation.
      Smoke-tested (agent-side, no live mic/speaker): started a MicMonitor
      thread and played TTS audio concurrently via sd.play in the same
      process — no crashes, confirming sounddevice input+output streams can
      run concurrently without device conflicts. This does NOT validate real
      barge-in accuracy or the AEC/self-triggering risk — that requires a
      human actually interrupting live playback.
- [~] Test edge cases (mid-reminder-confirmation, mid-weather-report, rapid interrupts)
      First live test by user surfaced a serious problem: audible crackling
      during playback ("something is breaking" after each word) and a hard
      SEGFAULT on shutdown (`zsh: segmentation fault`) after Ctrl+C. No actual
      barge-in was detected when the user tried to interrupt mid-speech.
      ROOT CAUSE identified: the first-pass design (src/vad/barge_in.py
      MicMonitor) opened a separate raw input stream for mic monitoring
      *concurrently* with a separate output stream for TTS playback (two
      independent PortAudio streams against the same device), while also
      running Silero VAD neural-net inference directly inside that
      concurrency window. Two independent streams contending for the same
      device, combined with GIL contention from real-time neural inference,
      caused output buffer underruns (the crackle) and an unstable shutdown
      path (the segfault).
      FIX: rewrote playback+monitoring as a single DUPLEX stream
      (src/vad/duplex_player.py, DuplexBargeInPlayer) — one sd.Stream driving
      both input and output through one realtime callback. The callback only
      does cheap numpy array work (copy mic frames to a queue, copy TTS
      samples to the output buffer); the actual VAD inference runs on a
      separate ordinary thread fed by that queue, fully decoupled from the
      realtime audio path. This is the standard architecture for real
      duplex/barge-in audio systems and directly targets the two-streams
      root cause. src/tts/piper_tts.py gained speak_streaming_duplex() using
      this player; the old MicMonitor+separate-streams approach
      (speak_streaming_interruptible) is kept only for reference and marked
      deprecated for live use. voice_pipeline_v4.py's speak_with_barge_in()
      now uses the duplex player.
      Smoke-tested (agent-side): synthesized ~6s of speech and played it
      through DuplexBargeInPlayer — completed without exceptions. This does
      NOT prove the crackle/segfault are gone in practice; only a live retest
      can confirm that.
      Also fixed during this same session (found from the transcript, not
      barge-in-related): "the weather of Mumbai" / "the weather of Danbad"
      both failed to extract a location because the regex only recognized
      in/for/at/near as location prepositions, missing the common "of"
      phrasing — added "of" to src/handlers/weather.py's _LOCATION_PATTERN.
      STILL PENDING: user needs to re-run voice_pipeline_v4.py and confirm
      (a) no more crackling, (b) no more segfault on exit, (c) barge-in
      actually triggers when interrupting mid-speech, (d) whether the
      no-AEC self-triggering risk (assistant's own voice picked up by the
      mic) materializes in practice.

## Phase 8 — Latency Measurement
- [ ] Log per-stage latency (STT, classification, handler, LLM gen, TTS)
- [ ] Compare reminder/weather path vs Q&A path latency

## Phase 9 — Full Integration Demo
- [ ] Demo script covering all three intents
- [ ] Barge-in demo (interrupt reminder confirmation to ask weather)

## Phase 10 — Documentation
- [ ] Architecture diagram
- [ ] Classifier accuracy reported
- [ ] Latency breakdown table
- [ ] Barge-in demo (video/GIF)
- [ ] Cost confirmation (all components $0)
- [ ] Known limitations
- [ ] Lessons learned

---

## Environment Notes
- macOS 26.6.2, Apple Silicon (M4 Pro, 24GB unified memory)
- Python 3.13.5, venv created at `venv/` (not committed)
- Project dir: `/Users/aayush/VOICE AI AGENT`
- Dependencies listed in `requirements.txt` (not yet installed — pending confirmation of exact package set as each phase starts, since some like `pipecat-ai`/`piper-tts` may need version pinning once we reach that phase)

## Open Items / Blockers
- Default location for weather (when none given in speech) — to be decided in Phase 2
