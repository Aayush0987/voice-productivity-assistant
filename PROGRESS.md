# Progress Tracker — Voice Productivity Assistant

Commit and push to origin/main as work progresses (repo is public: never commit .env, models, DBs).

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
- [~] Test edge cases (mid-reminder-confirmation, mid-weather-report, rapid interrupts) — qna verified, others pending
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
      Second live test (user, VAD_DEBUG=1): crackling and segfault GONE
      (duplex fix confirmed), but barge-in still never fired. The debug log
      exposed two stacked bugs:
        1. My trigger counted "3 consecutive VADIterator 'start' events", but
           VADIterator emits ONE start at onset then None until 'end' — the
           counter reset every chunk and could never reach 3. Barge-in could
           not fire for any input. (Same bug in the first MicMonitor.)
        2. The log showed {'start': ...} ~0.2s into EVERY playback before the
           user spoke: the assistant's own voice leaking into the mic scores
           VAD p~1.0 (confirmed again on hardware). So VAD alone can never
           separate user from echo — the no-AEC risk was real, not theoretical.
      Also found: after an interrupt the old player stopped the whole stream,
      which stopped the mic, truncating the user's interrupting sentence.
      REWRITE (src/vad/barge_in_engine.py, duplex_player.py, barge_in.py):
        - BargeInEngine: raw per-chunk Silero probability + DoubleTalkDetector
          (Geigel-style): compares mic RMS to the RMS of what is being played,
          with the echo gain estimated online as a high quantile of recent
          mic/playback ratios; triggers only when speech AND mic energy exceed
          the echo ceiling by a margin for N consecutive chunks. Pure logic,
          no audio I/O, so it is testable offline.
        - Player keeps the duplex stream running after an interrupt (output
          silenced immediately, mic keeps capturing until the user pauses).
        - Worker thread can no longer wait forever after the stream closes;
          pipeline exits via os._exit to skip a native teardown abort
          ('recursive_mutex lock failed', likely the earlier segfault too).
      TUNING METHOD (worth stating in the README): hand-tuning against one
      metric at a time kept regressing. Instead recorded 5 real speaker->mic
      echo takes on this Mac (scripts/record_echo_takes.py, played the way the
      pipeline plays: a fresh stream per sentence), then grid-searched
      detector configs offline against real echo + injected click transients +
      simulated user voice at 3 loudness levels (scripts/tune_barge_in.py).
      Measured on this machine: echo delay ~130ms, echo gain ~0.12-0.16, and
      single-chunk 0.4-RMS clicks occur (a mean-smoother let one click trigger
      it; switching to consecutive-chunk counting fixed that). Recordings that
      were one continuous text needed 5 consecutive chunks; sentence-structured
      recordings needed 7 — sentence boundaries matter.
      Chosen defaults: margin 1.3, quantile 0.8, echo-lag window chunks 2-8,
      7 consecutive chunks (~224ms). Offline: 0 false triggers on 5 real takes
      + 5 with clicks; caught 25/30 simulated interruptions (user at ~1.5-4x
      echo loudness), median delay ~360ms.
      Live no-user validation (assistant speaking, nobody talking, this Mac):
      first version 2/3 runs false-triggered; tuned version 1/10. NOT ZERO.
      KNOWN LIMITATIONS (README): (a) ~10% chance per 10s message that the
      assistant cuts itself off (echo/clicks), (b) a quiet user (<~1.5x echo
      level) is not detected, (c) ~0.8s at message start is deaf while the
      echo gain is estimated, (d) headphones remove echo entirely and should
      make it reliable, (e) real fix = true AEC (e.g. WebRTC APM), out of scope
      for a free/local pure-Python stack.
      Third live test (user, real interruption, speakers): BARGE-IN WORKS.
        - Interrupt while thinking (qna): "How is a rainbow made?" was cut off
          by the next question; the discarded answer was never spoken.
        - Interrupt mid-speech (qna): assistant was reading a 3-sentence
          rainbow answer; user talked over it, trigger fired at ratio 0.97 vs
          echo ceiling 0.24 (~4x), playback stopped, and the full interrupting
          sentence ("Can you tell that in short?") was captured intact and
          routed. Ctrl+C exited cleanly ("Bye.", no segfault).
        - Cosmetic: a joblib 'leaked semaphore' warning prints at exit because
          os._exit skips cleanup; harmless.
      Exposed the missing conversational memory: "Can you tell that in short?"
      got "What do you need to know?" (no context) — addressed next.
      Observed latency: a 3-sentence Q&A answer took 13.5s to generate
      (non-streaming LLM) vs ~3-5s for short ones -> stream tokens to TTS is
      the obvious improvement (README).
      NOT YET TESTED live: reminder-confirmation interrupt -> weather question
      (the Phase 9 demo scenario), weather-report interrupt, rapid successive
      interrupts, headphones.

## Extra — Conversational memory (requested by user after live testing)
Not in the original brief; added because live use showed "Can you tell that in
short?" got "What do you need to know?" and a bare "Mumbai" after "Which city?"
was routed to Q&A.
- [x] src/router/router.py holds session state: last 8 chat messages (Q&A
      gets them as Ollama chat history), last weather city, and an
      awaiting-city flag (a short reply after "Which city?" is taken as the
      city). route() only READS state; the pipeline calls remember() once the
      user actually heard the reply, so an answer discarded by a barge-in
      while thinking is never remembered.
- [x] scripts/test_memory.py (7 checks, all pass): follow-up summarizes the
      previous answer; bare city after a prompt; remembered city reused; new
      explicit city overrides; discarded answer not remembered; a long new
      request during awaiting-city is not hijacked.
- [x] Reminder handler bugs found by these tests / a live transcript, fixed:
        - dateparser cannot read "at 6 in the evening", "evening at 6" or a
          bare "at 6" and silently substitutes the CURRENT CLOCK TIME, so the
          reminder announced a time the user never said (a 2:21 AM result for
          "6 in the evening"). Now rewritten to an explicit "6 pm" first
          (bare hour: 1-6 -> pm, 7-11 -> am, stated as-is so the user can
          correct it).
        - Date with no time ("remind me tomorrow") was stamped with the
          current time of day; now defaults to 9 AM and SAYS so
          ("You didn't give a time, so I picked 9 AM.").
        - Leading "Actually,/Also,/Okay so" no longer leaks into the task;
          trailing "?", "next", "on", "by" etc. are stripped.
- Limits: memory is per-session (lost on restart), Q&A-only history, no
  entity resolution beyond the weather city.

## Phase 8 — Latency Measurement
- [x] Log per-stage latency (STT, classification, handler, LLM gen, TTS)
      Router now returns classify_ms and handler_ms separately (for Q&A the
      handler time IS the LLM generation). src/utils/latency.py appends JSONL
      to logs/latency.jsonl. scripts/benchmark_latency.py runs offline (no mic):
      macOS `say` audio -> Whisper -> router -> Piper synth of first sentence,
      after a warm-up pass. Excludes fixed VAD silence wait (0.7s) and
      playback duration, which are constants not compute.
- [x] Compare reminder/weather path vs Q&A path latency
      Median ms, n=4 per intent, M4 Pro, all 12 phrases routed correctly:
        intent    STT  classify  handler  TTS(1st sent)  total
        qna       326      20     1187        197        1730
        reminder  376      18        3        119         515
        weather   334      19     1546        103        2001
      [Superseded by the idle-machine re-run below; kept for the record.]
      FINDING (contradicts the brief's assumption): reminder is ~3.4x faster
      than Q&A as expected, but weather is SLOWER than Q&A (2.0s vs 1.7s)
      because it makes two sequential network calls to Open-Meteo (geocode,
      then forecast) — skipping the LLM does not make it fast. Caveats: Q&A
      answers here were short (system prompt asks for 1-3 sentences) and the
      LLM call is non-streaming, so time-to-first-audio grows with answer
      length; small n. Possible improvements for README: cache geocoding
      results, stream LLM tokens to TTS.

### Latency investigation after live memory test (2026-09-27)
User's first Q&A answer took 18.4s vs 4.6s for the follow-up. Measured:
- Cold start: Ollama unloads the model after 5 idle minutes; reloading cost
  ~6s. FIXED: qna.warm_up() at pipeline start + keep_alive="30m" (first
  question after warm-up 2.2s; model shows "29 minutes from now" in ollama ps).
- Generation was only 5.1 tok/s (100% GPU, so not a CPU fallback) with 11.4GB
  of 12GB swap in use. Cause: the user's Project #4 `mlx_lm lora --train`
  fine-tuning job was running on the same GPU/unified memory. Not a bug in
  this project; NOT touched.
- IMPORTANT for README/demo: the Phase 8 latency table (Q&A handler 1.2s) was
  measured on an uncontended machine. Re-measure Q&A/weather latency and
  record the demo only after the training job has finished, or numbers and
  the video will look artificially slow.

### Final latency re-measurement (idle machine)
User's training job finished (Ollama back to 27.9 tok/s, was 5.1; 27% memory
free; preflight shows no competing job). logs/latency.jsonl cleared and the
benchmark re-run, all 12 phrases routed correctly. Median ms, n=4 per intent:
        intent    STT  classify  handler  TTS(1st sent)  total
        qna       425      32     1733        231        2421
        reminder  471      27        4        151         653
        weather   401      26     1475        124        2025
Slightly higher than the first run (1730/515/2001 totals), so ~+-30% is
run-to-run noise at n=4. Robust result: reminder is ~4x faster than the
other two (no LLM, no network). Not robust: weather vs Q&A ordering (weather
was slower in run 1, slightly faster in run 2); both are ~2s. README table
updated with these numbers and this caveat. Swap still read 89% after the
job ended (macOS keeps swap allocated); generation speed was the real check.

### Streaming replies (LLM -> TTS), requested by user
Fixes the README's "time-to-first-word grows with answer length" limitation.
- [x] src/utils/sentences.py SentenceChunker: cuts streamed tokens into
      sentences as they complete. Rule that mattered: a "." followed by a
      LOWERCASE letter is not a sentence end ("5 p.m. yesterday", "U.S. is",
      "Wait... what"); needs one char of lookahead. Abbreviation list alone
      failed 3 of 11 cases. scripts/test_sentences.py: 11/11 incl. feeding 3
      characters at a time like a real stream.
- [x] src/handlers/qna.py LLMStream: Ollama stream=true, yields sentences,
      cancel() closes the HTTP connection.
- [x] Router.route_stream() + remember_spoken(): only what was actually
      spoken is remembered (an answer cut off after 2 sentences is remembered
      as those 2 sentences). test_memory.py still 7/7.
- [x] src/orchestration/voice_pipeline_v5.py: producer thread (stream ->
      sentences -> Piper synth) feeds the player, so sentence N+1 is written
      and synthesized while N plays; barge-in listens through the gaps too.
      v4 is kept for comparison.
- MEASURED (machine under load from the user's training job, so absolute
  times are inflated, the ratios are the point):
    time to first speech: 26.4s non-streamed -> 6.3s streamed (4.2x).
    cancel(): a short request right after cancelling a long stream took 1.1s;
    the same request while an abandoned long answer was still running took
    57.1s (51x). So in v4, interrupting a Q&A answer and asking something new
    would have queued behind the abandoned generation. Streaming + cancel
    fixes it; this also retires the "doesn't cancel the LLM" limitation.
- Hardening: MicMonitor.stop() is now a hard stop even mid-capture (a
  sentence arriving at the same instant the monitor triggered could leave a
  thread holding the mic for up to 15s). When both happen, the user wins.
- Verified: typed question -> real Ollama stream -> Piper -> real speakers,
  3 sentences in order, no false interrupt, spoken text remembered.
- NOT verified: spoken barge-in on v5 (user to test). Also removed a fake
  {"live": true, stt_ms: 0} record my typed test wrote to logs/latency.jsonl.

## Phase 9 — Full Integration Demo
- [x] Demo script covering all three intents
      docs/DEMO_SCRIPT.md: pre-flight steps, Part 1 (intents + memory, 6
      utterances with expected routes), Part 2 (barge-in scenarios A-D incl.
      the brief's reminder-then-weather interruption), recording notes, and
      what is/isn't verified live. scripts/demo_preflight.py checks Ollama,
      model files, audio devices, Open-Meteo, competing GPU jobs and swap
      pressure (it correctly flagged the user's running LoRA training and 90%
      swap on its first run).
- [ ] Barge-in demo (interrupt reminder confirmation to ask weather)
      NOT recorded / NOT verified live. Blocked on the user's fine-tuning job
      finishing (it throttles the LLM to ~5 tok/s) and on the user running
      scenarios A and D.

## Phase 10 — Documentation
- [x] Architecture diagram (mermaid in README.md, renders on GitHub)
- [x] Classifier accuracy reported (100%, 102/102, with the optimistic-ceiling caveat)
- [x] Latency breakdown table (from Phase 8; flagged as measured on an idle
      machine, to be re-measured after the training job ends)
- [ ] Barge-in demo (video/GIF) — not recorded yet
- [x] Cost confirmation (all components $0, table in README)
- [x] Known limitations
- [x] Lessons learned
      README.md written from measured results; unverified items are labelled.
      DEVIATION FROM THE BRIEF, stated in the README: Pipecat was specified
      for orchestration but is NOT used. The pipeline (turn-taking, duplex
      stream, barge-in) is hand-written. requirements.txt previously listed
      pipecat-ai and pydantic though neither is imported anywhere; replaced
      with the packages actually used, pinned to the tested versions.

---

## Environment Notes
- macOS 26.6.2, Apple Silicon (M4 Pro, 24GB unified memory)
- Python 3.13.5, venv created at `venv/` (not committed)
- Project dir: `/Users/aayush/VOICE AI AGENT`
- Dependencies: `requirements.txt`, pinned to the installed versions (Pipecat was dropped, see Phase 10 note)

## Open Items / Blockers
- (none open) Weather default location was resolved in Phase 2: ask each time.
