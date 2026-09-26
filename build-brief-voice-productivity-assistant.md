# Build Brief: Multi-Function Voice Productivity Assistant

## Project Summary
A real-time, voice-driven personal productivity assistant with three routed
functionalities (Q&A, reminders, weather), built entirely with free/open-source
components — no cloud infrastructure, no paid APIs, no credit card required anywhere
in the stack. Runs fully on Apple Silicon (M4 Pro, 24GB unified memory).

## Cost Constraint (strict)
Every component in this project must be free with no payment method required.
Confirmed free, no-card options to use:
- **Weather data**: Open-Meteo (https://open-meteo.com) — no API key, no signup, no card
- **Synthetic training data generation**: Groq API (free tier, email signup only,
  no card) — used only to generate labeled example phrases for the intent classifier,
  not used at inference time
- **STT, TTS, VAD, LLM, classifier training**: all open-source, run locally, no
  ongoing cost

Do not substitute any paid/metered service (e.g., OpenWeatherMap, OpenAI API,
AWS/Azure/GCP services) at any point in this build.

## Functionalities
1. **General Q&A** — open-ended questions, answered by a local LLM
2. **Reminder/Task Setting** — extracts (task, datetime) from speech, stores locally
3. **Weather Lookup** — extracts location (or uses a default), calls Open-Meteo,
   returns a spoken-style summary

## Tech Stack
- **STT**: Whisper via `faster-whisper`
- **VAD**: Silero VAD
- **Orchestration**: Pipecat
- **Intent classifier**: small fine-tuned encoder (e.g., DistilBERT) — routes
  transcript to `qna` / `reminder` / `weather`
- **Q&A LLM**: a separate local open-source LLM (kept independent from the
  Project #4 fine-tuned model — different purpose, not reused)
- **Weather**: Open-Meteo API (free, no key)
- **Storage**: local JSON or SQLite for reminders
- **TTS**: Piper
- **Environment**: macOS, M4 Pro, fully local

---

## Phase 0 — Study
- STT/TTS/VAD fundamentals, audio sampling basics
- Barge-in / interruption handling patterns
- Text classification basics (encoder-only models, not generation)
- Function-calling / tool-routing patterns in voice assistants
- Structured extraction (pulling task + datetime from free text)
- Pipecat's pipeline/frame architecture

## Phase 1 — Intent Classifier (build and validate in isolation first)
- Generate a synthetic labeled dataset: ~150–300 example phrases per intent
  (`qna`, `reminder`, `weather`), bootstrapped using Groq's free tier for varied phrasing
- Fine-tune a small classifier (e.g., DistilBERT) on this dataset
- Evaluate accuracy on a held-out test set — record this number for the README
- Validate with typed text input before introducing audio

## Phase 2 — Individual Functionality Handlers (text-only, no voice yet)
- **Q&A handler**: routes to local LLM, generates answer
- **Reminder handler**: extracts (task, datetime) from request, stores to local
  JSON/SQLite, confirms back to user
- **Weather handler**: extracts location (or defaults to a fixed location), calls
  Open-Meteo, formats a natural spoken-style response
- Test all three handlers independently via text input

## Phase 3 — Router Integration
- Wire together: text input → intent classifier → correct handler → response text
- Test ambiguous/edge-case phrasings to see how the classifier behaves

## Phase 4 — STT Integration
- Microphone → Whisper → transcript → feeds into the Phase 3 router

## Phase 5 — VAD + Turn-Taking
- Silero VAD for end-of-speech detection
- Basic turn-taking: user speaks → silence detected → pipeline triggers response

## Phase 6 — TTS Integration
- Response text → Piper → audio output
- Stream text to TTS incrementally where possible to reduce perceived latency

## Phase 7 — Barge-In / Interruption Handling (core differentiator — budget the most time here)
- Detect user speech while the agent is talking
- On interruption: stop TTS playback immediately, discard in-flight response,
  start listening to the new input
- Test edge cases: interrupting mid-reminder-confirmation, interrupting
  mid-weather-report, rapid successive interruptions

## Phase 8 — Latency Measurement
- Log per-stage latency: STT, intent classification, handler execution,
  LLM generation (Q&A only), TTS
- Note and highlight: reminder/weather paths should be faster than the Q&A path
  since they skip LLM generation — a good architecture point for the README

## Phase 9 — Full Integration Demo
- Demo script covering all three intents
- Include a barge-in example: user interrupts a reminder confirmation to ask a
  weather question instead — demonstrates routing and interruption handling
  working together

## Phase 10 — Documentation
README should include:
- Architecture diagram (intent classifier + 3 handlers + audio pipeline)
- Intent classifier accuracy on held-out test set
- Per-intent, per-stage latency breakdown table
- Barge-in demo (video/GIF strongly recommended — inherently audio/visual)
- Confirmation that the entire stack is free/local (list each component and its cost = $0)
- Known limitations
- Lessons learned

---

## Success Criteria
- Intent classifier achieves solid accuracy on held-out test data (report exact number)
- All three functionalities work correctly end-to-end by voice
- Barge-in interruption reliably works across all three intents
- Entire stack runs with zero cost and no payment method used anywhere
- Documented, reproducible, interview-ready README
