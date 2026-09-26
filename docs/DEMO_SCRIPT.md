# Demo script

Goal: show all three intents by voice, then the feature that matters most —
interrupting the assistant mid-sentence — including the scenario from the brief
(interrupt a reminder confirmation to ask about the weather).

## Before recording

1. **Finish or pause any GPU-heavy job** (e.g. MLX fine-tuning). Measured on this
   machine: a concurrent LoRA training run dropped Q&A generation to ~5 tokens/s
   and pushed swap to 90%. The demo will look slow for reasons unrelated to the design.
2. `ollama serve` (if not already running).
3. `python scripts/demo_preflight.py` — should say `Ready.` with no WARN lines.
4. Delete old reminders for a clean run: `rm -f data/reminders/reminders.db`.
5. **Use speakers for the honest version, headphones for the reliable one.** With speakers
   the assistant's voice leaks into the mic and the barge-in detector can occasionally
   cut the assistant off by itself (~1 in 10 messages measured). Record the speaker
   version if you want to show the real limitation; use headphones for a clean take.
6. Quiet room. Speak at normal-to-firm volume; a whisper will not trigger an interruption.
7. Screen recording: `Cmd+Shift+5`, record with the microphone on. If using speakers,
   the mic will capture the assistant's voice too, which is what you want for the video.

Run:

```
cd "/Users/aayush/VOICE AI AGENT" && source venv/bin/activate && python src/orchestration/voice_pipeline_v4.py
```

Wait for `Listening...` before each line. Pause briefly after each sentence; it decides
you are done after ~0.7 s of silence.

## Part 1 — the three intents (about 1 minute)

| # | Say | Expected route | Expected behaviour |
|---|-----|----------------|--------------------|
| 1 | "What is the capital of Australia?" | qna | Short spoken answer (Canberra) |
| 2 | "Can you tell me more about it?" | qna | Uses memory: talks about Canberra, not "what do you mean?" |
| 3 | "Remind me to call mom tomorrow at 6 p.m." | reminder | "Got it, I'll remind you to call mom on <day> at 6:00 PM." |
| 4 | "What's the weather?" | weather | Asks "Which city would you like the weather for?" |
| 5 | "Mumbai" | weather | Reads the Mumbai forecast (a bare city name works after being asked) |
| 6 | "And what about the weather in London?" | weather | Reads London; overrides the remembered city |

What to point out: the printed line `[qna @ 1.00, …ms]` shows the intent and the
classifier's confidence; reminder is the fastest path (no LLM, ~3 ms handler), weather
is slower than you might expect (two network calls).

## Part 2 — barge-in (about 1 minute)

**Scenario A (from the brief): interrupt a reminder confirmation with a weather question**

1. Say: "Remind me to submit the report on Friday at noon."
2. While it is speaking the confirmation, cut in firmly: "Actually, what's the weather in Tokyo?"
3. Expected: playback stops within a fraction of a second, the terminal prints
   `[barge-in] interrupted mid-speech — stopping playback`, your new question is routed
   to weather and answered. The reminder is still saved (the confirmation was
   interrupted, the action was not undone).

**Scenario B: interrupt a long Q&A answer**

1. Say: "Explain how a rainbow forms."
2. Talk over it after the first sentence: "Can you tell that in short?"
3. Expected: it stops mid-answer and gives a one-sentence summary of the rainbow
   (memory + interruption working together).

**Scenario C: interrupt while it is still thinking**

1. Ask a Q&A question, and immediately (before it starts speaking) ask something else.
2. Expected: `[barge-in] interrupted while thinking — discarding in-flight response`.
   The first answer is never spoken and is not added to the conversation memory.

**Scenario D: rapid successive interruptions** — interrupt, then interrupt the
answer to your interruption. It should stay stable and keep responding to the latest thing you said.

Verified live so far: scenarios B and C, and the memory behaviour. **Not yet
verified live: A and D.** Record what actually happens rather than the ideal.

## Exit

`Ctrl+C` — prints `Bye.` and exits cleanly.

## After recording

- Trim to under ~2 minutes; export a GIF of Part 2 for the README.
- Note the model/hardware in the caption: local Llama 3.1 8B, Whisper base.en, Piper, M4 Pro.
- Re-run `python scripts/benchmark_latency.py` **with no training job running** and update the
  latency table in the README.
