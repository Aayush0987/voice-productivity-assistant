"""
Generate a synthetic labeled dataset for the intent classifier using Groq's
free-tier API. Output: data/classifier/{train,test}.json and all_examples.json

Intents: qna, reminder, weather
"""
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"
TARGET_PER_INTENT = 220
BATCH_SIZE = 25
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "classifier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

client = Groq(api_key=os.environ["GROQ_API_KEY"])

INTENT_PROMPTS = {
    "qna": (
        "Generate {n} short, varied things a person might SAY OUT LOUD to a voice "
        "assistant when asking a general knowledge / open-ended question (NOT about "
        "weather, and NOT asking to set a reminder or task). Cover a wide mix of "
        "topics: trivia, science, history, definitions, how-to, advice, opinions, "
        "math, current events phrasing, casual chit-chat questions. Vary sentence "
        "style: some formal, some very casual with fillers ('um', 'hey'), some "
        "short, some longer. Do not number them or add quotes. Output ONLY a JSON "
        "array of strings, nothing else."
    ),
    "reminder": (
        "Generate {n} short, varied things a person might SAY OUT LOUD to a voice "
        "assistant to set a reminder or task with a time/date. Cover many phrasings: "
        "'remind me to...', 'set a reminder for...', 'don't let me forget to...', "
        "'add a task to...', relative times ('in 20 minutes', 'tomorrow at 9am', "
        "'next Friday', 'every morning'), casual and formal styles, some with no "
        "explicit time mentioned. Do not number them or add quotes. Output ONLY a "
        "JSON array of strings, nothing else."
    ),
    "weather": (
        "Generate {n} short, varied things a person might SAY OUT LOUD to a voice "
        "assistant to ask about the weather. Cover variations: asking about today, "
        "tomorrow, the weekend, a specific city, no city mentioned (implying current "
        "location), asking about rain/temperature/wind/snow specifically, casual and "
        "formal phrasing. Do not number them or add quotes. Output ONLY a JSON array "
        "of strings, nothing else."
    ),
}


def request_batch(intent: str, n: int, temperature: float) -> list[str]:
    prompt = INTENT_PROMPTS[intent].format(n=n)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        items = json.loads(text[start : end + 1])
    return [str(x).strip() for x in items if str(x).strip()]


def generate_for_intent(intent: str, target: int) -> list[str]:
    seen = set()
    results = []
    temperature = 0.9
    attempts = 0
    while len(results) < target and attempts < 30:
        attempts += 1
        remaining = target - len(results)
        n = min(BATCH_SIZE, max(10, remaining))
        try:
            batch = request_batch(intent, n, temperature)
        except Exception as e:
            print(f"  [{intent}] batch error: {e}, retrying...")
            time.sleep(2)
            continue
        new_count = 0
        for phrase in batch:
            key = phrase.lower().strip()
            if key and key not in seen:
                seen.add(key)
                results.append(phrase)
                new_count += 1
        print(f"  [{intent}] +{new_count} (total {len(results)}/{target})")
        temperature = min(1.1, temperature + 0.02)  # nudge diversity if stalling
        time.sleep(1)  # be polite to free-tier rate limits
    return results[:target]


def main():
    all_examples = {}
    for intent in INTENT_PROMPTS:
        print(f"Generating examples for intent: {intent}")
        all_examples[intent] = generate_for_intent(intent, TARGET_PER_INTENT)

    with open(OUT_DIR / "all_examples.json", "w") as f:
        json.dump(all_examples, f, indent=2)

    train, test = [], []
    for intent, phrases in all_examples.items():
        random.shuffle(phrases)
        split = int(len(phrases) * 0.85)
        for p in phrases[:split]:
            train.append({"text": p, "label": intent})
        for p in phrases[split:]:
            test.append({"text": p, "label": intent})

    random.shuffle(train)
    random.shuffle(test)

    with open(OUT_DIR / "train.json", "w") as f:
        json.dump(train, f, indent=2)
    with open(OUT_DIR / "test.json", "w") as f:
        json.dump(test, f, indent=2)

    print(f"\nDone. train={len(train)} test={len(test)}")
    for intent, phrases in all_examples.items():
        print(f"  {intent}: {len(phrases)} examples")


if __name__ == "__main__":
    main()
