"""Text-only checks of the router's session memory, replaying real live failures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router.router import Router

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def turn(router, text, remember=True):
    r = router.route(text)
    if remember:
        router.remember(text, r)
    print(f"  > {text!r}\n    [{r.intent} @ {r.confidence:.2f}] {r.message}")
    return r


router = Router()

print("1. Follow-up refers to the previous answer")
turn(router, "How is a rainbow formed?")
r = turn(router, "Can you tell that in short?")
check("follow-up answer mentions the rainbow topic",
      any(w in r.message.lower() for w in ("rainbow", "light", "droplet", "water", "sun", "refract")))

print("\n2. 'Which city?' then a bare city name")
router = Router()
r = turn(router, "What is the weather?")
check("asked for a city", r.needs_location)
r = turn(router, "Mumbai")
check("bare 'Mumbai' routed to weather and answered", r.intent == "weather" and "Mumbai" in r.message)

print("\n3. Last city is remembered")
r = turn(router, "What's the weather now?")
check("reused Mumbai without asking", r.intent == "weather" and not r.needs_location and "Mumbai" in r.message)

print("\n4. A new explicit city overrides the remembered one")
r = turn(router, "What is the weather in London?")
check("used London", "London" in r.message)

print("\n5. A discarded (never heard) answer is not remembered")
router = Router()
router.route("Tell me about the planet Neptune")  # interrupted while thinking: no remember()
check("history stays empty", router.history == [])

print("\n6. A non-location reply while awaiting a city is not hijacked")
router = Router()
turn(router, "What is the weather?")
r = turn(router, "Actually, remind me to call mom tomorrow at 6 pm")
check("long new request routed normally", r.intent == "reminder")

sys.exit(1 if failures else 0)
