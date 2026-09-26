"""
Reminder handler: extracts (task, datetime) from free text and stores it in
a local SQLite database. No network calls, no external services.
"""
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import dateparser
from dateparser.search import search_dates

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "reminders" / "reminders.db"

# Phrases to strip off the front once the datetime phrase has been removed.
# Optional polite prefixes ("can/could/would you", "please") come before the
# core verb phrase, e.g. "Can you remind me to call mom..." — a real voice
# transcript that the earlier anchored-only pattern missed entirely.
_LEADING_FILLERS = re.compile(
    r"^(?:(?:actually|also|okay|ok|hey|so|and|um|uh|well|then|now)[,\s]+)*"
    r"(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(remind me to|remind me|set a reminder to|set a reminder for|"
    r"don'?t let me forget to|add a task to|add a reminder to|"
    r"i need to|please remind me to)\s*",
    re.IGNORECASE,
)

_TRAILING_CONNECTORS = re.compile(r"\s*\b(that|to|next|this|on|at|by|for|in)\s*$", re.IGNORECASE)

# search_dates sometimes cuts a matched phrase short right before a trailing
# time-of-day word, e.g. "tomorrow at 6 pm in the" leaves a dangling "evening?"
# in the task text even though the time was already correctly captured from
# "6 pm". Drop the stray qualifier when a datetime was already found.
_DANGLING_TIME_QUALIFIER = re.compile(
    r"\b(?:in the\s+)?(morning|afternoon|evening|night)\b[?!.]*\s*$",
    re.IGNORECASE,
)

# Whisper transcribes "6pm" as "6 p.m." — the periods break dateparser's
# tokenization (it splits "p.m." into stray fragments like a lone "m").
# Normalize before extraction so both typed and voice input parse the same way.
_AM_PM_DOTS = re.compile(r"\b([ap])\.\s*m\.?", re.IGNORECASE)


def _normalize_ampm(text: str) -> str:
    return _AM_PM_DOTS.sub(lambda m: m.group(1).lower() + "m", text)


# dateparser cannot read "at 6 in the evening", "evening at 6" or a bare "at 6":
# it silently substitutes the CURRENT clock time, so the reminder would announce
# a time the user never said. Rewrite them to an explicit "6 pm" first.
_HOUR_IN_PART = re.compile(
    r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:o'?clock\s+)?(?:in the|at)\s+(morning|afternoon|evening|night)\b",
    re.IGNORECASE,
)
_PART_AT_HOUR = re.compile(
    r"\b(morning|afternoon|evening|night)\s+at\s+(\d{1,2})(?::(\d{2}))?\b", re.IGNORECASE
)
_BARE_HOUR = re.compile(
    r"\bat\s+(\d{1,2})(?::(\d{2}))?\b(?!\s*(?:am|pm|a\.m|p\.m|o'?clock|minutes?|hours?|in the))",
    re.IGNORECASE,
)


def _meridiem(hour: int, part: str | None) -> str:
    if part in ("afternoon", "evening"):
        return "pm"
    if part == "morning":
        return "am"
    if part == "night":
        return "am" if hour == 12 or hour <= 4 else "pm"
    return "pm" if hour == 12 or 1 <= hour <= 6 else "am"  # bare hour: "at 6" -> 6 pm, "at 8" -> 8 am


def _fmt(hour: str, minute: str | None, part: str | None) -> str:
    h = int(hour)
    return f"{h}{':' + minute if minute else ''} {_meridiem(h, part)}"


def _normalize_times(text: str) -> str:
    text = _HOUR_IN_PART.sub(lambda m: _fmt(m.group(1), m.group(2), m.group(3).lower()), text)
    text = _PART_AT_HOUR.sub(lambda m: _fmt(m.group(2), m.group(3), m.group(1).lower()), text)
    return _BARE_HOUR.sub(lambda m: "at " + _fmt(m.group(1), m.group(2), None), text)


_EXPLICIT_TIME = re.compile(
    r"(\d{1,2}(:\d{2})?\s*(am|pm)\b|\bnoon\b|\bmidnight\b|\d{1,2}:\d{2}|"
    r"\b\d+\s*(second|minute|hour)s?\b|\ban?\s+(hour|minute)\b|\bhalf an hour\b)",
    re.IGNORECASE,
)
DEFAULT_HOUR = 9


@dataclass
class ReminderResult:
    ok: bool
    message: str
    task: str | None = None
    when: datetime | None = None
    time_assumed: bool = False


def _init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            due_at TEXT,
            raw_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def extract_task_and_datetime(text: str) -> tuple[str, datetime | None, bool]:
    """Returns (task, datetime, time_assumed). time_assumed is True when the
    user gave a date but no time, so a default was filled in."""
    text = _normalize_times(_normalize_ampm(text))
    settings = {
        "PREFER_DATES_FROM": "future",
        "RETURN_AS_TIMEZONE_AWARE": False,
    }
    found = search_dates(text, languages=["en"], settings=settings)

    when = None
    time_assumed = False
    remaining = text
    if found:
        # take the longest matched date phrase (most specific)
        phrase, _ = max(found, key=lambda p: len(p[0]))
        # search_dates can miscompute the merged datetime for compound phrases
        # (e.g. "tomorrow at 9am" loses the time and keeps current time instead) -
        # re-parsing the isolated phrase directly is reliable.
        when = dateparser.parse(phrase, languages=["en"], settings=settings)
        # A date with no time of day comes back stamped with the current clock time;
        # announcing that as if the user said it would be wrong. Use a stated default.
        if when is not None and not _EXPLICIT_TIME.search(phrase):
            when = when.replace(hour=DEFAULT_HOUR, minute=0, second=0, microsecond=0)
            time_assumed = True
        remaining = text.replace(phrase, " ")

    task = _LEADING_FILLERS.sub("", remaining).strip(" ,.?!")
    if when is not None:
        task = _DANGLING_TIME_QUALIFIER.sub("", task).strip(" ,.?!")
    task = _TRAILING_CONNECTORS.sub("", task).strip(" ,.?!")
    task = re.sub(r"\s{2,}", " ", task).strip(" ,.?!")

    if not task:
        task = remaining.strip(" ,.")

    return task, when, time_assumed


def handle(text: str) -> ReminderResult:
    task, when, time_assumed = extract_task_and_datetime(text)

    if not task:
        return ReminderResult(ok=False, message="I couldn't figure out what to remind you about.")

    conn = _init_db()
    conn.execute(
        "INSERT INTO reminders (task, due_at, raw_text, created_at) VALUES (?, ?, ?, ?)",
        (task, when.isoformat() if when else None, text, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    if when:
        when_str = when.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
        message = f"Got it, I'll remind you to {task} on {when_str}."
        if time_assumed:
            message += f" You didn't give a time, so I picked {DEFAULT_HOUR} AM."
    else:
        message = f"Got it, I've added a reminder to {task}. No specific time was mentioned."

    return ReminderResult(ok=True, message=message, task=task, when=when, time_assumed=time_assumed)


def list_reminders() -> list[dict]:
    conn = _init_db()
    rows = conn.execute(
        "SELECT id, task, due_at, raw_text, created_at FROM reminders ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "task": r[1], "due_at": r[2], "raw_text": r[3], "created_at": r[4]}
        for r in rows
    ]


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "remind me to call mom tomorrow at 6pm"
    result = handle(text)
    print(result.message)
