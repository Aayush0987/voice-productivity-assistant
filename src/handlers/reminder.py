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
    r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(remind me to|remind me|set a reminder to|set a reminder for|"
    r"don'?t let me forget to|add a task to|add a reminder to|"
    r"i need to|please remind me to)\s*",
    re.IGNORECASE,
)

_TRAILING_CONNECTORS = re.compile(r"\s*(that|to)\s*$", re.IGNORECASE)

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


@dataclass
class ReminderResult:
    ok: bool
    message: str
    task: str | None = None
    when: datetime | None = None


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


def extract_task_and_datetime(text: str) -> tuple[str, datetime | None]:
    text = _normalize_ampm(text)
    settings = {
        "PREFER_DATES_FROM": "future",
        "RETURN_AS_TIMEZONE_AWARE": False,
    }
    found = search_dates(text, languages=["en"], settings=settings)

    when = None
    remaining = text
    if found:
        # take the longest matched date phrase (most specific)
        phrase, _ = max(found, key=lambda p: len(p[0]))
        # search_dates can miscompute the merged datetime for compound phrases
        # (e.g. "tomorrow at 9am" loses the time and keeps current time instead) -
        # re-parsing the isolated phrase directly is reliable.
        when = dateparser.parse(phrase, languages=["en"], settings=settings)
        remaining = text.replace(phrase, " ")

    task = _LEADING_FILLERS.sub("", remaining).strip()
    if when is not None:
        task = _DANGLING_TIME_QUALIFIER.sub("", task).strip()
    task = _TRAILING_CONNECTORS.sub("", task).strip()
    task = re.sub(r"\s{2,}", " ", task).strip(" ,.")

    if not task:
        task = remaining.strip(" ,.")

    return task, when


def handle(text: str) -> ReminderResult:
    task, when = extract_task_and_datetime(text)

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
    else:
        message = f"Got it, I've added a reminder to {task}. No specific time was mentioned."

    return ReminderResult(ok=True, message=message, task=task, when=when)


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
