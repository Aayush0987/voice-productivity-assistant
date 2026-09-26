"""Append per-turn stage latencies to logs/latency.jsonl."""
import json
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "latency.jsonl"


def log_turn(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
