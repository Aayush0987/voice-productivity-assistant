"""
Run before recording a demo. Checks everything the demo depends on and warns
about things that would make it look worse than it is (a competing GPU job,
memory pressure, a cold LLM).

usage: python scripts/demo_preflight.py
"""
import re
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"[{status:<4}] {name}" + (f": {detail}" if detail else ""))


def check_ollama():
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=3).json()
    except requests.RequestException:
        return report("FAIL", "Ollama", "not reachable, run `ollama serve`")
    names = [m["name"] for m in tags.get("models", [])]
    if "llama3.1:8b" not in names:
        return report("FAIL", "Ollama model", "llama3.1:8b not pulled")
    loaded = [m["name"] for m in requests.get("http://localhost:11434/api/ps", timeout=3).json().get("models", [])]
    if "llama3.1:8b" in loaded:
        report("PASS", "Ollama + llama3.1:8b", "loaded in memory (no cold start)")
    else:
        report("WARN", "Ollama + llama3.1:8b", "installed but not loaded; first answer will be slow (the pipeline warms it at start)")


def check_files():
    for label, path in [
        ("intent classifier", ROOT / "models" / "intent_classifier" / "config.json"),
        ("Piper voice", ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"),
    ]:
        report("PASS" if path.exists() else "FAIL", label, "" if path.exists() else f"missing {path.relative_to(ROOT)}")


def check_audio():
    try:
        import sounddevice as sd

        inp, out = sd.query_devices(kind="input"), sd.query_devices(kind="output")
        report("PASS", "audio devices", f"in: {inp['name']} | out: {out['name']}")
        if "bluetooth" in (inp["name"] + out["name"]).lower() or "airpods" in (inp["name"] + out["name"]).lower():
            report("WARN", "bluetooth audio", "adds latency and can switch to a low-quality call profile")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "audio devices", str(e))


def check_weather_api():
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": "Mumbai", "count": 1}, timeout=8)
        r.raise_for_status()
        report("PASS", "Open-Meteo reachable")
    except requests.RequestException as e:
        report("FAIL", "Open-Meteo", str(e))


def check_contention():
    ps = subprocess.run(["ps", "-Ao", "command="], capture_output=True, text=True).stdout
    heavy = sorted({m.group(0) for m in re.finditer(r"mlx_lm[^\s]*\s+\w+|--train|torch\.distributed|stable-diffusion|comfyui", ps, re.I)})
    if heavy:
        report("WARN", "competing GPU/training job", f"found {heavy}; it throttles the LLM (measured 5 tok/s) and skews latency numbers")
    else:
        report("PASS", "no competing training job")

    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", swap)
    if m and float(m.group(1)) > 0:
        used = float(m.group(2)) / float(m.group(1))
        report("PASS" if used < 0.5 else "WARN", "swap usage", f"{used:.0%} of {float(m.group(1))/1024:.0f} GB" + ("" if used < 0.5 else "; memory pressure slows everything"))


def check_reminders():
    db = ROOT / "data" / "reminders" / "reminders.db"
    if db.exists():
        import sqlite3

        n = sqlite3.connect(db).execute("select count(*) from reminders").fetchone()[0]
        report("PASS" if n == 0 else "WARN", "reminders DB", f"{n} saved reminders" + ("" if n == 0 else " (delete data/reminders/reminders.db for a clean demo)"))
    else:
        report("PASS", "reminders DB", "empty")


if __name__ == "__main__":
    for check in (check_ollama, check_files, check_audio, check_weather_api, check_contention, check_reminders):
        check()
    print()
    if "FAIL" in results:
        print("Not ready: fix the FAIL items.")
        sys.exit(1)
    print("Ready." if "WARN" not in results else "Runnable, but read the WARN items first.")
