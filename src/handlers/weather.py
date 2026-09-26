"""
Weather handler: extracts a location from free text (or asks for one if
missing), geocodes it via Open-Meteo's free geocoding API, fetches current
conditions + today's forecast, and formats a natural spoken-style response.

No API key required anywhere in this module.
"""
import re
from dataclasses import dataclass

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with rime",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "severe thunderstorms with hail",
}

NEEDS_LOCATION_PROMPT = "Which city would you like the weather for?"

# crude location extraction: look for "in/for/at/near/of <Place>" — "of" covers
# a common real phrasing ("what's the weather of Mumbai") missed in earlier
# testing. Capturing only consecutive capitalized words so trailing lowercase
# words like "tomorrow" or "today" aren't swept into the place name.
_LOCATION_PATTERN = re.compile(
    r"\b(?:in|for|at|near|of)\s+([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)*)"
)


@dataclass
class WeatherResult:
    ok: bool
    message: str
    needs_location: bool = False


def extract_location(text: str) -> str | None:
    match = _LOCATION_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def geocode(location: str) -> tuple[float, float, str] | None:
    resp = requests.get(
        GEOCODE_URL, params={"name": location, "count": 1}, timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results")
    if not results:
        return None
    r = results[0]
    display_name = r["name"]
    if r.get("admin1"):
        display_name += f", {r['admin1']}"
    if r.get("country"):
        display_name += f", {r['country']}"
    return r["latitude"], r["longitude"], display_name


def fetch_forecast(lat: float, lon: float) -> dict:
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def format_response(place: str, data: dict) -> str:
    current = data["current"]
    daily = data["daily"]

    temp = round(current["temperature_2m"])
    condition = WEATHER_CODES.get(current["weather_code"], "unusual conditions")
    wind = round(current["wind_speed_10m"])
    humidity = current["relative_humidity_2m"]

    high = round(daily["temperature_2m_max"][0])
    low = round(daily["temperature_2m_min"][0])
    rain_chance = daily["precipitation_probability_max"][0]

    parts = [
        f"Right now in {place}, it's {temp} degrees with {condition}.",
        f"Today's high will be {high} and the low {low} degrees.",
        f"Wind is around {wind} km/h and humidity is {humidity} percent.",
    ]
    if rain_chance is not None and rain_chance >= 30:
        parts.append(f"There's a {rain_chance} percent chance of precipitation today.")

    return " ".join(parts)


def handle(text: str, default_location: str | None = None) -> WeatherResult:
    location = extract_location(text) or default_location
    if not location:
        return WeatherResult(ok=False, message=NEEDS_LOCATION_PROMPT, needs_location=True)

    geo = geocode(location)
    if geo is None:
        return WeatherResult(
            ok=False, message=f"I couldn't find a place called {location}."
        )
    lat, lon, place = geo

    try:
        data = fetch_forecast(lat, lon)
    except requests.RequestException as e:
        return WeatherResult(ok=False, message=f"I couldn't reach the weather service: {e}")

    return WeatherResult(ok=True, message=format_response(place, data))


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "what's the weather in Mumbai"
    result = handle(text)
    print(result.message)
