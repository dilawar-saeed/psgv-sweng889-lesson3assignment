import os

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def generate_guidance(inputs, prediction):
    prompt = f"""You are a concise bike-demand assistant.
The machine-learning model predicted exactly {prediction} rentals.
Conditions: month {inputs['Month']}, day {inputs['DayOfWeek']}, hour {inputs['Hour']}:00, season {inputs['Seasons']}, holiday {inputs['Holiday']}, temperature {inputs['Temperature(°C)']} C, humidity {inputs['Humidity(%)']}%, rainfall {inputs['Rainfall(mm)']} mm, snowfall {inputs['Snowfall (cm)']} cm.

Write exactly 1 or 2 short sentences. Classify demand as low, moderate, or high based on the supplied prediction, then give one-two practical suggestions. Use the exact prediction; do not calculate a different number. Do not use headings, bullets, station-level claims, or the phrase 'for the operator'."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return (text, None) if text else (None, "Ollama returned an empty response.")
    except (requests.RequestException, ValueError, KeyError) as exc:
        return None, f"Ollama request failed: {exc}"
