"""
LLM Engine - Low-level API wrapper for WaveSpeed AI.
Handles API calls, retries, rate limiting, and dual-model ensemble.
"""
import json
import time

import requests

import config


def call_llm(prompt, model, timeout=15):
    """
    Call a single LLM model via WaveSpeed AI API.

    Args:
        prompt: The full prompt string
        model: Model identifier (e.g., "anthropic/claude-3.7-sonnet")
        timeout: Request timeout in seconds

    Returns:
        Raw text response from the LLM
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.WAVESPEED_API_KEY}",
    }
    payload = {
        "enable_sync_mode": True,
        "model": model,
        "priority": "latency",
        "prompt": prompt,
        "reasoning": False,
    }

    response = requests.post(
        config.WAVESPEED_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("code") == 200 and data.get("data", {}).get("outputs"):
        return data["data"]["outputs"][0]

    raise RuntimeError(f"LLM API error ({model}): {data.get('message', 'Unknown')}")


def parse_json_response(text):
    """Parse JSON from LLM response, handling markdown fences."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def call_ensemble(prompt, models=None, delay=1.0):
    """
    Call multiple LLM models with the same prompt and return all responses.

    Args:
        prompt: The full prompt string
        models: List of model identifiers (defaults to config.LLM_MODELS)
        delay: Seconds between model calls (rate limiting)

    Returns:
        List of (model_name, raw_text) tuples for successful calls
    """
    models = models or config.LLM_MODELS
    results = []

    for model in models:
        model_short = model.split("/")[-1]
        try:
            raw = call_llm(prompt, model)
            results.append((model_short, raw))
        except Exception as e:
            print(f"    {model_short} failed: {e}")
        time.sleep(delay)

    return results


def ensemble_predict(prompt, models=None, delay=1.0):
    """
    Call ensemble and parse+average structured JSON predictions.

    Expected JSON format from LLM:
    {
        "direction": "up" or "down",
        "probability": 50-100,
        "target_return": float,
        "key_factors": [...],
        "risk_factors": [...],
        "reasoning": "..."
    }

    Returns:
        dict with averaged prediction and per-model details, or None on total failure
    """
    raw_results = call_ensemble(prompt, models, delay)
    if not raw_results:
        return None

    parsed = []
    for model_name, raw_text in raw_results:
        try:
            pred = parse_json_response(raw_text)
            pred["_model"] = model_name
            parsed.append(pred)
        except Exception as e:
            print(f"    {model_name} parse failed: {e}")

    if not parsed:
        return None

    # Average probabilities
    directions = [p.get("direction", "up") for p in parsed]
    probabilities = [p.get("probability", 50) for p in parsed]
    target_returns = [p.get("target_return", 0) for p in parsed]

    # Majority vote for direction
    up_count = sum(1 for d in directions if d == "up")
    avg_direction = "up" if up_count > len(directions) / 2 else "down"

    avg_prob = sum(probabilities) / len(probabilities)
    avg_return = sum(target_returns) / len(target_returns)

    # Collect all factors
    all_key_factors = []
    all_risk_factors = []
    all_reasoning = []
    for p in parsed:
        all_key_factors.extend(p.get("key_factors", []))
        all_risk_factors.extend(p.get("risk_factors", []))
        all_reasoning.append(f"[{p['_model']}] {p.get('reasoning', '')}")

    return {
        "direction": avg_direction,
        "probability": round(avg_prob, 1),
        "target_return": round(avg_return, 2),
        "key_factors": all_key_factors,
        "risk_factors": all_risk_factors,
        "reasoning": " | ".join(all_reasoning),
        "model_details": {p["_model"]: {
            "direction": p.get("direction"),
            "probability": p.get("probability"),
            "target_return": p.get("target_return"),
        } for p in parsed},
        "models_used": [p["_model"] for p in parsed],
        "agreement": len(set(directions)) == 1,
    }
