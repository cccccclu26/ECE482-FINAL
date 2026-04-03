"""
LLM Engine - OpenAI GPT-5.4 API wrapper using requests.
Handles API calls, retries, and JSON response parsing.
"""
import json

import requests

import config

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def call_llm(prompt, model=None, temperature=0.3, timeout=60):
    """
    Call GPT-5.4 via OpenAI Chat Completions API.

    Args:
        prompt: The full prompt string
        model: Model identifier (default: config.OPENAI_MODEL)
        temperature: Sampling temperature
        timeout: Request timeout in seconds

    Returns:
        Raw text response from the LLM
    """
    model = model or config.OPENAI_MODEL

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert quantitative analyst. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    response = requests.post(
        OPENAI_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]


def parse_json_response(text):
    """Parse JSON from LLM response, handling markdown fences."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def predict(prompt, model=None):
    """
    Call GPT-5.4 and parse structured JSON prediction.

    Expected JSON format from LLM:
    {
        "direction": "up" or "down",
        "probability": 50-100,
        "target_return": float,
        "key_factors": [...],
        "risk_factors": [...],
        "reasoning": "...",
        "indicator_importance": {"macd": 0.8, "rsi": 0.6, ...},
        "sentiment_weight": 0.3
    }

    Returns:
        dict with prediction, or None on failure
    """
    try:
        raw_text = call_llm(prompt, model)
    except Exception as e:
        print(f"    LLM call failed: {e}")
        return None

    try:
        pred = parse_json_response(raw_text)
    except Exception as e:
        print(f"    JSON parse failed: {e}")
        print(f"    Raw response: {raw_text[:200]}")
        return None

    pred["models_used"] = [model or config.OPENAI_MODEL]
    pred["agreement"] = True  # single model

    return pred
