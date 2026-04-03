"""
LLM Predictor - Core prediction engine using GPT-5.4 as the decision maker.

The LLM receives all technical indicators + news + per-stock context and
outputs a structured prediction with indicator importance ratings.

Adaptive learning loop:
  1. Load prompt template (versioned in prompts/ directory)
  2. Inject per-stock context (including learned indicator weights) + all indicators + news
  3. Call GPT-5.4
  4. Parse prediction, extract indicator_importance and sentiment_weight
  5. After backtest evaluation, update per-stock indicator weights and sentiment ratio
"""
import os
import time
from datetime import datetime

import config
from llm_engine import predict as llm_predict
from technical import compute_indicators, format_indicators_for_prompt
from data_fetcher import fetch_news, format_news_for_prompt
from memory import (
    format_stock_context_for_prompt,
    log_prediction,
    append_prediction_to_context,
)


class LLMPredictor:
    """LLM-based stock prediction engine with adaptive indicator learning."""

    def __init__(self, prompt_version="v1"):
        self.prompt_version = prompt_version
        self.template = self._load_template(prompt_version)
        print(f"[LLM Predictor] Loaded prompt template: {prompt_version}")
        print(f"[LLM Predictor] Model: {config.OPENAI_MODEL}")

    def _load_template(self, version):
        """Load a prompt template from the prompts/ directory."""
        path = os.path.join(config.PROMPTS_DIR, f"{version}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def build_prompt(self, ticker, date, technical_data, news_data, stock_context):
        """
        Construct the full prompt by filling in the template.

        Args:
            ticker: Stock ticker
            date: Prediction date string
            technical_data: Formatted technical indicators string (all indicators)
            news_data: Formatted news string
            stock_context: Formatted per-stock context string (includes learned weights)

        Returns:
            Complete prompt string ready for LLM
        """
        return self.template.format(
            ticker=ticker,
            date=date,
            horizon=config.PREDICTION_HORIZON_DAYS,
            news_days=config.NEWS_LOOKBACK_DAYS,
            technical_data=technical_data,
            news_data=news_data,
            stock_context=stock_context,
        )

    def predict(self, ticker, date, price_data, fetch_live_news=True):
        """
        Make a prediction for a single stock at a given date.

        Args:
            ticker: Stock ticker symbol
            date: Prediction date (datetime or string "YYYY-MM-DD")
            price_data: DataFrame with OHLCV for this ticker
            fetch_live_news: Whether to fetch news from API

        Returns:
            Prediction dict with direction, probability, indicator_importance,
            sentiment_weight, reasoning, etc.
            Returns a default neutral prediction on failure.
        """
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime("%Y-%m-%d")

        # 1. Compute ALL technical indicators (no look-ahead)
        indicators = compute_indicators(price_data, as_of_date=date)
        tech_text = format_indicators_for_prompt(indicators)

        # 2. Fetch news
        if fetch_live_news:
            news_list = fetch_news(ticker, date_str, limit=config.DEFAULT_NEWS_LIMIT)
            news_text = format_news_for_prompt(news_list)
        else:
            news_text = "News data not available for this date."

        # 3. Load per-stock context (includes learned indicator weights + sentiment ratio)
        context_text = format_stock_context_for_prompt(ticker)

        # 4. Build prompt
        prompt = self.build_prompt(ticker, date_str, tech_text, news_text, context_text)

        # 5. Call GPT-5.4
        prediction = llm_predict(prompt)

        if prediction is None:
            prediction = self._default_prediction()

        prediction["ticker"] = ticker
        prediction["date"] = date_str
        prediction["prompt_version"] = self.prompt_version

        # 6. Log prediction (including indicator_importance and sentiment_weight)
        log_prediction(ticker, date_str, prediction)

        return prediction

    def predict_all(self, tickers, date, price_data_dict, fetch_live_news=True, delay=1.5):
        """
        Predict all tickers at a given date.

        Args:
            tickers: List of ticker symbols
            date: Prediction date
            price_data_dict: Dict of {ticker: DataFrame}
            fetch_live_news: Whether to fetch news
            delay: Seconds between tickers (rate limiting)

        Returns:
            Dict of {ticker: prediction}
        """
        predictions = {}
        for i, ticker in enumerate(tickers):
            print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
            pred = self.predict(ticker, date, price_data_dict[ticker], fetch_live_news)
            prob_str = f"{pred['probability']}%"
            print(f"{pred['direction']} ({prob_str})", flush=True)
            predictions[ticker] = pred
            if i < len(tickers) - 1:
                time.sleep(delay)
        return predictions

    def _default_prediction(self):
        """Return a neutral default prediction when LLM fails."""
        return {
            "direction": "up",
            "probability": 50,
            "target_return": 0.0,
            "key_factors": [],
            "risk_factors": [],
            "reasoning": "LLM prediction failed, using neutral default.",
            "models_used": [],
            "agreement": False,
            "indicator_importance": {},
            "sentiment_weight": 0.5,
        }


def predictions_to_weights(predictions, threshold=None):
    """
    Convert LLM predictions to portfolio weights with defensive cash mode.

    Stocks predicted "up" with probability >= threshold get allocated.
    Weight is proportional to probability. Remaining goes to CASH.

    Args:
        predictions: Dict of {ticker: prediction_dict}
        threshold: Min probability to include (default: config.MIN_PROBABILITY)

    Returns:
        Dict of {ticker: weight} including possible "CASH" key
    """
    threshold = threshold or config.MIN_PROBABILITY

    qualifying = {}
    for ticker, pred in predictions.items():
        if pred["direction"] == "up" and pred["probability"] / 100 >= threshold:
            qualifying[ticker] = pred["probability"] / 100

    n_total = len(predictions)
    n_qualifying = len(qualifying)

    if n_qualifying == 0:
        return {"CASH": 1.0}

    equity_pct = n_qualifying / n_total
    total_score = sum(qualifying.values())
    weights = {t: (s / total_score) * equity_pct for t, s in qualifying.items()}

    cash_pct = 1.0 - equity_pct
    if cash_pct > 0.001:
        weights["CASH"] = cash_pct

    return weights
