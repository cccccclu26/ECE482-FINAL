"""
Memory Manager - JSON-based persistent state for LLM prediction system.

Four layers of memory:
  1. Stock Context     - Per-stock learned patterns, indicator weights, sentiment ratio
  2. Prompt History    - Track prompt versions and their backtest performance
  3. Eval Logs         - Individual prediction results for analysis
  4. Indicator Learning - Track which indicators are most predictive per stock
"""
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

import config


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# All indicators the LLM can reference
ALL_INDICATORS = [
    "ema_alignment", "sma_trend", "golden_cross", "macd", "adx",
    "rsi", "stochastic", "williams_r", "cci",
    "bollinger_bands", "atr",
    "volume_ratio", "obv",
    "news_sentiment",
]


def _ensure_dirs():
    """Create memory subdirectories if they don't exist."""
    for subdir in ["stock_context", "prompt_history", "eval_logs"]:
        os.makedirs(os.path.join(config.MEMORY_DIR, subdir), exist_ok=True)


# ============================================================
# Stock Context (per-stock learned patterns + indicator weights)
# ============================================================

def load_stock_context(ticker):
    """
    Load learned context for a specific stock.

    Returns dict with keys:
        learned_pattern, historical_accuracy, known_bias,
        previous_predictions, indicator_weights, sentiment_weight,
        technical_weight, indicator_history
    """
    _ensure_dirs()
    path = os.path.join(config.MEMORY_DIR, "stock_context", f"{ticker}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            ctx = json.load(f)
        # Ensure new fields exist for backward compatibility
        ctx.setdefault("indicator_weights", {})
        ctx.setdefault("sentiment_weight", 0.5)
        ctx.setdefault("technical_weight", 0.5)
        ctx.setdefault("indicator_history", [])
        return ctx
    return {
        "ticker": ticker,
        "learned_pattern": "",
        "historical_accuracy": "N/A",
        "known_bias": "",
        "previous_predictions": [],
        "indicator_weights": {},
        "sentiment_weight": 0.5,
        "technical_weight": 0.5,
        "indicator_history": [],
    }


def save_stock_context(ticker, context):
    """Save updated stock context."""
    _ensure_dirs()
    path = os.path.join(config.MEMORY_DIR, "stock_context", f"{ticker}.json")
    context["ticker"] = ticker
    context["updated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def append_prediction_to_context(ticker, prediction_record):
    """
    Append a prediction result to the stock's context history.
    Keeps only the most recent 10 predictions.
    """
    ctx = load_stock_context(ticker)
    ctx["previous_predictions"].append(prediction_record)
    ctx["previous_predictions"] = ctx["previous_predictions"][-10:]
    save_stock_context(ticker, ctx)


def format_stock_context_for_prompt(ticker):
    """Format stock context into text for LLM prompt injection."""
    ctx = load_stock_context(ticker)

    lines = []
    if ctx.get("learned_pattern"):
        lines.append(f"Learned Pattern: {ctx['learned_pattern']}")
    if ctx.get("historical_accuracy") and ctx["historical_accuracy"] != "N/A":
        lines.append(f"Historical Accuracy: {ctx['historical_accuracy']}")
    if ctx.get("known_bias"):
        lines.append(f"Known Bias: {ctx['known_bias']}")

    # Indicator weights (learned over time)
    weights = ctx.get("indicator_weights", {})
    if weights:
        sorted_indicators = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top = sorted_indicators[:5]
        lines.append(f"\nMost Predictive Indicators (learned from {len(ctx.get('indicator_history', []))} predictions):")
        for name, score in top:
            lines.append(f"  {name}: {score:.2f}")

    # Sentiment vs Technical ratio
    sw = ctx.get("sentiment_weight", 0.5)
    tw = ctx.get("technical_weight", 0.5)
    if sw != 0.5 or tw != 0.5:
        lines.append(f"\nLearned Weighting: Technical={tw:.0%} / Sentiment={sw:.0%}")

    # Recent predictions
    preds = ctx.get("previous_predictions", [])
    if preds:
        lines.append(f"\nRecent Predictions ({len(preds)} total):")
        for p in preds[-3:]:
            date = p.get("date", "?")
            direction = p.get("predicted_direction", "?")
            prob = p.get("predicted_probability", "?")
            actual = p.get("actual_return", "?")
            correct = p.get("correct", "?")
            lines.append(
                f"  {date}: predicted {direction} ({prob}%) -> "
                f"actual {actual}% ({'CORRECT' if correct else 'WRONG'})"
            )

    if not lines:
        return "No prior context available for this stock. This is the first prediction."
    return "\n".join(lines)


# ============================================================
# Indicator Learning
# ============================================================

def update_indicator_weights(ticker, prediction, correct):
    """
    Update indicator importance weights based on prediction outcome.

    When a prediction is correct, the indicators the LLM cited as important
    get their weight increased. When wrong, they get decreased.

    Args:
        ticker: Stock ticker
        prediction: Prediction dict (must have indicator_importance)
        correct: Whether the prediction was correct
    """
    ctx = load_stock_context(ticker)
    weights = ctx.get("indicator_weights", {})

    # Get indicator importance from LLM's response
    indicator_importance = prediction.get("indicator_importance", {})
    if not indicator_importance:
        return

    # Record this data point
    ctx["indicator_history"].append({
        "date": prediction.get("date", datetime.now().strftime("%Y-%m-%d")),
        "indicators_cited": indicator_importance,
        "correct": correct,
    })
    # Keep last 50 data points
    # Keep all history for full learning coverage

    # Recompute weights from full history
    indicator_scores = defaultdict(list)
    for entry in ctx["indicator_history"]:
        for ind, importance in entry["indicators_cited"].items():
            # Score = importance * (1 if correct, -0.5 if wrong)
            score = importance * (1.0 if entry["correct"] else -0.5)
            indicator_scores[ind].append(score)

    # Average scores -> weights (clamp to 0-1)
    for ind, scores in indicator_scores.items():
        avg = sum(scores) / len(scores)
        weights[ind] = round(max(0.0, min(1.0, (avg + 1) / 2)), 3)

    ctx["indicator_weights"] = weights
    save_stock_context(ticker, ctx)


def update_sentiment_ratio(ticker, prediction, correct):
    """
    Update the learned sentiment vs technical weighting for a stock.

    Args:
        ticker: Stock ticker
        prediction: Prediction dict (must have sentiment_weight)
        correct: Whether the prediction was correct
    """
    ctx = load_stock_context(ticker)

    pred_sentiment_w = prediction.get("sentiment_weight", 0.5)
    if pred_sentiment_w is None:
        pred_sentiment_w = 0.5

    # Exponential moving average of the optimal sentiment weight
    alpha = 0.15  # learning rate
    current_sw = ctx.get("sentiment_weight", 0.5)

    if correct:
        # Move toward the sentiment weight that worked
        new_sw = current_sw + alpha * (pred_sentiment_w - current_sw)
    else:
        # Move away from the sentiment weight that failed
        new_sw = current_sw - alpha * (pred_sentiment_w - current_sw) * 0.5

    new_sw = max(0.05, min(0.95, new_sw))
    ctx["sentiment_weight"] = round(new_sw, 3)
    ctx["technical_weight"] = round(1.0 - new_sw, 3)
    save_stock_context(ticker, ctx)


def get_recommended_indicators(ticker):
    """
    Return the top indicators for a stock based on learned weights.

    Returns:
        List of (indicator_name, weight) sorted by weight descending
    """
    ctx = load_stock_context(ticker)
    weights = ctx.get("indicator_weights", {})
    if not weights:
        return []
    return sorted(weights.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# Prompt History (version tracking)
# ============================================================

def save_prompt_version(version, template_text, eval_summary=None):
    """
    Save a prompt version with its template and evaluation results.

    Args:
        version: Version string (e.g., "v1", "v2")
        template_text: The full prompt template
        eval_summary: dict with accuracy, alpha, per-stock breakdown
    """
    _ensure_dirs()
    path = os.path.join(config.MEMORY_DIR, "prompt_history", f"{version}.json")
    record = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "template_text": template_text,
        "eval_summary": eval_summary or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def load_prompt_history():
    """Load all prompt versions and their performance."""
    _ensure_dirs()
    history_dir = os.path.join(config.MEMORY_DIR, "prompt_history")
    versions = []
    for fname in sorted(os.listdir(history_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(history_dir, fname), "r", encoding="utf-8") as f:
                versions.append(json.load(f))
    return versions


# ============================================================
# Evaluation Logs (per-prediction tracking)
# ============================================================

def log_prediction(ticker, date, prediction, actual_return=None):
    """
    Log a single prediction for later analysis.

    Args:
        ticker: Stock ticker
        date: Prediction date string
        prediction: dict from LLM (direction, probability, etc.)
        actual_return: Actual return over horizon (filled in later)
    """
    _ensure_dirs()
    path = os.path.join(config.MEMORY_DIR, "eval_logs", f"{ticker}_predictions.json")

    logs = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            logs = json.load(f)

    record = {
        "date": date,
        "predicted_direction": prediction.get("direction"),
        "predicted_probability": prediction.get("probability"),
        "predicted_return": prediction.get("target_return"),
        "key_factors": prediction.get("key_factors", []),
        "indicator_importance": prediction.get("indicator_importance", {}),
        "sentiment_weight": prediction.get("sentiment_weight"),
        "models_used": prediction.get("models_used", []),
        "agreement": prediction.get("agreement"),
        "actual_return": actual_return,
        "correct": None,
        "logged_at": datetime.now().isoformat(),
    }
    logs.append(record)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    return record


def update_prediction_actuals(ticker, date, actual_return):
    """
    Fill in the actual return for a previously logged prediction.
    Also updates the correct/incorrect flag and triggers indicator learning.
    """
    path = os.path.join(config.MEMORY_DIR, "eval_logs", f"{ticker}_predictions.json")
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    for entry in logs:
        if entry["date"] == date and entry["actual_return"] is None:
            entry["actual_return"] = actual_return
            predicted_up = entry["predicted_direction"] == "up"
            actually_up = actual_return > 0
            entry["correct"] = bool(predicted_up == actually_up)

            # Trigger indicator learning
            update_indicator_weights(ticker, entry, entry["correct"])
            update_sentiment_ratio(ticker, entry, entry["correct"])
            break

    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def compute_accuracy(ticker=None):
    """
    Compute prediction accuracy from eval logs.

    Args:
        ticker: Specific ticker, or None for all tickers

    Returns:
        dict with total, correct, accuracy, per_ticker breakdown
    """
    _ensure_dirs()
    log_dir = os.path.join(config.MEMORY_DIR, "eval_logs")
    results = {}

    files = os.listdir(log_dir)
    for fname in files:
        if not fname.endswith("_predictions.json"):
            continue
        t = fname.replace("_predictions.json", "")
        if ticker and t != ticker:
            continue

        with open(os.path.join(log_dir, fname), "r", encoding="utf-8") as f:
            logs = json.load(f)

        evaluated = [e for e in logs if e.get("correct") is not None]
        correct = sum(1 for e in evaluated if e["correct"])
        total = len(evaluated)
        acc = correct / total * 100 if total > 0 else 0

        results[t] = {"total": total, "correct": correct, "accuracy": round(acc, 1)}

    # Aggregate
    all_total = sum(r["total"] for r in results.values())
    all_correct = sum(r["correct"] for r in results.values())
    all_acc = all_correct / all_total * 100 if all_total > 0 else 0

    return {
        "overall": {"total": all_total, "correct": all_correct, "accuracy": round(all_acc, 1)},
        "per_ticker": results,
    }
