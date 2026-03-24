"""
Memory Manager - JSON-based persistent state for LLM prediction system.

Three layers of memory:
  1. Stock Context  - Per-stock learned patterns and biases
  2. Prompt History - Track prompt versions and their backtest performance
  3. Eval Logs      - Individual prediction results for analysis
"""
import json
import os
from datetime import datetime

import config


def _ensure_dirs():
    """Create memory subdirectories if they don't exist."""
    for subdir in ["stock_context", "prompt_history", "eval_logs"]:
        os.makedirs(os.path.join(config.MEMORY_DIR, subdir), exist_ok=True)


# ============================================================
# Stock Context (per-stock learned patterns)
# ============================================================

def load_stock_context(ticker):
    """
    Load learned context for a specific stock.

    Returns dict with keys:
        learned_pattern, historical_accuracy, known_bias,
        previous_predictions (list of recent prediction+outcome pairs)
    """
    _ensure_dirs()
    path = os.path.join(config.MEMORY_DIR, "stock_context", f"{ticker}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "ticker": ticker,
        "learned_pattern": "",
        "historical_accuracy": "N/A",
        "known_bias": "",
        "previous_predictions": [],
    }


def save_stock_context(ticker, context):
    """Save updated stock context."""
    _ensure_dirs()
    path = os.path.join(config.MEMORY_DIR, "stock_context", f"{ticker}.json")
    context["ticker"] = ticker
    context["updated_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)


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

    # Recent predictions
    preds = ctx.get("previous_predictions", [])
    if preds:
        lines.append(f"\nRecent Predictions ({len(preds)} total):")
        for p in preds[-3:]:  # Show last 3
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
        return "No prior context available for this stock."
    return "\n".join(lines)


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
        json.dump(record, f, indent=2, ensure_ascii=False)


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
        "models_used": prediction.get("models_used", []),
        "agreement": prediction.get("agreement"),
        "actual_return": actual_return,
        "correct": None,
        "logged_at": datetime.now().isoformat(),
    }
    logs.append(record)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    return record


def update_prediction_actuals(ticker, date, actual_return):
    """
    Fill in the actual return for a previously logged prediction.
    Also updates the correct/incorrect flag.
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
            entry["correct"] = predicted_up == actually_up
            break

    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


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
