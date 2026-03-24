"""
Backtesting Framework - Compare LLM predictions vs LSTM vs SPY benchmark.

Supports:
  - LLM-driven portfolio backtest
  - LSTM-driven portfolio backtest
  - Side-by-side comparison
  - Per-stock accuracy analysis
"""
import json
import os
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

import config
from data_fetcher import fetch_all_price_data
from llm_predictor import LLMPredictor, predictions_to_weights
from memory import update_prediction_actuals, append_prediction_to_context


def _get_rebalance_dates(trading_days, start_date, end_date, rebalance_days):
    """Get rebalance dates from trading day index."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    valid = trading_days[(trading_days >= start_ts) & (trading_days <= end_ts)]
    dates = []
    i = 0
    while i < len(valid):
        dates.append(valid[i])
        i += rebalance_days
    return dates, valid


def _compute_period_return(weights, price_data, rebal_date, next_rebal):
    """Compute weighted portfolio return over a hold period."""
    period_return = 0.0
    for ticker, weight in weights.items():
        if ticker == "CASH":
            continue
        ticker_data = price_data[ticker]
        entry_rows = ticker_data[ticker_data.index >= rebal_date]
        exit_rows = ticker_data[ticker_data.index >= next_rebal]
        if len(entry_rows) < 1 or len(exit_rows) < 1:
            continue
        entry_price = entry_rows.iloc[0]["Close"]
        exit_price = exit_rows.iloc[0]["Close"]
        ret = (exit_price - entry_price) / entry_price
        period_return += weight * ret
    return period_return


def run_llm_backtest(start_date, end_date, tickers=None, prompt_version="v1",
                     rebalance_days=None, threshold=None):
    """
    Run portfolio backtest using LLM predictions vs SPY benchmark.

    Args:
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        tickers: List of tickers (default: config.TICKERS)
        prompt_version: Prompt template version to use
        rebalance_days: Days between rebalances (default: config.REBALANCE_DAYS)
        threshold: Min P(up) for inclusion (default: config.MIN_PROBABILITY)

    Returns:
        dict with portfolio_history, final results, accuracy stats
    """
    tickers = tickers or config.TICKERS
    rebalance_days = rebalance_days or config.REBALANCE_DAYS
    threshold = threshold or config.MIN_PROBABILITY

    print(f"\n{'='*65}")
    print(f"LLM BACKTEST (prompt: {prompt_version})")
    print(f"{'='*65}")
    print(f"Period:     {start_date} to {end_date}")
    print(f"Tickers:    {', '.join(tickers)}")
    print(f"Rebalance:  every {rebalance_days} trading days")
    print(f"Threshold:  P(up) >= {threshold:.0%}")
    print(f"{'='*65}\n")

    # Initialize predictor
    predictor = LLMPredictor(prompt_version=prompt_version)

    # Fetch price data
    print("Fetching price data...")
    all_tickers = list(set(tickers + ["SPY"]))
    price_data = fetch_all_price_data(all_tickers, start_date, end_date)

    spy_data = price_data["SPY"]
    rebalance_dates, valid_days = _get_rebalance_dates(
        spy_data.index, start_date, end_date, rebalance_days
    )
    print(f"Rebalance dates: {len(rebalance_dates)}\n")

    # Run backtest
    portfolio_value = config.INITIAL_CAPITAL
    spy_value = config.INITIAL_CAPITAL
    history = []

    for r_idx, rebal_date in enumerate(rebalance_dates):
        # Next rebalance date
        if r_idx + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[r_idx + 1]
        else:
            future = valid_days[valid_days > rebal_date]
            next_rebal = future[-1] if len(future) > 0 else rebal_date

        date_str = rebal_date.strftime("%Y-%m-%d")
        next_str = next_rebal.strftime("%Y-%m-%d")
        print(f"[{r_idx+1:02d}/{len(rebalance_dates)}] {date_str} -> {next_str}")

        # LLM predictions for all tickers
        predictions = predictor.predict_all(tickers, rebal_date, price_data)

        # Convert to portfolio weights
        weights = predictions_to_weights(predictions, threshold=threshold)
        equity_tickers = [t for t in weights if t != "CASH"]
        cash_pct = weights.get("CASH", 0.0)

        print(f"  Portfolio: {len(equity_tickers)} stocks, cash={cash_pct:.0%}")

        # Portfolio return
        period_return = _compute_period_return(weights, price_data, rebal_date, next_rebal)
        portfolio_value *= (1 + period_return)

        # SPY return
        spy_entry = spy_data[spy_data.index >= rebal_date]
        spy_exit = spy_data[spy_data.index >= next_rebal]
        if len(spy_entry) > 0 and len(spy_exit) > 0:
            spy_ret = (spy_exit.iloc[0]["Close"] - spy_entry.iloc[0]["Close"]) / spy_entry.iloc[0]["Close"]
        else:
            spy_ret = 0.0
        spy_value *= (1 + spy_ret)

        # Update prediction actuals
        for ticker in equity_tickers:
            td = price_data[ticker]
            entry_rows = td[td.index >= rebal_date]
            exit_rows = td[td.index >= next_rebal]
            if len(entry_rows) > 0 and len(exit_rows) > 0:
                actual_ret = (exit_rows.iloc[0]["Close"] - entry_rows.iloc[0]["Close"]) / entry_rows.iloc[0]["Close"] * 100
                update_prediction_actuals(ticker, date_str, round(actual_ret, 2))
                pred = predictions[ticker]
                append_prediction_to_context(ticker, {
                    "date": date_str,
                    "predicted_direction": pred["direction"],
                    "predicted_probability": pred["probability"],
                    "actual_return": round(actual_ret, 2),
                    "correct": (pred["direction"] == "up") == (actual_ret > 0),
                })

        print(f"  Return: {period_return:+.2%} | SPY: {spy_ret:+.2%} | Portfolio: ${portfolio_value:,.0f} | SPY: ${spy_value:,.0f}")

        history.append({
            "date": date_str,
            "weights": {t: round(w, 4) for t, w in weights.items()},
            "predictions": {t: {"dir": p["direction"], "prob": p["probability"]}
                           for t, p in predictions.items()},
            "period_return": round(period_return * 100, 2),
            "spy_return": round(spy_ret * 100, 2),
            "portfolio_value": round(portfolio_value, 2),
            "spy_value": round(spy_value, 2),
        })

    return _print_summary("LLM", portfolio_value, spy_value, history, rebalance_dates, rebalance_days)


def run_lstm_backtest(start_date, end_date, tickers=None,
                      rebalance_days=None, threshold=None):
    """
    Run portfolio backtest using LSTM predictions vs SPY benchmark.
    Requires pre-trained LSTM models (run main.py train-lstm first).
    """
    tickers = tickers or config.TICKERS
    rebalance_days = rebalance_days or config.REBALANCE_DAYS
    threshold = threshold or config.MIN_PROBABILITY

    print(f"\n{'='*65}")
    print(f"LSTM BACKTEST")
    print(f"{'='*65}")
    print(f"Period:     {start_date} to {end_date}")
    print(f"{'='*65}\n")

    from lstm_model import LSTMTrainer
    trainer = LSTMTrainer()

    print("Fetching price data...")
    all_tickers = list(set(tickers + ["SPY"]))
    price_data = fetch_all_price_data(all_tickers, start_date, end_date)

    spy_data = price_data["SPY"]
    rebalance_dates, valid_days = _get_rebalance_dates(
        spy_data.index, start_date, end_date, rebalance_days
    )
    print(f"Rebalance dates: {len(rebalance_dates)}\n")

    portfolio_value = config.INITIAL_CAPITAL
    spy_value = config.INITIAL_CAPITAL
    history = []

    for r_idx, rebal_date in enumerate(rebalance_dates):
        if r_idx + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[r_idx + 1]
        else:
            future = valid_days[valid_days > rebal_date]
            next_rebal = future[-1] if len(future) > 0 else rebal_date

        date_str = rebal_date.strftime("%Y-%m-%d")
        print(f"[{r_idx+1:02d}/{len(rebalance_dates)}] {date_str}")

        # LSTM predictions
        predictions = {}
        for ticker in tickers:
            if ticker not in price_data:
                continue
            prob_up = trainer.predict(ticker, rebal_date, price_data[ticker])
            direction = "up" if prob_up >= 0.5 else "down"
            predictions[ticker] = {
                "direction": direction,
                "probability": round(prob_up * 100, 1),
            }
            print(f"    {ticker}: {direction} ({prob_up*100:.1f}%)")

        # Convert to weights
        weights = predictions_to_weights(predictions, threshold=threshold)
        cash_pct = weights.get("CASH", 0.0)
        print(f"  Cash: {cash_pct:.0%}")

        period_return = _compute_period_return(weights, price_data, rebal_date, next_rebal)
        portfolio_value *= (1 + period_return)

        spy_entry = spy_data[spy_data.index >= rebal_date]
        spy_exit = spy_data[spy_data.index >= next_rebal]
        if len(spy_entry) > 0 and len(spy_exit) > 0:
            spy_ret = (spy_exit.iloc[0]["Close"] - spy_entry.iloc[0]["Close"]) / spy_entry.iloc[0]["Close"]
        else:
            spy_ret = 0.0
        spy_value *= (1 + spy_ret)

        print(f"  Return: {period_return:+.2%} | SPY: {spy_ret:+.2%} | Portfolio: ${portfolio_value:,.0f}")

        history.append({
            "date": date_str,
            "weights": {t: round(w, 4) for t, w in weights.items()},
            "period_return": round(period_return * 100, 2),
            "spy_return": round(spy_ret * 100, 2),
            "portfolio_value": round(portfolio_value, 2),
            "spy_value": round(spy_value, 2),
        })

    return _print_summary("LSTM", portfolio_value, spy_value, history, rebalance_dates, rebalance_days)


def _print_summary(method, portfolio_value, spy_value, history, rebalance_dates, rebalance_days):
    """Print and return backtest summary."""
    initial = config.INITIAL_CAPITAL
    total_port = (portfolio_value - initial) / initial * 100
    total_spy = (spy_value - initial) / initial * 100
    alpha = total_port - total_spy

    n_months = len(rebalance_dates)
    years = n_months * rebalance_days / 252

    port_cagr = ((portfolio_value / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
    spy_cagr = ((spy_value / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    print(f"\n{'='*65}")
    print(f"RESULTS - {method} (${initial:,.0f} initial)")
    print(f"{'='*65}")
    print(f"{'':30} {'Portfolio':>12}  {'SPY':>10}")
    print(f"{'Final Value':30} ${portfolio_value:>11,.0f}  ${spy_value:>9,.0f}")
    print(f"{'Total Return':30} {total_port:>+11.1f}%  {total_spy:>+9.1f}%")
    print(f"{'CAGR':30} {port_cagr:>+11.1f}%  {spy_cagr:>+9.1f}%")
    print(f"{'Alpha (vs SPY)':30} {alpha:>+11.1f}%")
    print(f"{'='*65}")

    if alpha > 0:
        print(f"\n[WIN] Portfolio BEAT SPY by {alpha:.1f}%")
    else:
        print(f"\n[LOSS] Portfolio UNDERPERFORMED SPY by {abs(alpha):.1f}%")

    result = {
        "method": method,
        "portfolio_value": round(portfolio_value, 2),
        "spy_value": round(spy_value, 2),
        "total_return": round(total_port, 2),
        "spy_return": round(total_spy, 2),
        "alpha": round(alpha, 2),
        "cagr": round(port_cagr, 2),
        "spy_cagr": round(spy_cagr, 2),
        "history": history,
    }

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.RESULTS_DIR, f"backtest_{method.lower()}_{ts}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {path}")

    return result
