"""
ECE482-FINAL: LLM-Driven Stock Prediction System

Entry point with subcommands:
  predict     - Run LLM predictions for current date
  backtest    - Backtest LLM or LSTM portfolio vs SPY
  train-lstm  - Train LSTM models for all tickers
  compare     - Compare LLM vs LSTM backtest results
  accuracy    - Show prediction accuracy from eval logs
  info        - Show system info and loaded models

Usage:
  python main.py predict --ticker AAPL
  python main.py predict --all
  python main.py backtest --method llm --start 2023-01-01 --end 2025-12-01
  python main.py backtest --method lstm --start 2023-01-01 --end 2025-12-01
  python main.py train-lstm --start 2021-01-01 --end 2024-12-31
  python main.py compare --start 2025-01-01 --end 2026-03-01
  python main.py accuracy
"""
import argparse
import json
import sys
from datetime import datetime

import config
from data_fetcher import fetch_all_price_data


def cmd_predict(args):
    """Run LLM prediction for one or all tickers."""
    from llm_predictor import LLMPredictor

    predictor = LLMPredictor(prompt_version=args.prompt)
    tickers = config.TICKERS if args.all else [args.ticker.upper()]

    print(f"\nFetching price data...")
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    price_data = fetch_all_price_data(tickers, "2024-01-01", date_str)

    print(f"\nPredicting for {date_str} (prompt: {args.prompt}):\n")

    for ticker in tickers:
        if ticker not in price_data:
            print(f"{ticker}: No data available")
            continue

        pred = predictor.predict(ticker, date_str, price_data[ticker])

        print(f"\n{'='*55}")
        print(f"  {ticker}  |  {pred['direction'].upper()}  |  P={pred['probability']}%  |  Target: {pred['target_return']:+.1f}%")
        print(f"{'='*55}")
        print(f"  Models: {', '.join(pred.get('models_used', []))}")
        print(f"  Agreement: {'Yes' if pred.get('agreement') else 'No'}")
        if pred.get("key_factors"):
            print(f"  Key factors: {', '.join(pred['key_factors'][:3])}")
        if pred.get("risk_factors"):
            print(f"  Risks: {', '.join(pred['risk_factors'][:2])}")
        print(f"  Reasoning: {pred.get('reasoning', 'N/A')[:120]}")


def cmd_backtest(args):
    """Run backtest with specified method."""
    from backtest import run_llm_backtest, run_lstm_backtest

    tickers = args.tickers if args.tickers else config.TICKERS

    if args.method == "llm":
        run_llm_backtest(
            start_date=args.start,
            end_date=args.end,
            tickers=tickers,
            prompt_version=args.prompt,
            rebalance_days=args.rebalance_days,
            threshold=args.threshold,
        )
    elif args.method == "lstm":
        run_lstm_backtest(
            start_date=args.start,
            end_date=args.end,
            tickers=tickers,
            rebalance_days=args.rebalance_days,
            threshold=args.threshold,
        )
    else:
        print(f"Unknown method: {args.method}. Use 'llm' or 'lstm'.")


def cmd_train_lstm(args):
    """Train LSTM models for all tickers."""
    from lstm_model import LSTMTrainer

    tickers = args.tickers if args.tickers else config.TICKERS

    print(f"\n{'='*55}")
    print(f"LSTM Training")
    print(f"{'='*55}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"{'='*55}\n")

    print("Fetching price data...")
    price_data = fetch_all_price_data(tickers, args.start, args.end)

    trainer = LSTMTrainer()
    results = trainer.train_all(tickers, price_data, args.start, args.end,
                                 epochs=args.epochs)

    print(f"\n{'='*55}")
    print(f"Training Complete")
    print(f"{'='*55}")
    for ticker, meta in results.items():
        print(f"  {ticker}: val_acc={meta['val_accuracy']}%  samples={meta['n_samples']}")


def cmd_compare(args):
    """Run both LLM and LSTM backtests and compare."""
    from backtest import run_llm_backtest, run_lstm_backtest

    print("=" * 65)
    print("COMPARISON: LLM vs LSTM vs SPY")
    print("=" * 65)

    llm_result = run_llm_backtest(
        start_date=args.start,
        end_date=args.end,
        prompt_version=args.prompt,
    )

    lstm_result = run_lstm_backtest(
        start_date=args.start,
        end_date=args.end,
    )

    print(f"\n{'='*65}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"{'':15} {'LLM':>12} {'LSTM':>12} {'SPY':>12}")
    print(f"{'Total Return':15} {llm_result['total_return']:>+11.1f}% {lstm_result['total_return']:>+11.1f}% {llm_result['spy_return']:>+11.1f}%")
    print(f"{'Alpha':15} {llm_result['alpha']:>+11.1f}% {lstm_result['alpha']:>+11.1f}%")
    print(f"{'CAGR':15} {llm_result['cagr']:>+11.1f}% {lstm_result['cagr']:>+11.1f}% {llm_result['spy_cagr']:>+11.1f}%")
    print(f"{'='*65}")

    if llm_result['alpha'] > lstm_result['alpha']:
        print("\n[RESULT] LLM outperformed LSTM")
    else:
        print("\n[RESULT] LSTM outperformed LLM")


def cmd_accuracy(args):
    """Show prediction accuracy from memory."""
    from memory import compute_accuracy

    ticker = args.ticker.upper() if args.ticker else None
    stats = compute_accuracy(ticker)

    print(f"\n{'='*45}")
    print(f"Prediction Accuracy")
    print(f"{'='*45}")

    if stats["per_ticker"]:
        for t, s in sorted(stats["per_ticker"].items()):
            print(f"  {t:6s}: {s['correct']}/{s['total']}  ({s['accuracy']}%)")
        print(f"  {'------':6s}  --------")
        o = stats["overall"]
        print(f"  {'TOTAL':6s}: {o['correct']}/{o['total']}  ({o['accuracy']}%)")
    else:
        print("  No evaluation data yet. Run a backtest first.")


def cmd_info(args):
    """Show system info."""
    import os

    print(f"\n{'='*55}")
    print(f"ECE482-FINAL System Info")
    print(f"{'='*55}")
    print(f"Tickers:     {', '.join(config.TICKERS)}")
    print(f"LLM Models:  {', '.join(config.LLM_MODELS)}")
    print(f"Horizon:     {config.PREDICTION_HORIZON_DAYS} trading days")
    print(f"Rebalance:   {config.REBALANCE_DAYS} trading days")

    # Check LSTM models
    print(f"\nLSTM Models:")
    for ticker in config.TICKERS:
        meta_path = os.path.join(config.MODELS_DIR, f"{ticker}_lstm_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  {ticker}: val_acc={meta['val_accuracy']}%  "
                  f"train={meta['train_start']}~{meta['train_end']}  "
                  f"samples={meta['n_samples']}")
        else:
            print(f"  {ticker}: not trained")

    # Check prompt versions
    print(f"\nPrompt Templates:")
    for fname in sorted(os.listdir(config.PROMPTS_DIR)):
        if fname.endswith(".txt"):
            print(f"  {fname}")

    # Check API keys
    print(f"\nAPI Keys:")
    print(f"  Polygon.io: {'set' if config.POLYGON_API_KEY else 'MISSING'}")
    print(f"  WaveSpeed:  {'set' if config.WAVESPEED_API_KEY else 'MISSING'}")


def main():
    parser = argparse.ArgumentParser(
        description="ECE482-FINAL: LLM-Driven Stock Prediction System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    p_predict = subparsers.add_parser("predict", help="Run LLM prediction")
    p_predict.add_argument("-t", "--ticker", type=str, help="Single ticker")
    p_predict.add_argument("-a", "--all", action="store_true", help="All tickers")
    p_predict.add_argument("--date", type=str, default=None, help="Prediction date (default: today)")
    p_predict.add_argument("--prompt", type=str, default="v1", help="Prompt version (default: v1)")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("--method", type=str, default="llm", choices=["llm", "lstm"])
    p_bt.add_argument("--start", type=str, default="2023-01-01")
    p_bt.add_argument("--end", type=str, default="2025-12-01")
    p_bt.add_argument("--prompt", type=str, default="v1", help="Prompt version (LLM only)")
    p_bt.add_argument("--rebalance-days", type=int, default=None)
    p_bt.add_argument("--threshold", type=float, default=None)
    p_bt.add_argument("--tickers", nargs="+", default=None)

    # train-lstm
    p_train = subparsers.add_parser("train-lstm", help="Train LSTM models")
    p_train.add_argument("--start", type=str, default="2021-01-01")
    p_train.add_argument("--end", type=str, default="2024-12-31")
    p_train.add_argument("--tickers", nargs="+", default=None)
    p_train.add_argument("--epochs", type=int, default=100)

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare LLM vs LSTM")
    p_cmp.add_argument("--start", type=str, default="2025-01-01")
    p_cmp.add_argument("--end", type=str, default="2026-03-01")
    p_cmp.add_argument("--prompt", type=str, default="v1")

    # accuracy
    p_acc = subparsers.add_parser("accuracy", help="Show prediction accuracy")
    p_acc.add_argument("-t", "--ticker", type=str, default=None)

    # info
    subparsers.add_parser("info", help="Show system info")

    args = parser.parse_args()

    if args.command == "predict":
        if not args.ticker and not args.all:
            print("Specify --ticker AAPL or --all")
            return
        cmd_predict(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "train-lstm":
        cmd_train_lstm(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "accuracy":
        cmd_accuracy(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py predict --all")
        print("  python main.py backtest --method llm --start 2025-01-01 --end 2026-03-01")
        print("  python main.py train-lstm --start 2021-01-01 --end 2024-12-31")
        print("  python main.py compare --start 2025-01-01 --end 2026-03-01")


if __name__ == "__main__":
    main()
