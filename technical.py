"""
Technical Analysis - Compute EMA, RSI, volume ratio and related indicators.
"""
import numpy as np
import pandas as pd


def compute_ema(series, span):
    """Compute Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series, period=14):
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_indicators(price_df, as_of_date=None):
    """
    Compute all technical indicators from price data.

    Args:
        price_df: DataFrame with OHLCV columns, DatetimeIndex
        as_of_date: Only use data up to this date (prevents look-ahead bias)

    Returns:
        dict with all indicator values, or None if insufficient data
    """
    if as_of_date is not None:
        df = price_df[price_df.index <= as_of_date].copy()
    else:
        df = price_df.copy()

    if len(df) < 150:
        return None

    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None

    # EMAs
    ema25 = compute_ema(close, 25)
    ema50 = compute_ema(close, 50)
    ema100 = compute_ema(close, 100)

    # RSI
    rsi = compute_rsi(close, 14)

    # Latest values
    current_price = float(close.iloc[-1])
    ema25_val = float(ema25.iloc[-1])
    ema50_val = float(ema50.iloc[-1])
    ema100_val = float(ema100.iloc[-1])
    rsi_val = float(rsi.iloc[-1])

    # EMA trend signals
    ema100_uptrend = int(ema100.iloc[-1] > ema100.iloc[-20])
    ema50_above_100 = int(ema50_val > ema100_val)
    ema25_above_100 = int(ema25_val > ema100_val)

    # EMA alignment score (-4 to +4)
    ema25_above_50 = int(ema25_val > ema50_val)
    price_above_25 = int(current_price > ema25_val)
    alignment_score = (
        price_above_25 + ema25_above_50 + ema50_above_100 + ema100_uptrend
        - (1 - price_above_25) - (1 - ema25_above_50)
        - (1 - ema50_above_100) - (1 - ema100_uptrend)
    )

    # Price vs EMA deviations (%)
    price_vs_ema25 = round((current_price - ema25_val) / ema25_val * 100, 2)
    price_vs_ema50 = round((current_price - ema50_val) / ema50_val * 100, 2)
    price_vs_ema100 = round((current_price - ema100_val) / ema100_val * 100, 2)

    # Volume ratio
    vol_ratio = 1.0
    if volume is not None and len(volume) >= 20:
        avg_vol = volume.iloc[-20:].mean()
        if avg_vol > 0:
            vol_ratio = round(float(volume.iloc[-1] / avg_vol), 4)

    # RSI signal
    if rsi_val >= 70:
        rsi_signal = "overbought"
    elif rsi_val <= 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    return {
        "current_price": round(current_price, 2),
        "ema25": round(ema25_val, 2),
        "ema50": round(ema50_val, 2),
        "ema100": round(ema100_val, 2),
        "ema100_uptrend": ema100_uptrend,
        "ema50_above_ema100": ema50_above_100,
        "ema25_above_ema100": ema25_above_100,
        "ema_alignment_score": alignment_score,
        "rsi": round(rsi_val, 2),
        "rsi_signal": rsi_signal,
        "price_vs_ema25_pct": price_vs_ema25,
        "price_vs_ema50_pct": price_vs_ema50,
        "price_vs_ema100_pct": price_vs_ema100,
        "volume_ratio": vol_ratio,
    }


def format_indicators_for_prompt(indicators):
    """Format technical indicators into a readable text block for LLM prompt."""
    if indicators is None:
        return "Insufficient price history to compute technical indicators."

    lines = [
        f"Price: ${indicators['current_price']}",
        f"EMA25: ${indicators['ema25']} (price vs EMA25: {indicators['price_vs_ema25_pct']:+.2f}%)",
        f"EMA50: ${indicators['ema50']} (price vs EMA50: {indicators['price_vs_ema50_pct']:+.2f}%)",
        f"EMA100: ${indicators['ema100']} (price vs EMA100: {indicators['price_vs_ema100_pct']:+.2f}%)",
        f"EMA Alignment Score: {indicators['ema_alignment_score']}/4 "
        f"(EMA100 uptrend: {'Yes' if indicators['ema100_uptrend'] else 'No'}, "
        f"EMA50>EMA100: {'Yes' if indicators['ema50_above_ema100'] else 'No'}, "
        f"EMA25>EMA100: {'Yes' if indicators['ema25_above_ema100'] else 'No'})",
        f"RSI(14): {indicators['rsi']} ({indicators['rsi_signal']})",
        f"Volume Ratio (vs 20-day avg): {indicators['volume_ratio']:.2f}x",
    ]
    return "\n".join(lines)
