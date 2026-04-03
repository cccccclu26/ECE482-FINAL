"""
Technical Analysis - Compute all mainstream technical indicators.

Indicator categories:
  Trend:      EMA(25/50/100), SMA(20/50/200), MACD(12,26,9), ADX(14)
  Momentum:   RSI(14), Stochastic(14,3,3), Williams %R(14), CCI(20)
  Volatility: Bollinger Bands(20,2), ATR(14)
  Volume:     Volume Ratio, OBV
"""
import numpy as np
import pandas as pd


# ============================================================
# Basic Indicator Computations
# ============================================================

def compute_ema(series, span):
    """Compute Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_sma(series, period):
    """Compute Simple Moving Average."""
    return series.rolling(window=period).mean()


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


def compute_macd(close, fast=12, slow=26, signal=9):
    """Compute MACD line, signal line, and histogram."""
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(close, period=20, std_dev=2):
    """Compute Bollinger Bands (upper, middle, lower, width)."""
    middle = compute_sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle * 100  # as percentage
    return upper, middle, lower, width


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """Compute Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


def compute_atr(high, low, close, period=14):
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_obv(close, volume):
    """Compute On-Balance Volume."""
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    return obv


def compute_williams_r(high, low, close, period=14):
    """Compute Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr


def compute_cci(high, low, close, period=20):
    """Compute Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci


def compute_adx(high, low, close, period=14):
    """Compute Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = compute_atr(high, low, close, period)

    plus_di = 100 * compute_ema(plus_dm, period) / atr
    minus_di = 100 * compute_ema(minus_dm, period) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = compute_ema(dx, period)
    return adx, plus_di, minus_di


# ============================================================
# Main Indicator Computation
# ============================================================

def compute_indicators(price_df, as_of_date=None):
    """
    Compute ALL technical indicators from price data.

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

    if len(df) < 200:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)

    # --- Trend Indicators ---
    ema25 = compute_ema(close, 25)
    ema50 = compute_ema(close, 50)
    ema100 = compute_ema(close, 100)
    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)
    macd_line, macd_signal, macd_hist = compute_macd(close)
    adx, plus_di, minus_di = compute_adx(high, low, close)

    # --- Momentum Indicators ---
    rsi = compute_rsi(close, 14)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    williams_r = compute_williams_r(high, low, close)
    cci = compute_cci(high, low, close)

    # --- Volatility Indicators ---
    bb_upper, bb_middle, bb_lower, bb_width = compute_bollinger_bands(close)
    atr = compute_atr(high, low, close)

    # --- Volume Indicators ---
    obv = compute_obv(close, volume)
    vol_avg_20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg_20

    # --- Latest Values ---
    current_price = float(close.iloc[-1])

    # EMA values
    ema25_val = float(ema25.iloc[-1])
    ema50_val = float(ema50.iloc[-1])
    ema100_val = float(ema100.iloc[-1])

    # EMA alignment score (-4 to +4)
    ema100_uptrend = int(ema100.iloc[-1] > ema100.iloc[-20])
    ema50_above_100 = int(ema50_val > ema100_val)
    ema25_above_50 = int(ema25_val > ema50_val)
    price_above_25 = int(current_price > ema25_val)
    alignment_score = (
        price_above_25 + ema25_above_50 + ema50_above_100 + ema100_uptrend
        - (1 - price_above_25) - (1 - ema25_above_50)
        - (1 - ema50_above_100) - (1 - ema100_uptrend)
    )

    # Price deviations
    price_vs_ema25 = round((current_price - ema25_val) / ema25_val * 100, 2)
    price_vs_ema50 = round((current_price - ema50_val) / ema50_val * 100, 2)
    price_vs_ema100 = round((current_price - ema100_val) / ema100_val * 100, 2)

    # SMA values
    sma20_val = float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else None
    sma50_val = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None
    sma200_val = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None

    # RSI signal
    rsi_val = float(rsi.iloc[-1])
    if rsi_val >= 70:
        rsi_signal = "overbought"
    elif rsi_val <= 30:
        rsi_signal = "oversold"
    else:
        rsi_signal = "neutral"

    # Volume ratio
    vr = float(vol_ratio.iloc[-1]) if not np.isnan(vol_ratio.iloc[-1]) else 1.0

    # Stochastic signal
    stoch_k_val = float(stoch_k.iloc[-1]) if not np.isnan(stoch_k.iloc[-1]) else 50.0
    stoch_d_val = float(stoch_d.iloc[-1]) if not np.isnan(stoch_d.iloc[-1]) else 50.0
    if stoch_k_val > 80:
        stoch_signal = "overbought"
    elif stoch_k_val < 20:
        stoch_signal = "oversold"
    else:
        stoch_signal = "neutral"

    # MACD signal
    macd_val = float(macd_line.iloc[-1]) if not np.isnan(macd_line.iloc[-1]) else 0.0
    macd_sig_val = float(macd_signal.iloc[-1]) if not np.isnan(macd_signal.iloc[-1]) else 0.0
    macd_hist_val = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0.0
    if macd_val > macd_sig_val:
        macd_signal_str = "bullish"
    else:
        macd_signal_str = "bearish"

    # Bollinger Band position
    bb_upper_val = float(bb_upper.iloc[-1]) if not np.isnan(bb_upper.iloc[-1]) else current_price
    bb_lower_val = float(bb_lower.iloc[-1]) if not np.isnan(bb_lower.iloc[-1]) else current_price
    bb_middle_val = float(bb_middle.iloc[-1]) if not np.isnan(bb_middle.iloc[-1]) else current_price
    bb_width_val = float(bb_width.iloc[-1]) if not np.isnan(bb_width.iloc[-1]) else 0.0
    bb_range = bb_upper_val - bb_lower_val
    if bb_range > 0:
        bb_position = (current_price - bb_lower_val) / bb_range
    else:
        bb_position = 0.5

    return {
        # Trend
        "current_price": round(current_price, 2),
        "ema25": round(ema25_val, 2),
        "ema50": round(ema50_val, 2),
        "ema100": round(ema100_val, 2),
        "price_vs_ema25_pct": price_vs_ema25,
        "price_vs_ema50_pct": price_vs_ema50,
        "price_vs_ema100_pct": price_vs_ema100,
        "ema_alignment_score": alignment_score,
        "ema100_uptrend": ema100_uptrend,
        "ema50_above_ema100": ema50_above_100,
        "ema25_above_ema100": int(ema25_val > ema100_val),
        "sma20": round(sma20_val, 2) if sma20_val else None,
        "sma50": round(sma50_val, 2) if sma50_val else None,
        "sma200": round(sma200_val, 2) if sma200_val else None,
        "golden_cross": int(sma50_val > sma200_val) if sma50_val and sma200_val else None,
        "macd": round(macd_val, 4),
        "macd_signal": round(macd_sig_val, 4),
        "macd_histogram": round(macd_hist_val, 4),
        "macd_trend": macd_signal_str,
        "adx": round(float(adx.iloc[-1]), 2) if not np.isnan(adx.iloc[-1]) else None,
        "plus_di": round(float(plus_di.iloc[-1]), 2) if not np.isnan(plus_di.iloc[-1]) else None,
        "minus_di": round(float(minus_di.iloc[-1]), 2) if not np.isnan(minus_di.iloc[-1]) else None,

        # Momentum
        "rsi": round(rsi_val, 2),
        "rsi_signal": rsi_signal,
        "stochastic_k": round(stoch_k_val, 2),
        "stochastic_d": round(stoch_d_val, 2),
        "stochastic_signal": stoch_signal,
        "williams_r": round(float(williams_r.iloc[-1]), 2) if not np.isnan(williams_r.iloc[-1]) else None,
        "cci": round(float(cci.iloc[-1]), 2) if not np.isnan(cci.iloc[-1]) else None,

        # Volatility
        "bb_upper": round(bb_upper_val, 2),
        "bb_lower": round(bb_lower_val, 2),
        "bb_width_pct": round(bb_width_val, 2),
        "bb_position": round(bb_position, 3),
        "atr": round(float(atr.iloc[-1]), 2) if not np.isnan(atr.iloc[-1]) else None,
        "atr_pct": round(float(atr.iloc[-1]) / current_price * 100, 2) if not np.isnan(atr.iloc[-1]) else None,

        # Volume
        "volume_ratio": round(vr, 4),
        "obv_trend": "up" if float(obv.iloc[-1]) > float(obv.iloc[-5]) else "down",
    }


def format_indicators_for_prompt(indicators):
    """Format all technical indicators into a categorized text block for LLM prompt."""
    if indicators is None:
        return "Insufficient price history to compute technical indicators."

    lines = []

    # Trend Indicators
    lines.append("=== TREND INDICATORS ===")
    lines.append(f"Price: ${indicators['current_price']}")
    lines.append(f"EMA25: ${indicators['ema25']} ({indicators['price_vs_ema25_pct']:+.2f}% from price)")
    lines.append(f"EMA50: ${indicators['ema50']} ({indicators['price_vs_ema50_pct']:+.2f}% from price)")
    lines.append(f"EMA100: ${indicators['ema100']} ({indicators['price_vs_ema100_pct']:+.2f}% from price)")
    lines.append(f"EMA Alignment: {indicators['ema_alignment_score']}/4")
    if indicators.get("sma20"):
        lines.append(f"SMA20: ${indicators['sma20']}  SMA50: ${indicators['sma50']}  SMA200: ${indicators['sma200']}")
    if indicators.get("golden_cross") is not None:
        lines.append(f"Golden Cross (SMA50>SMA200): {'Yes' if indicators['golden_cross'] else 'No'}")
    lines.append(f"MACD: {indicators['macd']:.4f} | Signal: {indicators['macd_signal']:.4f} | Hist: {indicators['macd_histogram']:.4f} ({indicators['macd_trend']})")
    if indicators.get("adx") is not None:
        lines.append(f"ADX: {indicators['adx']} (+DI: {indicators['plus_di']}, -DI: {indicators['minus_di']})")

    # Momentum Indicators
    lines.append("")
    lines.append("=== MOMENTUM INDICATORS ===")
    lines.append(f"RSI(14): {indicators['rsi']} ({indicators['rsi_signal']})")
    lines.append(f"Stochastic: %K={indicators['stochastic_k']}, %D={indicators['stochastic_d']} ({indicators['stochastic_signal']})")
    if indicators.get("williams_r") is not None:
        lines.append(f"Williams %R(14): {indicators['williams_r']}")
    if indicators.get("cci") is not None:
        lines.append(f"CCI(20): {indicators['cci']}")

    # Volatility Indicators
    lines.append("")
    lines.append("=== VOLATILITY INDICATORS ===")
    lines.append(f"Bollinger Bands: Upper=${indicators['bb_upper']}, Lower=${indicators['bb_lower']}")
    lines.append(f"BB Width: {indicators['bb_width_pct']:.2f}% | Position: {indicators['bb_position']:.3f} (0=lower, 1=upper)")
    if indicators.get("atr") is not None:
        lines.append(f"ATR(14): ${indicators['atr']} ({indicators['atr_pct']:.2f}% of price)")

    # Volume Indicators
    lines.append("")
    lines.append("=== VOLUME INDICATORS ===")
    lines.append(f"Volume Ratio (vs 20-day avg): {indicators['volume_ratio']:.2f}x")
    lines.append(f"OBV Trend (5-day): {indicators['obv_trend']}")

    return "\n".join(lines)
