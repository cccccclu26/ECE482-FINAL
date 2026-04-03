"""
Data Fetcher - Price data (Polygon.io + yfinance) and news (Polygon.io).
"""
import ssl
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

import config

ssl._create_default_https_context = ssl._create_unverified_context


def _normalize_yf_index(df):
    """Normalize yfinance DataFrame index to tz-naive DatetimeIndex."""
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def fetch_polygon_ohlcv(ticker, start_str, end_str):
    """Fetch daily OHLCV from Polygon.io. Returns DataFrame with OHLCV columns."""
    url = (
        f"{config.POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day"
        f"/{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000"
        f"&apiKey={config.POLYGON_API_KEY}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"No Polygon data for {ticker}")

    df = pd.DataFrame(results)
    df["Date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("Date")
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = df.index.tz_localize(None)
    return df


def fetch_price_data(ticker, start_date, end_date, warmup_days=400):
    """
    Fetch price data for a single ticker with EMA warmup buffer.
    Uses Polygon.io first, falls back to yfinance for older data.

    Args:
        ticker: Stock ticker symbol
        start_date: "YYYY-MM-DD" backtest start
        end_date: "YYYY-MM-DD" backtest end
        warmup_days: Extra calendar days before start for EMA warmup

    Returns:
        DataFrame with OHLCV columns
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=warmup_days)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=10)
    s = start_dt.strftime("%Y-%m-%d")
    e = end_dt.strftime("%Y-%m-%d")

    try:
        df = fetch_polygon_ohlcv(ticker, s, e)
        # If Polygon data starts too late, prepend yfinance data
        if len(df) > 0 and df.index[0] > pd.Timestamp(s) + timedelta(days=30):
            yf_end = df.index[0].strftime("%Y-%m-%d")
            yf_df = yf.Ticker(ticker).history(start=s, end=yf_end)
            yf_df = _normalize_yf_index(yf_df)
            yf_df = yf_df[["Open", "High", "Low", "Close", "Volume"]]
            df = pd.concat([yf_df, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        return df
    except Exception:
        df = yf.Ticker(ticker).history(start=s, end=e)
        df = _normalize_yf_index(df)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df


def fetch_all_price_data(tickers, start_date, end_date):
    """Fetch price data for multiple tickers. Returns dict of DataFrames."""
    data = {}
    for ticker in tickers:
        try:
            df = fetch_price_data(ticker, start_date, end_date)
            print(f"  {ticker}: {len(df)} days", flush=True)
            data[ticker] = df
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})", flush=True)
    return data


def fetch_news(ticker, date, limit=None):
    """
    Fetch news articles for a ticker around a given date from Polygon.io.

    Args:
        ticker: Stock ticker symbol
        date: datetime or string "YYYY-MM-DD"
        limit: Max articles to return

    Returns:
        List of dicts with title, description, published_utc, source
    """
    limit = limit or config.DEFAULT_NEWS_LIMIT
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    end_str = date.strftime("%Y-%m-%d")
    start_str = (date - timedelta(days=config.NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    url = (
        f"{config.POLYGON_BASE_URL}/v2/reference/news"
        f"?ticker={ticker}"
        f"&published_utc.gte={start_str}"
        f"&published_utc.lte={end_str}"
        f"&order=desc&limit={limit}"
        f"&apiKey={config.POLYGON_API_KEY}"
    )

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        articles = r.json().get("results", [])
    except Exception as e:
        print(f"  News fetch failed for {ticker}: {e}")
        return []

    news_list = []
    for a in articles:
        news_list.append({
            "ticker": ticker,
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "published_utc": a.get("published_utc", ""),
            "source": a.get("publisher", {}).get("name", "Unknown"),
        })

    return news_list


def format_news_for_prompt(news_list, max_articles=5):
    """Format news articles into a text block for LLM prompt injection."""
    if not news_list:
        return "No recent news available."

    lines = []
    for i, n in enumerate(news_list[:max_articles], 1):
        title = n.get("title", "N/A")
        desc = n.get("description", "")
        source = n.get("source", "Unknown")
        date = n.get("published_utc", "")[:10]
        lines.append(f"{i}. [{date}] ({source}) {title}")
        if desc:
            lines.append(f"   {desc[:200]}")
    return "\n".join(lines)
