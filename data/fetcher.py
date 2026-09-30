"""
data/fetcher.py
===============
Download, cache, and preprocess OHLCV market data from multiple sources.

Sources supported:
    - yfinance (stocks, ETFs, crypto)
    - Alpaca (stocks, paper/live)
    - CCXT (crypto exchanges)
"""
from __future__ import annotations

import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import inspect

import numpy as np
import pandas as pd
from loguru import logger

# ─── Constants ────────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/cache")
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
_PRICE_NAMES = {"open", "high", "low", "close", "adj close", "volume"}


def _flatten_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame whose columns are price names, not a ticker level.

    Current ``yfinance.download`` defaults to a MultiIndex of
    ``(Price, Ticker)`` even for one symbol. Older frames are already flat.
    A ticker-first orientation is accepted too.
    """
    out = df.copy()
    columns = out.columns
    if isinstance(columns, pd.MultiIndex):
        chosen: int | None = None
        for level in range(columns.nlevels):
            names = {str(value).strip().lower() for value in columns.get_level_values(level)}
            if names & _PRICE_NAMES:
                chosen = level
                break
        if chosen is None:
            raise ValueError(
                "Could not find Open/High/Low/Close/Volume in MultiIndex columns."
            )
        out.columns = columns.get_level_values(chosen)
    out.columns = [str(column).strip().title() for column in out.columns]
    return out


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_path(ticker: str, start: str, end: str, interval: str) -> Path:
    key = f"{ticker}_{start}_{end}_{interval}"
    hashed = hashlib.md5(key.encode()).hexdigest()[:8]
    return CACHE_DIR / f"{ticker}_{interval}_{hashed}.parquet"


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        logger.debug(f"Cache hit: {path}")
        return pd.read_parquet(path)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.debug(f"Saved cache: {path}")


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich raw OHLCV data.

    Steps:
        1. Flatten a yfinance-style MultiIndex down to price names.
        2. Forward-fill small gaps (up to 5 bars).
        3. Drop rows where all OHLCV values are NaN.
        4. Add log-returns and typical price.
        5. Ensure DatetimeIndex with UTC-aware timestamps.
    """
    df = _flatten_ohlcv_columns(df)

    # Keep only standard OHLCV cols that exist
    cols = [c for c in OHLCV_COLS if c in df.columns]
    df = df[cols]

    # Drop fully-NaN rows then forward-fill up to 5 periods
    df.dropna(how="all", inplace=True)
    df.ffill(limit=5, inplace=True)
    df.dropna(inplace=True)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Derived columns
    df["Returns"] = df["Close"].pct_change()
    df["LogReturns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["TypicalPrice"] = (df["High"] + df["Low"] + df["Close"]) / 3

    return df.dropna()


# ─── yfinance fetcher ─────────────────────────────────────────────────────────

def fetch_yfinance(
    ticker: str,
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
    adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance with local parquet caching.

    Args:
        ticker:    Stock/crypto symbol (e.g. 'AAPL', 'BTC-USD').
        start:     ISO date string for period start.
        end:       ISO date string for period end (defaults to today).
        interval:  Bar interval: '1m','5m','1h','1d','1wk','1mo'.
        use_cache: Load from disk if available.
        adjust:    Auto-adjust for splits and dividends.

    Returns:
        Preprocessed DataFrame with OHLCV + derived columns.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache = _cache_path(ticker, start, end, interval)

    if use_cache:
        cached = _load_cache(cache)
        if cached is not None:
            return cached

    logger.info(f"Fetching {ticker} [{interval}] from {start} to {end}")
    import yfinance as yf

    download_kwargs = {
        "start": start,
        "end": end,
        "interval": interval,
        "auto_adjust": adjust,
        "progress": False,
    }
    # Default True since yfinance 0.2.50, which nests the ticker on the columns.
    if "multi_level_index" in inspect.signature(yf.download).parameters:
        download_kwargs["multi_level_index"] = False
    raw = yf.download(ticker, **download_kwargs)

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = preprocess(raw)
    if use_cache:
        _save_cache(df, cache)
    return df


def fetch_multiple(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple tickers; returns dict keyed by ticker symbol."""
    return {t: fetch_yfinance(t, start, end, interval, use_cache)
            for t in tickers}


# ─── Alpaca fetcher ───────────────────────────────────────────────────────────

def fetch_alpaca(
    ticker: str,
    start: str = "2020-01-01",
    end: str | None = None,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    """
    Fetch historical bars from Alpaca Markets API.

    Requires `alpaca-py`. Credentials come from the environment variables
    ALPACA_API_KEY and ALPACA_SECRET_KEY, not from function arguments.

    Args:
        ticker:    Equity symbol (e.g. 'AAPL').
        start:     ISO date string.
        end:       ISO date string (defaults to today).
        timeframe: Alpaca TimeFrame string: '1Min','5Min','1Hour','1Day'.

    Returns:
        Preprocessed OHLCV DataFrame.
    """
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment."
        )

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise ImportError("Install alpaca-py: pip install alpaca-py")

    client = StockHistoricalDataClient(api_key, secret_key)

    # Map string to TimeFrame
    tf_map = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

    end = end or datetime.today().strftime("%Y-%m-%d")
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf,
        start=start,
        end=end,
    )

    bars = client.get_stock_bars(request).df
    bars = bars.reset_index(level=0, drop=True)  # drop symbol level
    bars.columns = [c.title() for c in bars.columns]
    return preprocess(bars)


# ─── CCXT crypto fetcher ──────────────────────────────────────────────────────

def fetch_ccxt(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    start: str = "2020-01-01",
    limit_per_call: int = 1000,
    timeframe: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from any CCXT-compatible crypto exchange.

    Paginates automatically to pull full history.

    Args:
        symbol:         CCXT pair (e.g. 'BTC/USDT').
        exchange_id:    CCXT exchange id (e.g. 'binance', 'kraken').
        start:          ISO date string.
        limit_per_call: Candles per API call (max varies by exchange).
        timeframe:      CCXT timeframe string: '1m','5m','1h','1d'.
        use_cache:      Cache to parquet on disk.

    Public candle history does not need API keys. CCXT_API_KEY and
    CCXT_SECRET are attached only when both are set in the environment.

    Returns:
        Preprocessed OHLCV DataFrame.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("Install ccxt: pip install ccxt")

    safe_sym = symbol.replace("/", "_")
    cache = _cache_path(safe_sym, start, "now", timeframe)
    if use_cache:
        cached = _load_cache(cache)
        if cached is not None:
            return cached

    exchange_class = getattr(ccxt, exchange_id)
    kwargs: dict = {"enableRateLimit": True}
    api_key = os.environ.get("CCXT_API_KEY", "")
    secret = os.environ.get("CCXT_SECRET", "")
    if api_key and secret:
        kwargs["apiKey"] = api_key
        kwargs["secret"] = secret
    exchange = exchange_class(kwargs)

    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    all_candles: list = []

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms,
                                        limit=limit_per_call)
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1
        if len(candles) < limit_per_call:
            break

    df = pd.DataFrame(all_candles,
                      columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df.set_index("Timestamp", inplace=True)

    df = preprocess(df)
    if use_cache:
        _save_cache(df, cache)
    return df
