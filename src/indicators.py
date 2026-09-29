"""
src/indicators.py
=================
Pure-function technical indicator library.

Design principles:
    - Every function is a pure function: no side effects, no mutation.
    - All inputs/outputs are pd.Series or pd.DataFrame.
    - Functions are composable; pipe them to build complex signals.
    - NaN-safe: guard against empty or too-short series gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_min_length(series: pd.Series, min_len: int, name: str) -> None:
    if len(series) < min_len:
        raise ValueError(f"{name} requires at least {min_len} data points; "
                         f"got {len(series)}.")


# ─── Moving Averages ──────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    _check_min_length(series, period, "SMA")
    return series.rolling(period).mean().rename(f"SMA_{period}")


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    _check_min_length(series, period, "EMA")
    return series.ewm(span=period, adjust=False).mean().rename(f"EMA_{period}")


def wma(series: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average (linearly weighted)."""
    _check_min_length(series, period, "WMA")
    weights = np.arange(1, period + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        return np.dot(x, weights) / weights.sum()

    return series.rolling(period).apply(_wma, raw=True).rename(f"WMA_{period}")


# ─── Momentum / Oscillators ───────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder smoothing).

    Returns values in [0, 100].  Typical thresholds: 30 oversold / 70 overbought.
    """
    _check_min_length(series, period + 1, "RSI")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"RSI_{period}")


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

    Returns DataFrame with columns: MACD, Signal, Histogram.
    """
    _check_min_length(series, slow + signal, "MACD")
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = (fast_ema - slow_ema).rename("MACD")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("Signal")
    histogram = (macd_line - signal_line).rename("Histogram")
    return pd.concat([macd_line, signal_line, histogram], axis=1)


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).

    Returns DataFrame with columns: Stoch_K, Stoch_D.
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"Stoch_K": k, "Stoch_D": d})


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change (momentum)."""
    return ((series - series.shift(period)) / series.shift(period) * 100).rename(
        f"ROC_{period}"
    )


# ─── Volatility ───────────────────────────────────────────────────────────────

def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands (Middle, Upper, Lower) + %B + Bandwidth.

    %B = (price - lower) / (upper - lower); values outside [0,1] indicate extremes.
    Bandwidth = (upper - lower) / middle; rising = expanding volatility.
    """
    _check_min_length(series, period, "Bollinger Bands")
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    pct_b = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle
    return pd.DataFrame({
        "BB_Middle": middle,
        "BB_Upper": upper,
        "BB_Lower": lower,
        "BB_PctB": pct_b,
        "BB_Width": bandwidth,
    })


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range (Wilder smoothing).

    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().rename(f"ATR_{period}")


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels based on EMA ± multiplier * ATR."""
    middle = ema(close, ema_period).rename("KC_Middle")
    atr_val = atr(high, low, close, atr_period)
    upper = (middle + multiplier * atr_val).rename("KC_Upper")
    lower = (middle - multiplier * atr_val).rename("KC_Lower")
    return pd.concat([middle, upper, lower], axis=1)


# ─── Volume ───────────────────────────────────────────────────────────────────

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int | None = None,
) -> pd.Series:
    """
    Volume Weighted Average Price.

    If period is None, computes cumulative VWAP (anchored to start of series).
    If period is set, computes rolling VWAP over that window.
    """
    typical = (high + low + close) / 3
    tp_vol = typical * volume
    if period is None:
        return (tp_vol.cumsum() / volume.cumsum()).rename("VWAP")
    return (tp_vol.rolling(period).sum() / volume.rolling(period).sum()).rename(
        f"VWAP_{period}"
    )


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum().rename("OBV")


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow."""
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_volume = mf_multiplier * volume
    return (mf_volume.rolling(period).sum() / volume.rolling(period).sum()).rename(
        f"CMF_{period}"
    )


# ─── Trend ────────────────────────────────────────────────────────────────────

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index (+DI, -DI, ADX).

    ADX > 25 indicates a strong trend.
    """
    tr_val = atr(high, low, close, period)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm_s / tr_val
    minus_di = 100 * minus_dm_s / tr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return pd.DataFrame({
        "ADX": adx_val,
        "+DI": plus_di,
        "-DI": minus_di,
    })


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    SuperTrend indicator.

    Returns DataFrame with columns: SuperTrend, Direction (1=up, -1=down).
    """
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_basic = hl2 + multiplier * atr_val
    lower_basic = hl2 - multiplier * atr_val

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    supertrend_vals = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)

    for i in range(1, len(close)):
        # Adjust bands
        upper_band.iloc[i] = (
            upper_basic.iloc[i]
            if upper_basic.iloc[i] < upper_band.iloc[i - 1]
               or close.iloc[i - 1] > upper_band.iloc[i - 1]
            else upper_band.iloc[i - 1]
        )
        lower_band.iloc[i] = (
            lower_basic.iloc[i]
            if lower_basic.iloc[i] > lower_band.iloc[i - 1]
               or close.iloc[i - 1] < lower_band.iloc[i - 1]
            else lower_band.iloc[i - 1]
        )
        # Determine direction
        if close.iloc[i] <= upper_band.iloc[i]:
            direction.iloc[i] = -1
            supertrend_vals.iloc[i] = upper_band.iloc[i]
        else:
            direction.iloc[i] = 1
            supertrend_vals.iloc[i] = lower_band.iloc[i]

    return pd.DataFrame({"SuperTrend": supertrend_vals, "ST_Direction": direction})


# ─── Z-Score / Statistical ────────────────────────────────────────────────────

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score: (price - rolling_mean) / rolling_std."""
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ((series - mean) / std).rename(f"ZScore_{period}")


def spread_zscore(
    series_a: pd.Series,
    series_b: pd.Series,
    period: int = 60,
) -> pd.Series:
    """
    Z-score of a pairs spread whose hedge ratio is fit on past bars only.

    The regression at bar *t* uses ``[t - period, t)``. The current prices
    are the residual being scored, and they are not inside the fit. The
    full sample is never used.
    """
    from scipy import stats

    if len(series_a) != len(series_b):
        raise ValueError("spread_zscore requires aligned series of equal length")

    spreads: list[float] = []
    a_values = series_a.to_numpy(dtype=float)
    b_values = series_b.to_numpy(dtype=float)
    for i in range(len(series_a)):
        if i < period:
            spreads.append(np.nan)
            continue
        a_window = a_values[i - period:i]
        b_window = b_values[i - period:i]
        if np.std(b_window) == 0:
            spreads.append(np.nan)
            continue
        slope, intercept, *_ = stats.linregress(b_window, a_window)
        spreads.append(float(a_values[i] - slope * b_values[i] - intercept))

    spread_series = pd.Series(spreads, index=series_a.index)
    return zscore(spread_series, period).rename("SpreadZScore")
