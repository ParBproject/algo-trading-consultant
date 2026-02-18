"""
src/strategies.py
=================
Strategy classes for generating trading signals.

Architecture:
    - BaseStrategy defines the interface: generate_signals(df) -> pd.DataFrame
    - Each strategy is a pure-ish class (state only in config, never in data).
    - Signals: +1 = long, -1 = short, 0 = flat/exit.

Included strategies:
    1. MeanReversionStrategy  – RSI + Bollinger Bands z-score
    2. MomentumStrategy       – MACD crossover + ATR trend filter
    3. PairsTradingStrategy   – Cointegration-based spread mean reversion
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.indicators import (
    atr, bollinger_bands, macd, rsi, sma, spread_zscore, zscore, adx
)


# ─── Base class ───────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given OHLCV data (with columns Open/High/Low/Close/Volume),
        return a DataFrame that *includes* all original columns plus:
            Signal  : int  – +1 long, -1 short, 0 flat
            Position: int  – current held position after applying signal
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"


# ─── 1. Mean Reversion ────────────────────────────────────────────────────────

@dataclass
class MeanReversionStrategy(BaseStrategy):
    """
    Mean-Reversion Bot using RSI + Bollinger Band z-score.

    Entry rules:
        Long  : RSI < rsi_oversold  AND  price < BB_Lower
        Short : RSI > rsi_overbought AND  price > BB_Upper

    Exit rules:
        Long  : RSI > 50  OR  price crosses above BB_Middle
        Short : RSI < 50  OR  price crosses below BB_Middle

    Args:
        lookback:        Period for Bollinger Bands and z-score.
        rsi_period:      RSI lookback.
        rsi_oversold:    RSI threshold for oversold (long signal).
        rsi_overbought:  RSI threshold for overbought (short signal).
        bb_std:          Bollinger Band standard deviation multiplier.
        allow_short:     If False, only long trades are taken.
    """
    name: str = "mean_reversion"
    lookback: int = 20
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_std: float = 2.0
    allow_short: bool = False

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        bb = bollinger_bands(df["Close"], self.lookback, self.bb_std)
        df = pd.concat([df, bb], axis=1)
        df["RSI"] = rsi(df["Close"], self.rsi_period)
        df["ZScore"] = zscore(df["Close"], self.lookback)

        # --- Entry conditions ---
        long_entry = (df["RSI"] < self.rsi_oversold) & (df["Close"] < df["BB_Lower"])
        short_entry = (df["RSI"] > self.rsi_overbought) & (df["Close"] > df["BB_Upper"])

        # --- Exit conditions ---
        long_exit = (df["RSI"] > 50) | (df["Close"] > df["BB_Middle"])
        short_exit = (df["RSI"] < 50) | (df["Close"] < df["BB_Middle"])

        # --- Build signal series ---
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[long_entry] = 1
        if self.allow_short:
            signal[short_entry] = -1

        # Apply exits: if we have a position and the exit fires, go flat
        position = signal.copy()
        for i in range(1, len(position)):
            prev = position.iloc[i - 1]
            if position.iloc[i] == 0:  # no new entry
                if prev == 1 and long_exit.iloc[i]:
                    position.iloc[i] = 0
                elif prev == -1 and short_exit.iloc[i]:
                    position.iloc[i] = 0
                else:
                    position.iloc[i] = prev  # hold

        df["Signal"] = signal
        df["Position"] = position
        logger.debug(f"[{self.name}] Signals generated: "
                     f"longs={signal[signal==1].count()}, "
                     f"shorts={signal[signal==-1].count()}")
        return df.dropna(subset=["RSI", "BB_Middle"])


# ─── 2. Momentum ──────────────────────────────────────────────────────────────

@dataclass
class MomentumStrategy(BaseStrategy):
    """
    Momentum Bot using MACD crossover filtered by ADX trend strength.

    Entry rules:
        Long  : MACD line crosses above Signal line  AND  ADX > adx_threshold
        Short : MACD line crosses below Signal line  AND  ADX > adx_threshold

    Exit rules:
        Long  : MACD line crosses below Signal line
        Short : MACD line crosses above Signal line

    Optional ATR trailing stop applied during position holding.

    Args:
        fast_period:    MACD fast EMA period.
        slow_period:    MACD slow EMA period.
        signal_period:  MACD signal line smoothing.
        adx_period:     ADX calculation period.
        adx_threshold:  Minimum ADX to confirm trend (default 25).
        atr_period:     ATR period for position sizing reference.
        allow_short:    Enable short trades.
    """
    name: str = "momentum"
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    adx_period: int = 14
    adx_threshold: float = 25.0
    atr_period: int = 14
    allow_short: bool = True

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        macd_df = macd(df["Close"], self.fast_period, self.slow_period, self.signal_period)
        df = pd.concat([df, macd_df], axis=1)
        adx_df = adx(df["High"], df["Low"], df["Close"], self.adx_period)
        df = pd.concat([df, adx_df], axis=1)
        df["ATR"] = atr(df["High"], df["Low"], df["Close"], self.atr_period)

        # MACD crossovers
        macd_cross_up = (df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) <= df["Signal"].shift(1))
        macd_cross_dn = (df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) >= df["Signal"].shift(1))
        strong_trend = df["ADX"] > self.adx_threshold

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[macd_cross_up & strong_trend] = 1
        if self.allow_short:
            signal[macd_cross_dn & strong_trend] = -1

        # Carry position forward until opposite signal
        position = _carry_signal(signal)

        df["Signal"] = signal
        df["Position"] = position
        logger.debug(f"[{self.name}] Crossovers: up={macd_cross_up.sum()}, "
                     f"down={macd_cross_dn.sum()}, strong={strong_trend.sum()}")
        return df.dropna(subset=["MACD", "ADX"])


# ─── 3. Pairs Trading ─────────────────────────────────────────────────────────

@dataclass
class PairsTradingStrategy(BaseStrategy):
    """
    Statistical Arbitrage via cointegration-based pairs trading.

    The strategy trades the spread between two correlated assets.
    Spread = price_A - hedge_ratio * price_B

    Entry rules:
        Long  spread (long A, short B) : SpreadZScore < -entry_z
        Short spread (short A, long B) : SpreadZScore >  entry_z

    Exit rules:
        Spread mean-reverts: |SpreadZScore| < exit_z

    Args:
        entry_z:         Z-score threshold to enter.
        exit_z:          Z-score threshold to exit.
        lookback:        Rolling window for spread regression.
        min_half_life:   Minimum half-life (bars) for cointegration check.
        max_half_life:   Maximum half-life (bars) to accept as tradeable.
    """
    name: str = "pairs_trading"
    entry_z: float = 2.0
    exit_z: float = 0.5
    lookback: int = 60
    min_half_life: float = 1.0
    max_half_life: float = 126.0  # ~6 months in trading days

    def generate_signals(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate signals for a pair.

        Args:
            df_a: OHLCV DataFrame for asset A.
            df_b: OHLCV DataFrame for asset B (same index required).

        Returns:
            DataFrame with columns: Close_A, Close_B, SpreadZScore,
            Signal_A (position in A), Signal_B (position in B).
        """
        close_a = df_a["Close"].rename("Close_A")
        close_b = df_b["Close"].rename("Close_B")
        df = pd.concat([close_a, close_b], axis=1).dropna()

        # Check cointegration half-life
        half_life = self._half_life(df["Close_A"] - df["Close_B"])
        logger.info(f"[{self.name}] Spread half-life: {half_life:.1f} bars")

        if not (self.min_half_life <= half_life <= self.max_half_life):
            logger.warning(f"[{self.name}] Half-life {half_life:.1f} outside "
                           f"[{self.min_half_life}, {self.max_half_life}]. "
                           "Pair may not be cointegrated.")

        df["SpreadZScore"] = spread_zscore(df["Close_A"], df["Close_B"], self.lookback)

        signal_a = pd.Series(0, index=df.index, dtype=int)
        signal_b = pd.Series(0, index=df.index, dtype=int)

        # Long spread = long A, short B
        long_spread = df["SpreadZScore"] < -self.entry_z
        # Short spread = short A, long B
        short_spread = df["SpreadZScore"] > self.entry_z

        signal_a[long_spread] = 1
        signal_b[long_spread] = -1
        signal_a[short_spread] = -1
        signal_b[short_spread] = 1

        # Exit when spread reverts
        exit_cond = df["SpreadZScore"].abs() < self.exit_z

        pos_a = _carry_signal_with_exit(signal_a, exit_cond)
        pos_b = _carry_signal_with_exit(signal_b, exit_cond)

        df["Signal_A"] = signal_a
        df["Signal_B"] = signal_b
        df["Position_A"] = pos_a
        df["Position_B"] = pos_b
        return df.dropna(subset=["SpreadZScore"])

    # Override required by ABC but pairs uses two DataFrames
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        raise NotImplementedError(
            "PairsTradingStrategy.generate_signals() requires two DataFrames. "
            "Call generate_signals(df_a, df_b) instead."
        )

    @staticmethod
    def _half_life(spread: pd.Series) -> float:
        """Estimate mean-reversion half-life via OLS."""
        from scipy import stats
        delta = spread.diff().dropna()
        lag = spread.shift(1).dropna()
        slope, *_ = stats.linregress(lag, delta)
        if slope >= 0:
            return np.inf  # not mean-reverting
        return -np.log(2) / slope


# ─── Signal utilities ─────────────────────────────────────────────────────────

def _carry_signal(signal: pd.Series) -> pd.Series:
    """Carry a non-zero signal forward until the opposite signal fires."""
    position = signal.copy()
    for i in range(1, len(position)):
        if position.iloc[i] == 0:
            position.iloc[i] = position.iloc[i - 1]
    return position


def _carry_signal_with_exit(
    signal: pd.Series,
    exit_mask: pd.Series,
) -> pd.Series:
    """Carry signal forward, but set to 0 when exit_mask is True."""
    position = signal.copy()
    for i in range(1, len(position)):
        if position.iloc[i] == 0:
            if exit_mask.iloc[i]:
                position.iloc[i] = 0
            else:
                position.iloc[i] = position.iloc[i - 1]
    return position


# ─── Strategy factory ─────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseStrategy]] = {
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "pairs_trading": PairsTradingStrategy,
}


def get_strategy(name: str, **params: Any) -> BaseStrategy:
    """
    Factory function: instantiate a strategy by name with kwargs.

    Args:
        name:    One of 'mean_reversion', 'momentum', 'pairs_trading'.
        **params: Strategy-specific parameters.

    Returns:
        Instantiated strategy object.

    Example:
        >>> strat = get_strategy("momentum", fast_period=10, slow_period=21)
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. "
                         f"Available: {list(_REGISTRY)}")
    return cls(**params)
