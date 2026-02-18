"""
src/backtester.py
=================
Vectorized backtesting engine with realistic cost modeling.

Provides:
    VectorizedBacktester  – Fast pandas-based simulation (good for optimization).
    BacktraderRunner      – Wraps backtrader for event-driven realism.
    PerformanceMetrics    – Sharpe, Sortino, Calmar, max drawdown, win rate, etc.

Usage::

    bt = VectorizedBacktester(initial_capital=100_000, commission=0.001,
                              slippage=0.0005)
    results = bt.run(signals_df)
    print(results.metrics())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ─── Performance metrics ──────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    """Container for all backtest performance statistics."""

    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int   # bars
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    volatility_annual: float
    equity_curve: pd.Series

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "equity_curve"}

    def summary(self) -> str:
        lines = [
            "=" * 45,
            "  BACKTEST PERFORMANCE SUMMARY",
            "=" * 45,
            f"  Total Return       : {self.total_return:>10.2%}",
            f"  Annual Return      : {self.annual_return:>10.2%}",
            f"  Volatility (ann.)  : {self.volatility_annual:>10.2%}",
            f"  Sharpe Ratio       : {self.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio      : {self.sortino_ratio:>10.2f}",
            f"  Calmar Ratio       : {self.calmar_ratio:>10.2f}",
            f"  Max Drawdown       : {self.max_drawdown:>10.2%}",
            f"  DD Duration (bars) : {self.max_drawdown_duration:>10d}",
            f"  # Trades           : {self.num_trades:>10d}",
            f"  Win Rate           : {self.win_rate:>10.2%}",
            f"  Profit Factor      : {self.profit_factor:>10.2f}",
            "=" * 45,
        ]
        return "\n".join(lines)


def _compute_metrics(
    equity: pd.Series,
    positions: pd.Series,
    bars_per_year: int = 252,
    risk_free_rate: float = 0.04,
) -> PerformanceMetrics:
    """Compute all performance metrics from equity curve and position series."""
    returns = equity.pct_change().dropna()

    # Annualisation
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(equity) / bars_per_year
    annual_ret = (1 + total_ret) ** (1 / max(n_years, 1e-6)) - 1
    vol_annual = returns.std() * np.sqrt(bars_per_year)

    # Sharpe
    excess = returns - risk_free_rate / bars_per_year
    sharpe = (excess.mean() / returns.std() * np.sqrt(bars_per_year)
              if returns.std() > 0 else 0.0)

    # Sortino (downside only)
    downside = returns[returns < 0]
    sortino = (excess.mean() / downside.std() * np.sqrt(bars_per_year)
               if len(downside) > 0 and downside.std() > 0 else 0.0)

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Drawdown duration
    in_dd = (drawdown < 0).astype(int)
    dd_runs = (in_dd != in_dd.shift()).cumsum()
    dd_dur = in_dd.groupby(dd_runs).sum()
    max_dd_dur = int(dd_dur.max()) if len(dd_dur) > 0 else 0

    # Calmar
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0.0

    # Trade-level stats
    trade_changes = positions.diff().fillna(0)
    trade_entries = trade_changes[trade_changes != 0].index
    num_trades = len(trade_entries)

    trade_returns_list: list[float] = []
    entry_i: Optional[int] = None
    entry_pos: int = 0
    pos_arr = positions.values
    ret_arr = returns.values
    eq_arr = equity.values

    # Simple per-trade P&L using equity snapshots
    prev_eq = equity.iloc[0]
    for i in range(1, len(equity)):
        if positions.iloc[i] != positions.iloc[i - 1]:
            if entry_i is not None:
                trade_ret = (eq_arr[i] - prev_eq) / prev_eq
                trade_returns_list.append(trade_ret)
            prev_eq = eq_arr[i]
            entry_i = i

    win_rate = (np.array(trade_returns_list) > 0).mean() if trade_returns_list else 0.0
    gains = sum(r for r in trade_returns_list if r > 0)
    losses = abs(sum(r for r in trade_returns_list if r < 0))
    profit_factor = gains / losses if losses > 0 else np.inf
    avg_trade_ret = np.mean(trade_returns_list) if trade_returns_list else 0.0

    return PerformanceMetrics(
        total_return=total_ret,
        annual_return=annual_ret,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=num_trades,
        avg_trade_return=avg_trade_ret,
        volatility_annual=vol_annual,
        equity_curve=equity,
    )


# ─── Vectorized Backtester ────────────────────────────────────────────────────

@dataclass
class VectorizedBacktester:
    """
    Fast vectorized backtester using pandas/numpy.

    Suitable for:
        - Strategy development and rapid iteration.
        - Parameter optimization (thousands of runs).

    Limitations:
        - Assumes next-bar execution (no intrabar simulation).
        - No partial fills or order books.

    Args:
        initial_capital: Starting portfolio value in dollars.
        commission:      Round-trip commission as fraction (0.001 = 0.1%).
        slippage:        Slippage as fraction of price (0.0005 = 0.05%).
        bars_per_year:   Trading bars per year (252 for daily stocks).
        risk_free_rate:  Annual risk-free rate for Sharpe calculation.
    """

    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    bars_per_year: int = 252
    risk_free_rate: float = 0.04

    def run(
        self,
        df: pd.DataFrame,
        position_col: str = "Position",
        price_col: str = "Close",
    ) -> PerformanceMetrics:
        """
        Run vectorized backtest.

        Args:
            df:            Signal DataFrame with Position and Close columns.
            position_col:  Column name for position signal (+1/-1/0).
            price_col:     Column name for execution price.

        Returns:
            PerformanceMetrics dataclass with full statistics + equity curve.
        """
        df = df[[position_col, price_col]].copy().dropna()
        position = df[position_col].shift(1).fillna(0)  # execute next bar
        price = df[price_col]

        # Slippage: degrade fill price by slippage in direction of trade
        trade = position.diff().fillna(0)
        fill_price = price * (1 + self.slippage * np.sign(trade))

        # Returns: position * price return
        price_return = price.pct_change()
        strategy_return = position * price_return

        # Commission: charge on every position change
        commission_drag = trade.abs() * self.commission
        net_return = strategy_return - commission_drag

        # Equity curve
        equity = self.initial_capital * (1 + net_return).cumprod()
        equity.iloc[0] = self.initial_capital

        metrics = _compute_metrics(equity, position, self.bars_per_year,
                                   self.risk_free_rate)
        logger.info(f"Backtest complete. Sharpe={metrics.sharpe_ratio:.2f}, "
                    f"MaxDD={metrics.max_drawdown:.2%}")
        return metrics


# ─── Backtrader event-driven runner ───────────────────────────────────────────

def run_backtrader(
    df: pd.DataFrame,
    strategy_signals: pd.Series,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> PerformanceMetrics:
    """
    Run event-driven simulation via backtrader.

    Wraps a pre-computed signal series into a backtrader strategy so that
    order fills, commissions, and slippage are handled by the backtrader engine.

    Args:
        df:                OHLCV DataFrame.
        strategy_signals:  Signal series aligned to df index (+1/-1/0).
        initial_capital:   Starting cash.
        commission:        Fractional commission per trade.
        slippage:          Fixed slippage fraction applied to fill price.

    Returns:
        PerformanceMetrics with equity curve from backtrader.

    Note:
        Requires ``backtrader`` to be installed.
    """
    try:
        import backtrader as bt
    except ImportError:
        raise ImportError("Install backtrader: pip install backtrader")

    signals = strategy_signals.reindex(df.index).fillna(0)

    class SignalBridge(bt.Strategy):
        """Translate pre-computed signals into backtrader orders."""

        def __init__(self):
            self.signals = signals
            self.equity_history: list[float] = []

        def next(self):
            dt = self.datas[0].datetime.date(0)
            ts = pd.Timestamp(dt)
            sig = self.signals.get(ts, 0)
            current_pos = self.getposition().size

            if sig == 1 and current_pos <= 0:
                self.order_target_percent(target=0.95)
            elif sig == -1 and current_pos >= 0:
                self.order_target_percent(target=-0.95)
            elif sig == 0 and current_pos != 0:
                self.close()

            self.equity_history.append(self.broker.getvalue())

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SignalBridge)

    # Feed OHLCV data
    data_feed = bt.feeds.PandasData(
        dataname=df[["Open", "High", "Low", "Close", "Volume"]].copy(),
        dtformat="%Y-%m-%d",
    )
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage)

    results = cerebro.run()
    strat = results[0]

    equity = pd.Series(strat.equity_history, index=df.index[-len(strat.equity_history):])
    return _compute_metrics(equity, signals[-len(equity):])


# ─── Walk-forward helper ──────────────────────────────────────────────────────

def walk_forward_test(
    df: pd.DataFrame,
    strategy_factory,
    n_splits: int = 5,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> list[PerformanceMetrics]:
    """
    Run walk-forward out-of-sample validation.

    Args:
        df:               Full OHLCV DataFrame.
        strategy_factory: Callable() -> BaseStrategy instance.
        n_splits:         Number of folds (in-sample fit, out-of-sample test).
        initial_capital:  Capital per fold.
        commission:       Commission fraction.
        slippage:         Slippage fraction.

    Returns:
        List of PerformanceMetrics, one per out-of-sample fold.
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    bt = VectorizedBacktester(initial_capital, commission, slippage)
    fold_metrics: list[PerformanceMetrics] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        test_df = df.iloc[test_idx]
        strategy = strategy_factory()
        try:
            signals_df = strategy.generate_signals(test_df)
            metrics = bt.run(signals_df)
            fold_metrics.append(metrics)
            logger.info(f"Fold {fold+1}/{n_splits} Sharpe={metrics.sharpe_ratio:.2f}")
        except Exception as e:
            logger.warning(f"Fold {fold+1} failed: {e}")

    return fold_metrics
