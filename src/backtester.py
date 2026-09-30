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

from dataclasses import dataclass

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


def _round_trip_returns(equity: pd.Series, positions: pd.Series) -> list[float]:
    """Percent return of each position, including entry and exit costs.

    The base is the equity on the bar before the fill, so the entry cost
    is part of the trade. A sign flip closes one trade and opens another.
    """
    pos = positions.fillna(0).to_numpy(dtype=float)
    eq = equity.to_numpy(dtype=float)
    trips: list[float] = []
    entry_i: int | None = None

    def _append(exit_i: int, open_i: int) -> None:
        base_i = open_i - 1 if open_i > 0 else open_i
        base = eq[base_i]
        end = eq[exit_i]
        if np.isfinite(base) and base != 0 and np.isfinite(end):
            trips.append(float(end / base - 1))

    for i, cur in enumerate(pos):
        prev = float(pos[i - 1]) if i else 0.0
        sign_flip = prev * cur < 0
        closed = (prev != 0 and cur == 0) or sign_flip
        opened = (prev == 0 and cur != 0) or sign_flip
        if closed and entry_i is not None:
            _append(i, entry_i)
            entry_i = None
        if opened:
            entry_i = i
    if entry_i is not None:
        _append(len(eq) - 1, entry_i)
    return trips


def execution_lag_rows(signals_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Split the latest bars into ``(decision, fill)``.

    A signal that uses the close of bar *t* is not tradable at that close.
    ``decision`` is the last completed signal bar. ``fill`` is the next bar,
    which is the first bar where that decision may be executed. Callers must
    trade ``decision["Position"]`` at the fill bar's price, never the fill
    bar's own signal.
    """
    if len(signals_df) < 2:
        raise ValueError("Need a completed signal bar and a later fill bar")
    return signals_df.iloc[-2], signals_df.iloc[-1]


def _lookup_signal(signals: pd.Series, when) -> float:
    """Match a backtrader bar date to a signal index.

    Signal indexes are often timezone-aware timestamps. Backtrader exposes
    the bar as a naive date, and a direct ``Series.get`` on that timestamp
    misses every row.
    """
    indexed = signals.copy()
    indexed.index = pd.to_datetime(indexed.index)
    if getattr(indexed.index, "tz", None) is not None:
        indexed.index = indexed.index.tz_localize(None)
    indexed.index = indexed.index.normalize()
    indexed = indexed[~indexed.index.duplicated(keep="last")]
    key = pd.Timestamp(when)
    if key.tzinfo is not None:
        key = key.tz_localize(None)
    key = key.normalize()
    value = indexed.get(key, 0)
    return float(0 if pd.isna(value) else value)


def _slice_from(
    equity: pd.Series,
    position: pd.Series,
    metrics_start,
) -> tuple[pd.Series, pd.Series]:
    """Keep the bar before ``metrics_start`` as the equity base."""
    start_ts = pd.Timestamp(metrics_start)
    if equity.index.tz is not None and start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(equity.index.tz)
    elif equity.index.tz is None and start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    later = equity.index[equity.index >= start_ts]
    if len(later) == 0:
        raise ValueError("metrics_start is after the last bar")
    pos = int(equity.index.get_loc(later[0]))
    start = max(pos - 1, 0)
    return equity.iloc[start:], position.iloc[start:]


def _compute_metrics(
    equity: pd.Series,
    positions: pd.Series,
    bars_per_year: int = 252,
    risk_free_rate: float = 0.04,
) -> PerformanceMetrics:
    """Compute all performance metrics from equity curve and position series."""
    returns = equity.pct_change().dropna()

    # Annualisation uses the number of return observations. Counting the
    # starting equity snapshot as a bar overstates the sample length.
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = len(returns) / bars_per_year
    if total_ret <= -1:
        annual_ret = -1.0
    else:
        annual_ret = (1 + total_ret) ** (1 / max(n_years, 1e-6)) - 1
    vol_annual = returns.std() * np.sqrt(bars_per_year)

    # Sharpe
    excess = returns - risk_free_rate / bars_per_year
    sharpe = (excess.mean() / returns.std() * np.sqrt(bars_per_year)
              if returns.std() > 0 else 0.0)

    # Sortino uses the full-sample downside deviation. The standard
    # deviation of the negative returns alone drops the zero-contribution
    # up days and is undefined when only one down bar exists.
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0
    sortino = (float(excess.mean()) / downside_dev * np.sqrt(bars_per_year)
               if downside_dev > 0 else 0.0)

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

    # One round trip per entry. A flip closes the open trade and opens
    # the next. The position still open on the last bar is marked to
    # the last equity value so it is not dropped from the stats.
    trade_returns_list = _round_trip_returns(equity, positions)
    num_trades = len(trade_returns_list)

    win_rate = (np.array(trade_returns_list) > 0).mean() if trade_returns_list else 0.0
    gains = sum(r for r in trade_returns_list if r > 0)
    losses = abs(sum(r for r in trade_returns_list if r < 0))
    profit_factor = gains / losses if losses > 0 else (np.inf if gains > 0 else 0.0)
    avg_trade_ret = float(np.mean(trade_returns_list)) if trade_returns_list else 0.0

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
        - Fills on the next bar's close (no intrabar path).
        - Positions are weights (+1, -1, or 0), not share counts.
        - No partial fills or order books.

    Args:
        initial_capital: Starting portfolio value in dollars.
        commission:      One-way commission as a fraction of traded notional
                         (0.001 = 10 bps each time the weight changes).
        slippage:        One-way slippage as a fraction of traded notional
                         (0.0005 = 5 bps). Charged on turnover, with commission.
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
        metrics_start=None,
    ) -> PerformanceMetrics:
        """
        Run vectorized backtest.

        ``Position`` on bar *t* is the holding desired after seeing that
        bar's close. It is filled at the next bar's close, so it does not
        earn the close-to-close return that starts at the signal bar.

        Commission and slippage are one-way costs on absolute turnover,
        charged at the fill. A flip from long to short turns over two
        units of weight.

        Args:
            df:            Signal DataFrame with Position and Close columns.
            position_col:  Column name for the desired position (+1/-1/0).
            price_col:     Column name for the mark-to-market price.
            metrics_start: If set, Sharpe and the other stats use returns
                           from this timestamp forward. The prior bar is kept
                           only as the equity base so the first in-window
                           return is not dropped. Parameter search uses this
                           to score a test fold without fitting on it.

        Returns:
            PerformanceMetrics dataclass with full statistics + equity curve.
        """
        frame = df[[position_col, price_col]].copy().dropna()
        signal = frame[position_col]
        price = frame[price_col]
        # Position[t] is known only once close[t] exists, so it cannot be
        # the fill at close[t]. The order goes out at close[t+1].
        # The weight that earned close[t-1] -> close[t] was filled at
        # close[t-1], which is Position[t-2].
        filled = signal.shift(1).fillna(0.0)
        held = signal.shift(2).fillna(0.0)

        price_return = price.pct_change().fillna(0.0)
        strategy_return = held * price_return
        turnover = (filled - held).abs()
        trading_cost = turnover * (self.commission + self.slippage)
        net_return = strategy_return - trading_cost
        equity = self.initial_capital * (1.0 + net_return).cumprod()

        if metrics_start is not None:
            equity, filled = _slice_from(equity, filled, metrics_start)
        if len(equity) < 2:
            raise ValueError("Need at least two bars to compute returns")

        # Trade stats follow the position after the fill, so an exit that
        # completes on the last close is a closed trade.
        metrics = _compute_metrics(equity, filled, self.bars_per_year,
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

    Backtrader fills these market orders at the next bar's open. The
    vectorized engine fills at the next bar's close. The two equity curves
    are different execution models and are not expected to match.

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
            # The bar is complete, so its signal is known. Backtrader fills
            # the market order on the next bar; this method does not fill
            # at the signal close.
            sig = _lookup_signal(self.signals, self.datas[0].datetime.date(0))
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
    Score a fixed strategy on each walk-forward test fold.

    The factory is not refit. Each fold sees history only through the end
    of that test fold, and the reported metrics use the test window. Bars
    before the test window warm up indicators; they are not a parameter
    search. ``StrategyOptimizer`` is what holds out a final untouched tail.

    Args:
        df:               Full OHLCV DataFrame.
        strategy_factory: Callable() -> strategy with ``generate_signals``.
        n_splits:         Number of TimeSeriesSplit folds.
        initial_capital:  Capital per fold.
        commission:       One-way commission fraction.
        slippage:         One-way slippage fraction.

    Returns:
        List of PerformanceMetrics, one per out-of-sample fold.
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    bt = VectorizedBacktester(initial_capital, commission, slippage)
    fold_metrics: list[PerformanceMetrics] = []

    for fold, (_train_idx, test_idx) in enumerate(tscv.split(df)):
        history = df.iloc[: test_idx[-1] + 1]
        strategy = strategy_factory()
        try:
            signals_df = strategy.generate_signals(history)
            metrics = bt.run(signals_df, metrics_start=df.index[test_idx[0]])
            fold_metrics.append(metrics)
            logger.info(f"Fold {fold+1}/{n_splits} Sharpe={metrics.sharpe_ratio:.2f}")
        except Exception as e:
            logger.warning(f"Fold {fold+1} failed: {e}")

    return fold_metrics


def buy_and_hold_equity(close: pd.Series, capital: float) -> pd.Series:
    """Dollar path of holding the asset from the first close in ``close``.

    The series starts at ``capital`` on the first timestamp. Later points
    are ``capital * close_t / close_0``. No costs are charged.
    """
    prices = close.astype(float)
    if len(prices) == 0:
        raise ValueError("buy-and-hold needs at least one close")
    base = float(prices.iloc[0])
    if not np.isfinite(base) or base == 0:
        raise ValueError("buy-and-hold start price must be a non-zero finite number")
    return capital * prices / base
