"""Pins for next-bar fills, trading costs, and return accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    VectorizedBacktester,
    _compute_metrics,
    _lookup_signal,
    _round_trip_returns,
    execution_lag_rows,
)


def _frame(prices: list[float], position: list[int]) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=len(prices))
    return pd.DataFrame({"Close": prices, "Position": position}, index=index)


def test_signal_does_not_earn_the_bar_that_created_it():
    """A position that turns on as the price jumps must not capture that jump.

    Prices go 100 -> 110 on the same bar the signal becomes long. Next-bar
    execution holds the previous (flat) position across that return.
    """
    df = _frame([100, 100, 110, 110], [0, 0, 1, 1])
    metrics = VectorizedBacktester(
        initial_capital=100_000, commission=0, slippage=0, risk_free_rate=0
    ).run(df)

    assert metrics.total_return == pytest.approx(0.0)
    assert metrics.equity_curve.iloc[-1] == pytest.approx(100_000)


def test_signal_earns_the_following_bar():
    """The long signal on the bar before a 10% jump earns that next return."""
    df = _frame([100, 100, 110, 121], [0, 1, 1, 1])
    metrics = VectorizedBacktester(
        initial_capital=100_000, commission=0, slippage=0, risk_free_rate=0
    ).run(df)

    # 100 -> 110 -> 121 while long, two 10% steps.
    assert metrics.total_return == pytest.approx(0.21)
    assert metrics.equity_curve.iloc[-1] == pytest.approx(121_000)


def test_commission_and_slippage_are_charged_on_turnover():
    """Flat prices isolate costs. Entry and exit each pay commission + slippage."""
    commission = 0.001
    slippage = 0.0005
    df = _frame([100, 100, 100, 100, 100], [0, 1, 1, 0, 0])
    metrics = VectorizedBacktester(
        initial_capital=100_000,
        commission=commission,
        slippage=slippage,
        risk_free_rate=0,
    ).run(df)

    one_way = commission + slippage
    # Signal enters on bar 1 (filled on bar 2) and exits on bar 3 (filled on bar 4).
    expected = 100_000 * (1 - one_way) ** 2
    assert metrics.equity_curve.iloc[-1] == pytest.approx(expected)
    assert metrics.total_return < 0


def test_flip_pays_two_units_of_cost():
    df = _frame([100, 100, 100, 100], [1, 1, -1, -1])
    commission = 0.01
    metrics = VectorizedBacktester(
        initial_capital=100_000, commission=commission, slippage=0, risk_free_rate=0
    ).run(df)

    # Executed weights: 0, 1, 1, -1. The flip turns over 2.
    assert metrics.equity_curve.iloc[-1] == pytest.approx(100_000 * (1 - commission) * (1 - 2 * commission))


def test_sortino_uses_full_sample_downside_deviation():
    equity = pd.Series([100.0, 110.0, 121.0, 108.9])
    position = pd.Series([1, 1, 1, 1])
    metrics = _compute_metrics(equity, position, bars_per_year=252, risk_free_rate=0.0)

    returns = np.array([0.1, 0.1, -0.1])
    downside_dev = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    expected = returns.mean() / downside_dev * np.sqrt(252)
    assert metrics.sortino_ratio == pytest.approx(expected)
    assert metrics.sortino_ratio != 0


def test_round_trip_counts_an_entry_not_every_change():
    equity = pd.Series([100.0, 100.0, 90.0, 90.0, 100.0])
    position = pd.Series([0, 0, 1, 1, 0])
    trips = _round_trip_returns(equity, position)
    assert trips == pytest.approx([0.0])

    metrics = _compute_metrics(equity, position, risk_free_rate=0.0)
    assert metrics.num_trades == 1


def test_open_trade_is_marked_at_the_last_bar():
    equity = pd.Series([100.0, 100.0, 110.0])
    position = pd.Series([0, 1, 1])
    assert _round_trip_returns(equity, position) == pytest.approx([0.1])


def test_execution_lag_rows_use_the_prior_signal():
    df = _frame([10, 11, 12], [0, 1, -1])
    decision, fill = execution_lag_rows(df)
    assert int(decision["Position"]) == 1
    assert float(fill["Close"]) == 12
    with pytest.raises(ValueError):
        execution_lag_rows(df.iloc[:1])


def test_signal_lookup_matches_a_timezone_aware_index():
    index = pd.bdate_range("2020-01-01", periods=3, tz="UTC")
    signals = pd.Series([0, 1, -1], index=index)
    assert _lookup_signal(signals, index[1].date()) == 1
    assert _lookup_signal(signals, "1999-01-01") == 0


def test_metrics_start_drops_earlier_returns():
    df = _frame([100, 110, 110, 110], [1, 1, 1, 1])
    engine = VectorizedBacktester(commission=0, slippage=0, risk_free_rate=0)
    full = engine.run(df)
    tail = engine.run(df, metrics_start=df.index[2])
    # The 10% jump is on bar 1 and is inside the full window only.
    assert full.total_return == pytest.approx(0.1)
    assert tail.total_return == pytest.approx(0.0)
