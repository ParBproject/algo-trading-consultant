"""Pairs hedge ratio and half-life must not use bars that come later."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.indicators import spread_zscore
from src.strategies import PairsTradingStrategy


def _leg(values: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2019-01-01", periods=len(values))
    close = pd.Series(values, index=index, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(close), 1_000),
        },
        index=index,
    )


def test_hedge_ratio_excludes_the_current_bar():
    period = 10
    b = np.arange(40, dtype=float) + 10
    a = 1.5 * b + 0.25
    a[20] += 50  # residual at the bar being scored, not a regression point
    a[21:] += 80  # future path the fit must ignore

    z = spread_zscore(pd.Series(a), pd.Series(b), period)
    i = 20
    slope, intercept, *_ = stats.linregress(b[i - period:i], a[i - period:i])
    assert slope == pytest.approx(1.5)
    causal_spread = a[i] - slope * b[i] - intercept
    in_sample_residual = a[i - 1] - slope * b[i - 1] - intercept
    assert abs(causal_spread) > abs(in_sample_residual) + 10
    assert np.isfinite(z.iloc[i])


def test_future_prices_do_not_change_earlier_zscores():
    period = 15
    b = np.linspace(50, 80, 80)
    a = 2.0 * b + np.sin(np.linspace(0, 8, 80))
    base = spread_zscore(pd.Series(a), pd.Series(b), period)

    shocked = a.copy()
    shocked[-12:] += 500
    later = spread_zscore(pd.Series(shocked), pd.Series(b), period)
    pd.testing.assert_series_equal(base.iloc[:-12], later.iloc[:-12])


def test_generate_signals_uses_a_formation_half_life():
    b = np.linspace(40, 70, 90)
    a = 0.8 * b + np.sin(np.linspace(0, 6, 90))
    strategy = PairsTradingStrategy(lookback=20, entry_z=2.0, exit_z=0.5)
    original = strategy.generate_signals(_leg(a), _leg(b))

    shocked = a.copy()
    shocked[-25:] += 1_000
    revised = strategy.generate_signals(_leg(shocked), _leg(b))

    assert original["HalfLife"].iloc[0] == pytest.approx(revised["HalfLife"].iloc[0])
    pd.testing.assert_series_equal(
        original["SpreadZScore"].iloc[:-25],
        revised["SpreadZScore"].iloc[:-25],
    )
    assert {"Position_A", "Position_B", "HalfLife"} <= set(original.columns)


def test_pairs_call_requires_both_legs():
    with pytest.raises(TypeError):
        PairsTradingStrategy().generate_signals(_leg(np.arange(30, dtype=float)))
