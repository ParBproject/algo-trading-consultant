"""Charts: benchmark rebase and signal entry/exit markers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.utils import plot_equity_curve, plot_signals


def test_benchmark_line_starts_at_100_on_the_strategy_window():
    index = pd.bdate_range("2020-01-01", periods=4)
    equity = pd.Series([100.0, 110.0, 120.0], index=index[1:])
    # Full-sample path starts at 50, before the strategy window.
    benchmark = pd.Series([50.0, 80.0, 80.0, 100.0], index=index)
    fig = plot_equity_curve(equity, benchmark=benchmark)
    line = fig.axes[0].get_lines()[1].get_ydata()
    assert line[0] == pytest.approx(100)
    assert line[-1] == pytest.approx(125)
    plt.close(fig)


def _scatter_ys(fig: plt.Figure, label: str) -> list[float]:
    for collection in fig.axes[0].collections:
        if collection.get_label() == label:
            return [float(point[1]) for point in collection.get_offsets()]
    raise AssertionError(f"no scatter labeled {label}")


def test_exit_marker_is_the_flat_transition_not_every_zero_signal():
    index = pd.bdate_range("2020-01-01", periods=6)
    df = pd.DataFrame(
        {
            "Close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "Signal": [0, 1, 0, 0, 0, 0],
            "Position": [0, 1, 1, 1, 0, 0],
        },
        index=index,
    )
    fig = plot_signals(df)
    assert _scatter_ys(fig, "Long Entry") == pytest.approx([11.0])
    assert _scatter_ys(fig, "Short Entry") == []
    assert _scatter_ys(fig, "Exit") == pytest.approx([14.0])
    plt.close(fig)


def test_exit_marker_uses_signal_when_position_is_absent():
    index = pd.bdate_range("2020-01-01", periods=4)
    df = pd.DataFrame(
        {"Close": [10.0, 11.0, 12.0, 13.0], "Signal": [1, 1, 0, 0]},
        index=index,
    )
    fig = plot_signals(df)
    assert _scatter_ys(fig, "Exit") == pytest.approx([12.0])
    plt.close(fig)


def test_pairs_leg_exit_uses_the_matching_position_column():
    index = pd.bdate_range("2020-01-01", periods=4)
    df = pd.DataFrame(
        {
            "Close": [10.0, 11.0, 12.0, 13.0],
            "Signal_A": [1, 0, 0, 0],
            "Position_A": [1, 1, 0, 0],
            "Position": [0, 0, 0, 0],
        },
        index=index,
    )
    fig = plot_signals(df, signal_col="Signal_A")
    assert _scatter_ys(fig, "Long Entry") == pytest.approx([10.0])
    assert _scatter_ys(fig, "Exit") == pytest.approx([12.0])
    plt.close(fig)
