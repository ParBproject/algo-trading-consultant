"""Benchmark chart uses the same start date as the strategy equity."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.utils import plot_equity_curve


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
