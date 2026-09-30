"""OHLCV preprocessing, including the yfinance column layout."""
from __future__ import annotations

import pandas as pd
import pytest

from data.fetcher import preprocess


def _ohlcv(index: pd.DatetimeIndex, closes: list[float]) -> list[list[float]]:
    rows = []
    for close in closes:
        rows.append([close - 0.5, close + 0.5, close - 1, close, 1_000])
    return rows


def test_preprocess_flattens_price_then_ticker_columns():
    index = pd.bdate_range("2020-01-01", periods=4)
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(_ohlcv(index, [10.5, 11.0, 11.5, 12.5]), index=index, columns=columns)
    out = preprocess(raw)
    assert not isinstance(out.columns, pd.MultiIndex)
    assert out["Close"].iloc[-1] == pytest.approx(12.5)
    assert out.index.tz is not None


def test_preprocess_flattens_ticker_then_price_columns():
    index = pd.bdate_range("2020-01-01", periods=3)
    columns = pd.MultiIndex.from_product(
        [["AAPL"], ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Price"],
    )
    raw = pd.DataFrame(_ohlcv(index, [5.0, 6.0, 7.0]), index=index, columns=columns)
    out = preprocess(raw)
    assert out["Close"].iloc[-1] == pytest.approx(7.0)


def test_preprocess_keeps_a_flat_frame():
    index = pd.bdate_range("2020-01-01", periods=3)
    raw = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 4],
            "volume": [10, 10, 10],
        },
        index=index,
    )
    out = preprocess(raw)
    assert out["Close"].iloc[-1] == pytest.approx(4)
