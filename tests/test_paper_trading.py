"""Paper fills, position limits, and the closed live-order gate."""
from __future__ import annotations

import pytest

from src.executor import AlpacaExecutor, CCXTExecutor, PaperExecutor
from src.risk_manager import (
    apply_risk_filter,
    fixed_pct_size,
    kelly_size,
    volatility_scaled_size,
)
import pandas as pd


def test_paper_round_trip_keeps_cash_without_costs():
    book = PaperExecutor(initial_capital=100_000, commission=0, slippage=0)
    book.set_price("AAA", 100)
    assert book.submit_order("AAA", "buy", 10).status == "filled"
    assert book.get_equity() == pytest.approx(100_000)
    book.set_price("AAA", 110)
    assert book.get_equity() == pytest.approx(100_100)
    assert book.submit_order("AAA", "sell", 10).status == "filled"
    assert book.get_positions() == {}
    assert book.get_equity() == pytest.approx(100_100)


def test_short_from_flat_loses_when_price_rises():
    book = PaperExecutor(initial_capital=100_000, commission=0, slippage=0)
    book.set_price("AAA", 100)
    order = book.submit_order("AAA", "sell", 10)
    assert order.status == "filled"
    assert book.get_positions()["AAA"].qty == -10
    assert book.get_equity() == pytest.approx(100_000)

    book.set_price("AAA", 110)
    assert book.get_equity() == pytest.approx(99_900)
    assert book.submit_order("AAA", "buy", 10).status == "filled"
    assert book.get_positions() == {}
    assert book.get_equity() == pytest.approx(99_900)


def test_buy_slippage_and_commission_hit_equity_once():
    book = PaperExecutor(initial_capital=100_000, commission=0.001, slippage=0.001)
    book.set_price("AAA", 100)
    book.submit_order("AAA", "buy", 10)
    # Fill 100.10, commission 10 bps of notional, mark stays at the mid.
    assert book.get_equity() == pytest.approx(100_000 - 10 * 0.10 - 1001 * 0.001)


def test_paper_rejects_orders_larger_than_equity():
    book = PaperExecutor(initial_capital=100_000, commission=0, slippage=0)
    book.set_price("AAA", 100)
    assert book.submit_order("AAA", "buy", 2_000).status == "rejected"
    assert book.submit_order("AAA", "sell", 2_000).status == "rejected"
    assert book.get_equity() == pytest.approx(100_000)
    assert book.get_positions() == {}


def test_live_routes_reject_constructor_keys_and_stay_off(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("CCXT_API_KEY", raising=False)
    monkeypatch.delenv("CCXT_SECRET", raising=False)

    with pytest.raises(TypeError):
        AlpacaExecutor(api_key="embedded", secret_key="embedded")
    with pytest.raises(RuntimeError, match="disabled"):
        AlpacaExecutor(paper=False)
    with pytest.raises(RuntimeError, match="environment"):
        AlpacaExecutor(paper=True)

    with pytest.raises(TypeError):
        CCXTExecutor(api_key="embedded", secret="embedded")
    with pytest.raises(RuntimeError, match="disabled"):
        CCXTExecutor(sandbox=False)
    with pytest.raises(RuntimeError, match="environment"):
        CCXTExecutor(sandbox=True)


def test_fixed_and_volatility_sizing_cannot_exceed_capital():
    assert fixed_pct_size(100_000, 50, risk_pct=0.02) == 40
    assert fixed_pct_size(100_000, 50, risk_pct=5, max_pct=1) == 2_000
    assert fixed_pct_size(0, 50) == 0
    assert fixed_pct_size(100_000, 0) == 0
    # A 1-cent ATR would otherwise request 100_000 shares against a $50 price.
    assert volatility_scaled_size(100_000, price=50, atr=0.01, target_risk_pct=0.01) == 2_000
    assert kelly_size(100_000, price=0, win_rate=0.6, avg_win=1, avg_loss=1) == 0


def test_drawdown_halt_stays_flat_after_the_breach():
    index = pd.RangeIndex(5)
    equity = pd.Series([100, 90, 80, 95, 110], index=index, dtype=float)
    signals = pd.DataFrame({"Position": [1, 1, 1, 1, 1]}, index=index)
    halted = apply_risk_filter(signals, equity, max_drawdown_limit=0.15)
    # 80 is 20% below the peak of 100. Later recovery does not re-enable risk.
    assert halted["Position"].tolist() == [1, 1, 0, 0, 0]
