"""Paper fills, position limits, and the closed live-order gate."""
from __future__ import annotations

import pytest

from src.backtester import VectorizedBacktester
from src.executor import AlpacaExecutor, CCXTExecutor, PaperExecutor
from src.risk_manager import (
    actions_for_bar,
    apply_risk_filter,
    fixed_pct_size,
    kelly_size,
    order_to_target,
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


def test_stacking_shorts_cannot_exceed_equity_but_a_cover_can():
    book = PaperExecutor(initial_capital=100_000, commission=0, slippage=0)
    book.set_price("AAA", 100)
    assert book.submit_order("AAA", "sell", 600).status == "filled"
    assert book.get_positions()["AAA"].qty == -600
    # A second short of 600 would leave 1,200 shares, $120k of notional.
    assert book.submit_order("AAA", "sell", 600).status == "rejected"
    assert book.get_positions()["AAA"].qty == -600

    book.set_price("AAA", 200)
    # Mark-to-market loss does not block the cover.
    assert book.submit_order("AAA", "buy", 600).status == "filled"
    assert book.get_positions() == {}
    assert book.get_equity() == pytest.approx(40_000)


def test_paper_replay_matches_next_close_backtest():
    """The in-memory broker and the vectorized engine fill on the same bar.

    Signal turns on at the second close (100) and off at the fourth (110).
    Both paths buy the 100 print and sell the later 110 print.
    """
    df = pd.DataFrame(
        {
            "Close": [100, 100, 100, 110, 110],
            "Position": [0, 1, 1, 0, 0],
        },
        index=pd.bdate_range("2020-01-01", periods=5),
    )
    metrics = VectorizedBacktester(
        initial_capital=100_000, commission=0, slippage=0, risk_free_rate=0
    ).run(df)

    book = PaperExecutor(initial_capital=100_000, commission=0, slippage=0)
    for i in range(1, len(df)):
        price = float(df["Close"].iloc[i])
        book.set_price("AAA", price)
        held = book.get_positions().get("AAA")
        current = held.qty if held else 0
        order = order_to_target(
            current,
            int(df["Position"].iloc[i - 1]),
            fixed_pct_size(book.get_equity(), price, risk_pct=1.0),
        )
        if order is not None:
            assert book.submit_order("AAA", order[0], order[1]).status == "filled"

    assert metrics.total_return == pytest.approx(0.10)
    assert metrics.equity_curve.iloc[-1] == pytest.approx(110_000)
    assert book.get_equity() == pytest.approx(110_000)
    assert book.get_positions() == {}


def test_order_to_target_flips_and_halt_does_not_reenter():
    assert order_to_target(40, -1, 10) == ("sell", 50)
    assert order_to_target(-10, 1, 10) == ("buy", 20)
    assert order_to_target(40, 1, 10) is None
    assert order_to_target(40, 0, 10) == ("sell", 40)
    assert actions_for_bar(
        halted=True, target=1, current_qty=40, open_qty=10, stop_hit=False
    ) == [("sell", 40)]
    assert actions_for_bar(
        halted=False, target=-1, current_qty=40, open_qty=10, stop_hit=True
    ) == [("sell", 40)]


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
