"""YAML config reaches the strategy and the data request."""
from __future__ import annotations

import argparse

from src.run_bot import apply_config


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        interval="1d",
        allow_live=False,
        position_pct=0.02,
        strategy_params={},
    )


def _config() -> dict:
    return {
        "data": {
            "tickers": ["MSFT"],
            "start_date": "2021-01-01",
            "end_date": "2022-01-01",
            "interval": "1wk",
        },
        "strategy": {
            "name": "mean_reversion",
            "params": {
                "lookback": 15,
                "rsi_period": 10,
                "entry_z": 2.0,
                "fast_period": 8,
            },
        },
        "backtesting": {
            "initial_capital": 50_000,
            "commission_pct": 0.002,
            "slippage_pct": 0.001,
        },
        "execution": {"broker": "paper", "paper_trading": True},
        "risk": {"max_position_pct": 0.10},
        "logging": {"level": "WARNING"},
    }


def test_config_applies_interval_and_known_strategy_params():
    args = _args()
    ignored = apply_config(args, _config())
    assert args.ticker == "MSFT"
    assert args.interval == "1wk"
    assert args.strategy_params == {"lookback": 15, "rsi_period": 10}
    assert ignored == ["entry_z", "fast_period"]
    assert args.capital == 50_000
    assert args.position_pct == 0.10


def test_config_cannot_enable_a_live_broker_on_its_own():
    args = _args()
    cfg = _config()
    cfg["execution"]["broker"] = "alpaca"
    cfg["execution"]["paper_trading"] = False
    apply_config(args, cfg)
    assert args.broker == "paper"
