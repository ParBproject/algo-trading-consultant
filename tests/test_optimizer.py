"""The search prefix chooses parameters. The tail only reports a score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimizer import StrategyOptimizer, resolve_sampled_params
from src.strategies import BaseStrategy


class SpyStrategy(BaseStrategy):
    """Records every frame it is shown and holds a constant weight."""

    seen: list[pd.Timestamp] = []

    def __init__(self, flag: int = 1):
        self.flag = flag

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        SpyStrategy.seen.append(df.index.max())
        out = df.copy()
        out["Position"] = int(self.flag)
        return out


def _prices(n: int = 80) -> pd.DataFrame:
    index = pd.bdate_range("2018-01-01", periods=n)
    close = np.linspace(100, 140, n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
        },
        index=index,
    )


def _optimizer(df: pd.DataFrame) -> StrategyOptimizer:
    return StrategyOptimizer(
        strategy_cls=SpyStrategy,
        df=df,
        param_space={"flag": [0, 1]},
        metric="total_return",
        method="grid",
        cv_folds=3,
        holdout_fraction=0.25,
        commission=0.0,
        slippage=0.0,
        n_jobs=1,
    )


def test_search_does_not_read_the_holdout_and_still_scores_it():
    SpyStrategy.seen = []
    df = _prices()
    opt = _optimizer(df)
    best, _results = opt.run()

    holdout_start = opt._holdout_df.index[0]
    search_calls = SpyStrategy.seen[:-1]
    assert search_calls, "walk-forward search made no calls"
    assert all(stamp < holdout_start for stamp in search_calls)
    assert SpyStrategy.seen[-1] >= opt._holdout_df.index[-1]
    assert best["flag"] == 1
    assert type(best["flag"]) is int
    assert opt.oos_score is not None


def test_wrecking_the_holdout_does_not_change_the_selected_params():
    df = _prices()
    crashed = df.copy()
    cut = int(len(df) * 0.75)
    crashed.loc[crashed.index[cut:], "Close"] = np.linspace(140, 20, len(df) - cut)

    baseline, _ = _optimizer(df).run()
    opt_crash = _optimizer(crashed)
    chosen, _ = opt_crash.run()

    assert chosen == baseline
    assert opt_crash.oos_score < 0


def test_choice_values_are_not_used_as_indexes():
    assert resolve_sampled_params({"lookback": [10, 20, 30]}, {"lookback": 20}) == {
        "lookback": 20
    }
    assert resolve_sampled_params({"lookback": [10, 20, 30]}, {"lookback": 0}) == {
        "lookback": 10
    }


class _Coded(BaseStrategy):
    seen: list[int] = []

    def __init__(self, code: int = 0):
        self.code = code

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        _Coded.seen.append(int(self.code))
        out = df.copy()
        out["Position"] = 1
        return out


def test_random_search_repeats_for_the_same_seed():
    df = _prices(40)

    def _run(seed: int) -> list[int]:
        _Coded.seen = []
        StrategyOptimizer(
            strategy_cls=_Coded,
            df=df,
            param_space={"code": list(range(100))},
            metric="total_return",
            method="random",
            n_trials=6,
            cv_folds=2,
            holdout_fraction=0.25,
            commission=0.0,
            slippage=0.0,
            n_jobs=1,
            seed=seed,
        ).run()
        return list(_Coded.seen)

    assert _run(0) == _run(0)
    assert _run(0) != _run(1)


def test_bayesian_search_keeps_integer_choice_values():
    df = _prices(40)

    class _Lookback(BaseStrategy):
        def __init__(self, lookback: int = 10):
            self.lookback = lookback

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            if self.lookback not in (10, 20, 30):
                raise AssertionError(self.lookback)
            out = df.copy()
            out["Position"] = 1 if self.lookback == 20 else 0
            return out

    best, _results = StrategyOptimizer(
        strategy_cls=_Lookback,
        df=df,
        param_space={"lookback": [10, 20, 30]},
        metric="total_return",
        method="bayesian",
        n_trials=4,
        cv_folds=2,
        holdout_fraction=0.25,
        commission=0.0,
        slippage=0.0,
        seed=0,
    ).run()
    assert best["lookback"] in (10, 20, 30)
    assert type(best["lookback"]) is int
