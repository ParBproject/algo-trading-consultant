"""The search prefix chooses parameters. The tail only reports a score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimizer import StrategyOptimizer
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
