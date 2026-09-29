"""
src/optimizer.py
================
Parameter optimization for trading strategies.

Supported methods:
    - Grid Search     : Exhaustive search over specified value grid.
    - Random Search   : Randomly sampled configurations (faster).
    - Bayesian Search : Hyperopt Tree-Parzen Estimator (most efficient).

Parameter search scores walk-forward test folds on a prefix of the sample.
The final holdout is scored once, after the choice is made, and is not
an input to that choice.

Usage::

    opt = StrategyOptimizer(
        strategy_cls=MeanReversionStrategy,
        df=price_df,
        param_space={"lookback": [10, 20, 30], "rsi_period": [10, 14, 21]},
        metric="sharpe",
        method="bayesian",
        n_trials=200,
    )
    best_params, results_df = opt.run()
    print(best_params)
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from src.backtester import VectorizedBacktester, PerformanceMetrics
from src.strategies import BaseStrategy


# ─── Optimizer ────────────────────────────────────────────────────────────────

MetricName = Literal["sharpe", "sortino", "calmar", "total_return"]


@dataclass
class StrategyOptimizer:
    """
    Hyperparameter optimizer for trading strategies.

    Args:
        strategy_cls:   Strategy class (not instance).
        df:             Full OHLCV DataFrame.
        param_space:    Dict mapping param name to list of values (grid/random)
                        or a hyperopt hp.* expression dict (Bayesian).
        metric:         Objective metric to maximise.
        method:         'grid', 'random', or 'bayesian'.
        n_trials:       Number of trials for random/Bayesian search.
        cv_folds:       Walk-forward folds inside the search prefix.
        holdout_fraction: Tail fraction excluded from selection. The chosen
                        parameters are scored on it once and stored on
                        ``oos_score``.
        initial_capital:Backtesting starting capital.
        commission:     One-way commission fraction.
        slippage:       One-way slippage fraction.
        n_jobs:         Parallel jobs for grid/random search (-1 = all CPUs).
    """

    strategy_cls: type[BaseStrategy]
    df: pd.DataFrame
    param_space: dict[str, Any]
    metric: MetricName = "sharpe"
    method: Literal["grid", "random", "bayesian"] = "bayesian"
    n_trials: int = 100
    cv_folds: int = 5
    holdout_fraction: float = 0.2
    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    n_jobs: int = -1

    _results: list[dict] = field(default_factory=list, init=False, repr=False)
    oos_score: float | None = field(default=None, init=False, repr=False)

    def run(self) -> tuple[dict[str, Any], pd.DataFrame]:
        """
        Execute optimization.

        Returns:
            (best_params, results_df)
            best_params: Parameter dict with the highest walk-forward score
                         on the search prefix. The holdout is not used.
            results_df:  Search results sorted by that score. ``oos_score``
                         on the optimizer is the untouched tail.
        """
        self._search_df, self._holdout_df = self._split_holdout(self.df)
        if self.method == "grid":
            best, results = self._grid_search()
        elif self.method == "random":
            best, results = self._random_search()
        elif self.method == "bayesian":
            best, results = self._bayesian_search()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.oos_score = self._score_holdout(best)
        results.attrs["oos_score"] = self.oos_score
        results.attrs["holdout_fraction"] = self.holdout_fraction
        logger.info(
            f"Holdout {self.metric}={self.oos_score:.3f} "
            "(not used to choose parameters)"
        )
        return best, results

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _split_holdout(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cut a tail the search is not allowed to see."""
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between 0 and 1")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2 for a walk-forward split")
        cut = int(len(df) * (1 - self.holdout_fraction))
        min_search = self.cv_folds + 2
        if cut < min_search or cut >= len(df):
            raise ValueError(
                f"Need more than {min_search} rows before a "
                f"{self.holdout_fraction:.0%} holdout; got {len(df)}."
            )
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()

    def _metric_value(self, metrics: PerformanceMetrics) -> float:
        values = {
            "sharpe": metrics.sharpe_ratio,
            "sortino": metrics.sortino_ratio,
            "calmar": metrics.calmar_ratio,
            "total_return": metrics.total_return,
        }
        score = float(values[self.metric])
        return score if np.isfinite(score) else -np.inf

    def _evaluate(self, params: dict[str, Any]) -> float:
        """
        Mean walk-forward score on the search prefix only.

        Each fold generates signals on history through the end of the test
        window and records the metric on that window. The holdout tail is
        not in ``_search_df``.
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        bt = VectorizedBacktester(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage,
        )
        scores: list[float] = []

        for _train_idx, test_idx in tscv.split(self._search_df):
            history = self._search_df.iloc[: test_idx[-1] + 1]
            try:
                strategy = self.strategy_cls(**params)
                signals_df = strategy.generate_signals(history)
                scored = bt.run(signals_df, metrics_start=self._search_df.index[test_idx[0]])
                scores.append(self._metric_value(scored))
            except Exception as e:
                logger.debug(f"Eval failed for {params}: {e}")
                scores.append(-np.inf)

        finite = [s for s in scores if np.isfinite(s)]
        return float(np.mean(finite)) if finite else -np.inf

    def _score_holdout(self, params: dict[str, Any]) -> float:
        """Score the already chosen parameters on the untouched tail."""
        bt = VectorizedBacktester(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage,
        )
        try:
            strategy = self.strategy_cls(**params)
            signals_df = strategy.generate_signals(self.df)
            scored = bt.run(signals_df, metrics_start=self._holdout_df.index[0])
            return self._metric_value(scored)
        except Exception as e:
            logger.warning(f"Holdout evaluation failed: {e}")
            return -np.inf

    # ── Grid Search ───────────────────────────────────────────────────────────

    def _grid_search(self) -> tuple[dict, pd.DataFrame]:
        keys = list(self.param_space.keys())
        values = [self.param_space[k] for k in keys]
        combos = list(itertools.product(*values))
        logger.info(f"Grid search: {len(combos)} combinations × {self.cv_folds} folds")

        def _eval_combo(combo):
            params = dict(zip(keys, combo))
            score = self._evaluate(params)
            return {**params, self.metric: score}

        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_eval_combo)(c) for c in combos
            )
        except ImportError:
            results = [_eval_combo(c) for c in combos]

        return self._finalise(results)

    # ── Random Search ─────────────────────────────────────────────────────────

    def _random_search(self) -> tuple[dict, pd.DataFrame]:
        keys = list(self.param_space.keys())
        values = [self.param_space[k] for k in keys]
        results: list[dict] = []
        logger.info(f"Random search: {self.n_trials} trials × {self.cv_folds} folds")

        for _ in range(self.n_trials):
            sampled = [random.choice(v) for v in values]
            params = dict(zip(keys, sampled))
            score = self._evaluate(params)
            results.append({**params, self.metric: score})

        return self._finalise(results)

    # ── Bayesian (hyperopt) ───────────────────────────────────────────────────

    def _bayesian_search(self) -> tuple[dict, pd.DataFrame]:
        try:
            from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
        except ImportError:
            raise ImportError("Install hyperopt: pip install hyperopt")

        # Build hyperopt space from lists (uniform choice)
        from hyperopt import hp as hp_module
        hp_space = {
            k: hp_module.choice(k, v) if isinstance(v, list) else v
            for k, v in self.param_space.items()
        }

        trials = Trials()
        results: list[dict] = []

        def objective(params):
            # hyperopt resolves hp.choice to indices – unwrap
            resolved = {}
            for k, v in params.items():
                choices = self.param_space.get(k)
                if isinstance(choices, list):
                    resolved[k] = choices[v] if isinstance(v, int) else v
                else:
                    resolved[k] = v

            score = self._evaluate(resolved)
            results.append({**resolved, self.metric: score})
            return {"loss": -score, "status": STATUS_OK if np.isfinite(score) else STATUS_FAIL}

        logger.info(f"Bayesian search: {self.n_trials} trials × {self.cv_folds} folds")
        fmin(objective, space=hp_space, algo=tpe.suggest,
             max_evals=self.n_trials, trials=trials, verbose=False)

        return self._finalise(results)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _finalise(self, results: list[dict]) -> tuple[dict, pd.DataFrame]:
        df = pd.DataFrame(results).sort_values(self.metric, ascending=False)
        df.reset_index(drop=True, inplace=True)
        best = df.iloc[0].to_dict()
        metric_val = best.pop(self.metric)
        logger.info(f"Best params: {best}  |  {self.metric}={metric_val:.3f}")
        return best, df
