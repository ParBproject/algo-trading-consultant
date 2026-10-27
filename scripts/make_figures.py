"""Write the PNG files in ``screenshots/`` from Yahoo Finance.

Strategy charts use daily bars from 2018-01-01 through 2024-01-01.
The indicator chart uses 2022-01-01 through 2024-01-01.
Capital, commission, and slippage match ``config/example.yaml``.

The paper-broker chart precomputes the signal on the full sample. The
indicators and the position carry only read bars up to the current one,
so that column matches a prefix loop. Fills still use the next bar.

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

from data.fetcher import fetch_yfinance
from src.backtester import VectorizedBacktester, buy_and_hold_equity
from src.executor import PaperExecutor
from src.indicators import adx, bollinger_bands, macd, rsi
from src.optimizer import StrategyOptimizer
from src.risk_manager import (
    PortfolioRiskState,
    actions_for_bar,
    compute_stops,
    fixed_pct_size,
    is_stop_hit,
)
from src.strategies import MeanReversionStrategy, MomentumStrategy, PairsTradingStrategy
from src.utils import plot_equity_curve, plot_optimization_heatmap, plot_signals

OUT = ROOT / "screenshots"
START = "2018-01-01"
END = "2024-01-01"
CAPITAL = 100_000.0
COMMISSION = 0.001
SLIPPAGE = 0.0005

_OLD_NAMES = (
    "01_indicators_dashboard.png",
    "02_mean_reversion_strategy.png",
    "03_momentum_strategy.png",
    "04_pairs_trading.png",
    "05_optimization.png",
    "06_performance_summary.png",
    "07_paper_trading.png",
)


def _engine() -> VectorizedBacktester:
    return VectorizedBacktester(CAPITAL, COMMISSION, SLIPPAGE)


def _prices(ticker: str, start: str, end: str = END) -> pd.DataFrame:
    return fetch_yfinance(ticker, start, end, use_cache=False)


def indicators() -> None:
    df = _prices("AAPL", "2022-01-01")
    bands = bollinger_bands(df["Close"])
    macd_df = macd(df["Close"])
    adx_df = adx(df["High"], df["Low"], df["Close"])
    rsi_14 = rsi(df["Close"], 14)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df.index, df["Close"], color="black", lw=1, label="Close")
    axes[0].plot(bands.index, bands["BB_Upper"], lw=0.8, label="Upper")
    axes[0].plot(bands.index, bands["BB_Middle"], lw=0.8, label="Middle")
    axes[0].plot(bands.index, bands["BB_Lower"], lw=0.8, label="Lower")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("AAPL daily indicators, 2022-01-01 to 2024-01-01")

    axes[1].plot(rsi_14.index, rsi_14, color="purple", lw=1)
    axes[1].axhline(70, color="red", ls="--", lw=0.8)
    axes[1].axhline(30, color="green", ls="--", lw=0.8)
    axes[1].set_ylabel("RSI(14)")

    axes[2].plot(macd_df.index, macd_df["MACD"], label="MACD", lw=1)
    axes[2].plot(macd_df.index, macd_df["Signal"], label="Signal", lw=1)
    axes[2].bar(macd_df.index, macd_df["Histogram"], width=1.0, alpha=0.4)
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].set_ylabel("MACD")

    axes[3].plot(adx_df.index, adx_df["ADX"], color="navy", lw=1, label="ADX")
    axes[3].axhline(25, color="red", ls="--", lw=0.8, label="25")
    axes[3].legend(loc="upper left", fontsize=8)
    axes[3].set_ylabel("ADX(14)")

    fig.tight_layout()
    fig.savefig(OUT / "01_indicators.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def mean_reversion() -> None:
    df = _prices("AAPL", START)
    signals = MeanReversionStrategy().generate_signals(df)
    metrics = _engine().run(signals)
    plot_signals(
        signals,
        title="AAPL mean reversion signals, 2018-01-01 to 2024-01-01",
        save_path=str(OUT / "02_mean_reversion_signals.png"),
    )
    plt.close("all")
    benchmark = buy_and_hold_equity(
        df["Close"].reindex(metrics.equity_curve.index), CAPITAL
    )
    plot_equity_curve(
        metrics.equity_curve,
        benchmark=benchmark,
        title="AAPL mean reversion vs buy and hold, 2018-01-01 to 2024-01-01",
        save_path=str(OUT / "06_mean_reversion_equity.png"),
    )
    plt.close("all")
    print(metrics.summary())


def momentum() -> None:
    df = _prices("QQQ", START)
    signals = MomentumStrategy().generate_signals(df)
    metrics = _engine().run(signals)
    benchmark = buy_and_hold_equity(
        df["Close"].reindex(metrics.equity_curve.index), CAPITAL
    )
    plot_equity_curve(
        metrics.equity_curve,
        benchmark=benchmark,
        title="QQQ momentum vs buy and hold, 2018-01-01 to 2024-01-01",
        save_path=str(OUT / "03_momentum_equity.png"),
    )
    plt.close("all")
    print(metrics.summary())


def pairs() -> None:
    asset_a = _prices("AAPL", START)
    asset_b = _prices("MSFT", START)
    frame = PairsTradingStrategy().generate_signals(asset_a, asset_b)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame.index, frame["SpreadZScore"], color="black", lw=1)
    ax.axhline(2.0, color="red", ls="--", lw=0.8)
    ax.axhline(-2.0, color="green", ls="--", lw=0.8)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.axhline(-0.5, color="gray", ls=":", lw=0.8)
    ax.set_title("AAPL/MSFT spread z-score, lookback 60, 2018-01-01 to 2024-01-01")
    ax.set_ylabel("z-score")
    fig.tight_layout()
    fig.savefig(OUT / "04_pairs_zscore.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def optimization() -> None:
    df = _prices("AAPL", START)
    opt = StrategyOptimizer(
        strategy_cls=MeanReversionStrategy,
        df=df,
        param_space={
            "lookback": [10, 20, 30, 50],
            "rsi_period": [10, 14, 21],
        },
        metric="sharpe",
        method="grid",
        cv_folds=3,
        holdout_fraction=0.2,
        initial_capital=CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        n_jobs=1,
    )
    best, results = opt.run()
    plot_optimization_heatmap(
        results,
        param_x="lookback",
        param_y="rsi_period",
        metric="sharpe",
        title="AAPL mean reversion, walk-forward Sharpe on the search prefix",
        save_path=str(OUT / "05_optimization_heatmap.png"),
    )
    plt.close("all")
    print("search best", best)
    print("holdout sharpe (not used to choose)", opt.oos_score)


def paper() -> None:
    """Replay momentum on QQQ through PaperExecutor.

    Same order rule as ``run_live``: the signal on the previous signal bar
    is filled at this bar's close, with a 2% notional and ATR stops.
    """
    ticker = "QQQ"
    df = _prices(ticker, "2022-01-01")
    signals = MomentumStrategy().generate_signals(df)
    executor = PaperExecutor(CAPITAL, COMMISSION, SLIPPAGE)
    risk = PortfolioRiskState(peak_equity=CAPITAL, current_equity=CAPITAL)
    open_stops: dict[str, tuple] = {}
    equity: list[float] = []
    index: list = []

    for i in range(1, len(signals)):
        decision = signals.iloc[i - 1]
        fill = signals.iloc[i]
        if int(df.index.get_loc(fill.name)) < 60:
            continue
        price = float(fill["Close"])
        position = int(decision["Position"])
        executor.set_price(ticker, price)
        mark = executor.get_equity()
        risk.update(mark)

        held = executor.get_positions().get(ticker)
        current_qty = held.qty if held else 0
        stop_hit = False
        if ticker in open_stops:
            stop_loss, take_profit, direction = open_stops[ticker]
            stop_hit, _reason = is_stop_hit(price, direction, stop_loss, take_profit)
        halted = risk.check_halt()
        qty = fixed_pct_size(mark, price, risk_pct=0.02)
        atr_value = decision["ATR"] if pd.notna(decision["ATR"]) else price * 0.02
        for side, order_qty in actions_for_bar(
            halted=halted,
            target=position,
            current_qty=current_qty,
            open_qty=qty,
            stop_hit=stop_hit,
        ):
            result = executor.submit_order(ticker, side, order_qty)
            if result.status != "filled":
                continue
            if halted or stop_hit or position == 0:
                open_stops.pop(ticker, None)
            else:
                stops = compute_stops(price, direction=position, atr=float(atr_value))
                open_stops[ticker] = (stops.stop_loss, stops.take_profit, position)

        equity.append(executor.get_equity())
        index.append(fill.name)

    series = pd.Series(equity, index=pd.DatetimeIndex(index))
    fills = executor.get_trade_log()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, color="#2196F3", lw=1.2)
    ax.axhline(CAPITAL, color="gray", ls="--", lw=0.8)
    ax.set_title(
        "QQQ momentum paper broker, 2022-01-01 to 2024-01-01, "
        f"warmup 60 bars, {len(fills)} fills"
    )
    ax.set_ylabel("Equity ($)")
    fig.tight_layout()
    fig.savefig(OUT / "07_paper_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("paper ending equity", round(executor.get_equity(), 2), "fills", len(fills))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name in _OLD_NAMES:
        (OUT / name).unlink(missing_ok=True)
    indicators()
    mean_reversion()
    momentum()
    pairs()
    paper()
    optimization()


if __name__ == "__main__":
    main()
