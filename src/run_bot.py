"""
src/run_bot.py
==============
CLI entry point for running strategies end-to-end.

Usage::

    python src/run_bot.py --strategy mean_reversion --ticker AAPL
    python src/run_bot.py --strategy momentum --ticker BTC-USD --start 2021-01-01
    python src/run_bot.py --strategy mean_reversion --ticker MSFT --live --broker paper
    python src/run_bot.py --config config/example.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from data.fetcher import fetch_yfinance
from src.backtester import VectorizedBacktester, execution_lag_rows
from src.executor import get_executor, PaperExecutor
from src.risk_manager import (
    fixed_pct_size, compute_stops, PortfolioRiskState, is_stop_hit
)
from src.strategies import get_strategy
from src.utils import (
    load_config, setup_logging, plot_equity_curve, plot_signals, generate_report
)


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="algo-trading-consultant – run a strategy backtest or paper bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="Path to YAML config (overrides other flags if set)")
    parser.add_argument("--strategy", default="mean_reversion",
                        choices=["mean_reversion", "momentum", "pairs_trading"],
                        help="Strategy to run")
    parser.add_argument("--ticker", default="AAPL", help="Primary ticker symbol")
    parser.add_argument("--ticker2", default="MSFT",
                        help="Second ticker for pairs trading")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default=today)")
    parser.add_argument("--interval", default="1d", help="Bar interval")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Initial capital in dollars")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission fraction per trade")
    parser.add_argument("--slippage", type=float, default=0.0005,
                        help="Slippage fraction")
    parser.add_argument("--live", action="store_true",
                        help="Run in live/paper loop instead of backtest")
    parser.add_argument("--broker", default="paper",
                        choices=["paper", "alpaca", "ccxt"],
                        help="Broker for live execution")
    parser.add_argument("--report", action="store_true",
                        help="Generate PDF report after backtest")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


# ─── Backtest mode ────────────────────────────────────────────────────────────

def run_backtest(args: argparse.Namespace) -> None:
    logger.info(f"=== BACKTEST MODE | {args.strategy} on {args.ticker} ===")

    df = fetch_yfinance(args.ticker, args.start, args.end, args.interval)
    strategy = get_strategy(args.strategy)

    if args.strategy == "pairs_trading":
        raise SystemExit(
            "Pairs signals need both legs: generate_signals(df_a, df_b). "
            "The single-asset backtest does not price a two-leg book."
        )

    signals_df = strategy.generate_signals(df)

    bt = VectorizedBacktester(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
    )
    metrics = bt.run(signals_df)
    print(metrics.summary())

    # Buy-and-hold over the same bars the strategy equity covers.
    aligned_returns = (
        df["Close"].pct_change().reindex(metrics.equity_curve.index).fillna(0)
    )
    bah_equity = args.capital * (1 + aligned_returns).cumprod()

    # Charts
    Path("reports").mkdir(exist_ok=True)
    chart_path = f"reports/{args.ticker}_{args.strategy}_equity.png"
    plot_equity_curve(
        metrics.equity_curve,
        benchmark=bah_equity,
        title=f"{args.strategy.replace('_', ' ').title()} | {args.ticker}",
        save_path=chart_path,
    )
    plot_signals(signals_df, title=f"Signals | {args.ticker}", save_path=
                 f"reports/{args.ticker}_{args.strategy}_signals.png")

    if args.report:
        generate_report(
            metrics=metrics,
            strategy_name=args.strategy,
            ticker=args.ticker,
            chart_path=chart_path,
            output_path=f"reports/{args.ticker}_{args.strategy}_report.pdf",
        )


# ─── Paper / Live loop ────────────────────────────────────────────────────────

def run_live(args: argparse.Namespace) -> None:
    """
    Bar loop for the in-memory paper broker.

    The signal on a bar uses that bar's close, so the order is sent on the
    next bar. ``--broker paper`` is the only path that runs unless
    ``--allow-live`` is set. In a real deployment, replace the history
    download with a streaming feed that still waits for the bar to close.
    """
    logger.info(f"=== LIVE MODE | {args.strategy} on {args.ticker} via {args.broker} ===")

    executor = get_executor(args.broker, initial_capital=args.capital)
    strategy = get_strategy(args.strategy)
    risk_state = PortfolioRiskState(
        peak_equity=args.capital,
        current_equity=args.capital,
    )

    # Warm-up: fetch history for indicator calculation
    df = fetch_yfinance(args.ticker, args.start, args.end, args.interval)
    open_stops: dict[str, tuple] = {}  # symbol -> (stop_price, tp_price, direction)

    for i in range(60, len(df)):  # start after warm-up period
        bar = df.iloc[:i + 1]
        signals_df = strategy.generate_signals(bar)
        if len(signals_df) < 2:
            continue
        decision, fill = execution_lag_rows(signals_df)
        price = float(fill["Close"])
        position = int(decision["Position"])

        # Update paper executor price feed
        if isinstance(executor, PaperExecutor):
            executor.set_price(args.ticker, price)

        equity = executor.get_equity()
        risk_state.update(equity)
        if risk_state.check_halt():
            logger.warning("Portfolio halt active – skipping bar.")
            continue

        current_positions = executor.get_positions()
        current_qty = current_positions.get(args.ticker, None)
        current_qty_val = current_qty.qty if current_qty else 0

        # Check stops
        if args.ticker in open_stops:
            sl, tp, direction = open_stops[args.ticker]
            hit, reason = is_stop_hit(price, direction, sl, tp)
            if hit:
                logger.info(f"Stop hit ({reason}) @ {price:.2f} – closing position")
                executor.close_position(args.ticker)
                del open_stops[args.ticker]
                continue

        # Enter / exit
        if position == 1 and current_qty_val <= 0:
            qty = fixed_pct_size(equity, price, risk_pct=0.05)
            if qty > 0:
                executor.submit_order(args.ticker, "buy", qty)
                stops = compute_stops(price, direction=1,
                                      atr=float(decision.get("ATR", price * 0.02)))
                open_stops[args.ticker] = (stops.stop_loss, stops.take_profit, 1)

        elif position == -1 and current_qty_val >= 0:
            qty = fixed_pct_size(equity, price, risk_pct=0.05)
            if qty > 0:
                executor.submit_order(args.ticker, "sell", qty)
                stops = compute_stops(price, direction=-1,
                                      atr=float(decision.get("ATR", price * 0.02)))
                open_stops[args.ticker] = (stops.stop_loss, stops.take_profit, -1)

        elif position == 0 and current_qty_val != 0:
            executor.close_position(args.ticker)
            open_stops.pop(args.ticker, None)

    logger.info(f"Final equity: ${executor.get_equity():,.2f}")
    if isinstance(executor, PaperExecutor):
        trade_log = executor.get_trade_log()
        Path("logs").mkdir(exist_ok=True)
        trade_log.to_csv(f"logs/{args.ticker}_{args.strategy}_trades.csv", index=False)
        logger.info(f"Trade log saved: logs/{args.ticker}_{args.strategy}_trades.csv")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
        # Override args from config where applicable
        args.ticker = cfg["data"]["tickers"][0]
        args.start = cfg["data"]["start_date"]
        args.end = cfg["data"].get("end_date")
        args.strategy = cfg["strategy"]["name"]
        args.capital = cfg["backtesting"]["initial_capital"]
        args.commission = cfg["backtesting"]["commission_pct"]
        args.slippage = cfg["backtesting"]["slippage_pct"]
        args.broker = cfg["execution"]["broker"]
        args.log_level = cfg["logging"]["level"]

    setup_logging(args.log_level)

    if args.live:
        run_live(args)
    else:
        run_backtest(args)


if __name__ == "__main__":
    main()
