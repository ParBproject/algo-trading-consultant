"""
src/risk_manager.py
===================
Position sizing, stop-loss, take-profit, and portfolio risk controls.

Philosophy:
    Risk management is applied *before* order submission.
    All functions are pure – they compute sizes/flags from inputs only.

Included:
    - Fixed percentage sizing
    - Volatility-scaled sizing (ATR-based)
    - Kelly criterion (fractional)
    - Stop-loss / take-profit / trailing stop price computation
    - Max drawdown circuit breaker
    - Portfolio heat check
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger


# ─── Position sizing ──────────────────────────────────────────────────────────

def fixed_pct_size(
    capital: float,
    price: float,
    risk_pct: float = 0.02,
    max_pct: float = 1.0,
) -> int:
    """
    Fixed-fraction notional sizing.

    ``risk_pct`` is the fraction of capital allocated to the position,
    not the distance to a stop. The result is capped by ``max_pct`` and
    is never larger than the account can pay for.

    Args:
        capital:   Available portfolio value in dollars.
        price:     Current asset price.
        risk_pct:  Fraction of capital to allocate (0.02 = 2%).
        max_pct:   Hard cap on that fraction (1.0 = no leverage).

    Returns:
        Number of shares/units to trade (rounded down).
    """
    if capital <= 0 or price <= 0 or risk_pct <= 0 or max_pct <= 0:
        return 0
    dollar_amount = capital * min(risk_pct, max_pct)
    return int(dollar_amount / price)


def volatility_scaled_size(
    capital: float,
    price: float,
    atr: float,
    target_risk_pct: float = 0.01,
    max_pct: float = 1.0,
) -> int:
    """
    Volatility-scaled position sizing (ATR-based).

    Shares = (capital * target_risk_pct) / ATR, so a more volatile asset
    gets a smaller position. Notional is capped at ``max_pct`` of capital
    so a tiny ATR cannot lever the account without bound.

    Args:
        capital:         Portfolio value.
        price:           Current price.
        atr:             Current ATR value.
        target_risk_pct: Fraction of capital at risk per ATR move.
        max_pct:         Maximum fraction of capital in notional terms.

    Returns:
        Number of shares/units.
    """
    if atr <= 0 or capital <= 0 or price <= 0 or target_risk_pct <= 0 or max_pct <= 0:
        return 0
    dollar_risk = capital * target_risk_pct
    shares = int(dollar_risk / atr)
    max_shares = int((capital * max_pct) / price)
    return max(0, min(shares, max_shares))


def kelly_size(
    capital: float,
    price: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.25,
    max_pct: float = 0.20,
) -> int:
    """
    Fractional Kelly criterion position sizing.

    Kelly% = (W/L * win_rate - (1-win_rate)) / (W/L)
    where W = avg_win, L = avg_loss (positive values).

    Args:
        capital:        Portfolio value.
        price:          Asset price.
        win_rate:       Historical win rate [0, 1].
        avg_win:        Average winning trade return (positive).
        avg_loss:       Average losing trade return (positive magnitude).
        kelly_fraction: Scale full Kelly (0.25 = quarter Kelly, safer).
        max_pct:        Cap position at this fraction of capital.

    Returns:
        Number of shares.
    """
    if avg_loss <= 0 or avg_win <= 0 or capital <= 0 or price <= 0:
        return 0
    win_loss_ratio = avg_win / avg_loss
    kelly_pct = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
    kelly_pct = max(0.0, kelly_pct) * kelly_fraction
    kelly_pct = min(kelly_pct, max_pct)
    dollar_amount = capital * kelly_pct
    return int(dollar_amount / price)


# ─── Stop levels ──────────────────────────────────────────────────────────────

@dataclass
class StopLevels:
    """Computed stop and target prices for an open position."""
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop_init: float


def compute_stops(
    entry_price: float,
    direction: int,          # +1 long, -1 short
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    atr: float | None = None,
    atr_stop_mult: float = 2.0,
    atr_tp_mult: float = 3.0,
) -> StopLevels:
    """
    Compute stop-loss and take-profit price levels.

    If ATR is provided, uses ATR-based stops (more adaptive).
    Otherwise uses fixed percentage stops.

    Args:
        entry_price:     Fill price.
        direction:       +1 for long, -1 for short.
        stop_loss_pct:   Fixed stop distance as fraction of price.
        take_profit_pct: Fixed target distance as fraction of price.
        atr:             Current ATR value (optional; overrides fixed pct if set).
        atr_stop_mult:   Stop = entry ± atr * multiplier.
        atr_tp_mult:     Target = entry ± atr * multiplier.

    Returns:
        StopLevels with stop_loss and take_profit prices.
    """
    if atr is not None and atr > 0:
        sl_dist = atr * atr_stop_mult
        tp_dist = atr * atr_tp_mult
    else:
        sl_dist = entry_price * stop_loss_pct
        tp_dist = entry_price * take_profit_pct

    if direction == 1:  # long
        stop = entry_price - sl_dist
        target = entry_price + tp_dist
    else:  # short
        stop = entry_price + sl_dist
        target = entry_price - tp_dist

    return StopLevels(
        entry_price=entry_price,
        stop_loss=round(stop, 4),
        take_profit=round(target, 4),
        trailing_stop_init=round(stop, 4),
    )


def update_trailing_stop(
    current_price: float,
    direction: int,
    trailing_stop: float,
    trail_pct: float = 0.01,
    atr: float | None = None,
    atr_mult: float = 2.0,
) -> float:
    """
    Ratchet trailing stop upward (long) or downward (short).

    Args:
        current_price:  Latest bar close.
        direction:      +1 long / -1 short.
        trailing_stop:  Current trailing stop level.
        trail_pct:      Trail distance as fraction of current price.
        atr:            ATR for dynamic trailing (overrides trail_pct).
        atr_mult:       ATR multiplier for trail distance.

    Returns:
        New trailing stop level (never worse than previous).
    """
    trail_dist = (atr * atr_mult) if (atr and atr > 0) else (current_price * trail_pct)

    if direction == 1:
        new_stop = current_price - trail_dist
        return max(trailing_stop, new_stop)
    else:
        new_stop = current_price + trail_dist
        return min(trailing_stop, new_stop)


def is_stop_hit(
    current_price: float,
    direction: int,
    stop_loss: float,
    take_profit: float,
) -> tuple[bool, str]:
    """
    Check if current price has hit stop-loss or take-profit.

    Returns:
        (hit: bool, reason: str)  reason is 'stop_loss', 'take_profit', or ''
    """
    if direction == 1:
        if current_price <= stop_loss:
            return True, "stop_loss"
        if current_price >= take_profit:
            return True, "take_profit"
    elif direction == -1:
        if current_price >= stop_loss:
            return True, "stop_loss"
        if current_price <= take_profit:
            return True, "take_profit"
    return False, ""


# ─── Portfolio-level risk ─────────────────────────────────────────────────────

@dataclass
class PortfolioRiskState:
    """Tracks portfolio-level risk metrics in real-time."""

    peak_equity: float
    current_equity: float
    max_drawdown_limit: float = 0.15
    max_portfolio_heat: float = 0.20   # max total open risk %
    halted: bool = False

    @property
    def current_drawdown(self) -> float:
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def update(self, new_equity: float) -> None:
        self.current_equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

    def check_halt(self) -> bool:
        """
        Return True (and set halted=True) if portfolio drawdown
        exceeds the configured limit. Once halted, stays halted.
        """
        if self.halted:
            return True
        if self.current_drawdown >= self.max_drawdown_limit:
            logger.warning(
                f"[RiskManager] HALT: drawdown={self.current_drawdown:.2%} "
                f">= limit={self.max_drawdown_limit:.2%}"
            )
            self.halted = True
        return self.halted


def apply_risk_filter(
    signals_df: pd.DataFrame,
    equity_curve: pd.Series,
    max_drawdown_limit: float = 0.15,
) -> pd.DataFrame:
    """
    Vectorized risk filter: zero out signals when rolling drawdown
    exceeds the maximum allowed level.

    Args:
        signals_df:         DataFrame with Position column.
        equity_curve:       Equity curve aligned to signals_df index.
        max_drawdown_limit: Halt threshold (0.15 = 15%).

    Returns:
        signals_df with Position set to 0 during drawdown breach periods.
    """
    df = signals_df.copy()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max.replace(0, np.nan)
    # Once the limit is breached, stay flat. Resuming on the same path
    # would use the recovery that only exists because trading continued.
    breached = (drawdown <= -max_drawdown_limit).fillna(False)
    halted = breached.astype(int).cummax().astype(bool)
    df.loc[halted, "Position"] = 0
    n_halted = int(halted.sum())
    if n_halted:
        logger.info(f"[RiskManager] {n_halted} bars halted due to drawdown limit.")
    return df
