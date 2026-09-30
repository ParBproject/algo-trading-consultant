"""
src/executor.py
===============
Broker API wrappers for paper and live order execution.

Supported brokers:
    - Paper (in-memory simulation)
    - Alpaca (stocks, paper + live)
    - CCXT  (crypto exchanges)

All executors implement the same interface::

    executor.submit_order(symbol, direction, qty, order_type)
    executor.get_positions()
    executor.get_equity()

The default route is the in-memory paper broker. Alpaca and CCXT read
API keys from the environment and stay on paper or sandbox endpoints
unless ``allow_live=True``.
"""
from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd
from loguru import logger


# ─── Order / Position models ──────────────────────────────────────────────────

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop"]


@dataclass
class Order:
    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    filled_price: Optional[float] = None
    status: str = "pending"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    symbol: str
    qty: int          # positive = long, negative = short
    avg_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.qty * (self.current_price - self.avg_price)


# ─── Base executor ────────────────────────────────────────────────────────────

class BaseExecutor(ABC):

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        order_type: OrderType = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Place an order and return Order record."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Return dict of open positions keyed by symbol."""

    @abstractmethod
    def get_equity(self) -> float:
        """Return current total portfolio equity."""

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close an open position by submitting opposite-side order."""
        positions = self.get_positions()
        pos = positions.get(symbol)
        if pos is None:
            logger.warning(f"No open position in {symbol}")
            return None
        side: OrderSide = "sell" if pos.qty > 0 else "buy"
        return self.submit_order(symbol, side, abs(pos.qty))


# ─── Paper executor ───────────────────────────────────────────────────────────

class PaperExecutor(BaseExecutor):
    """
    In-memory paper trading simulator.

    Simulates market fills with slippage and commission.
    Designed for strategy testing without connecting to a broker.

    Args:
        initial_capital: Starting cash balance.
        commission:      Fraction of trade value charged per order.
        slippage:        Fraction of price added to fill (buys) or subtracted (sells).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._prices: dict[str, float] = {}  # symbol -> latest price

    def set_price(self, symbol: str, price: float) -> None:
        """Update the latest price for a symbol (call per bar in simulation)."""
        self._prices[symbol] = price
        if symbol in self._positions:
            self._positions[symbol].current_price = price

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        order_type: OrderType = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        order = Order(symbol=symbol, side=side, qty=qty, order_type=order_type,
                      limit_price=limit_price, stop_price=stop_price)

        if qty <= 0:
            logger.warning(f"Order qty must be positive; got {qty}.")
            order.status = "rejected"
            self._orders.append(order)
            return order

        price = self._prices.get(symbol)
        if price is None or price <= 0:
            logger.warning(f"No price available for {symbol}. Order rejected.")
            order.status = "rejected"
            self._orders.append(order)
            return order

        # Buys pay above the quote and sells receive below it.
        fill_price = price * (1 + self.slippage) if side == "buy" else price * (1 - self.slippage)
        trade_value = fill_price * qty
        commission_cost = trade_value * self.commission
        held = self._positions[symbol].qty if symbol in self._positions else 0
        new_qty = held + qty if side == "buy" else held - qty

        # Opening or adding risk cannot exceed current equity. Reducing a
        # position (including a full close) is always allowed.
        increases = abs(new_qty) > abs(held)
        if increases and trade_value > self.get_equity() + 1e-6:
            logger.warning(
                f"Order notional {trade_value:.2f} exceeds equity {self.get_equity():.2f}"
            )
            order.status = "rejected"
            self._orders.append(order)
            return order

        if side == "buy":
            total_cost = trade_value + commission_cost
            if total_cost > self.cash + 1e-6:
                logger.warning(f"Insufficient cash: need {total_cost:.2f}, have {self.cash:.2f}")
                order.status = "rejected"
                self._orders.append(order)
                return order
            self.cash -= total_cost
        else:
            self.cash += trade_value - commission_cost

        self._set_qty(symbol, new_qty, fill_price, price)
        order.filled_price = round(fill_price, 4)
        order.status = "filled"
        self._orders.append(order)
        logger.info(f"[Paper] {side.upper()} {qty} {symbol} @ {fill_price:.4f} "
                    f"(commission={commission_cost:.2f})")
        return order

    def _set_qty(self, symbol: str, new_qty: int, fill_price: float, mark_price: float) -> None:
        """Store a signed position. Flat books drop the symbol.

        Average price is the fill. The mark stays at the quoted mid so
        slippage lives in cash rather than being marked in twice.
        """
        if new_qty == 0:
            self._positions.pop(symbol, None)
            return
        pos = self._positions.get(symbol)
        if pos is None or pos.qty == 0 or (pos.qty > 0) != (new_qty > 0):
            # New position, or a flip: the open remainder was filled now.
            self._positions[symbol] = Position(symbol, new_qty, fill_price, mark_price)
            return
        if abs(new_qty) > abs(pos.qty):
            added = abs(new_qty) - abs(pos.qty)
            pos.avg_price = (pos.avg_price * abs(pos.qty) + fill_price * added) / abs(new_qty)
        pos.qty = new_qty
        pos.current_price = mark_price

    def get_positions(self) -> dict[str, Position]:
        return self._positions.copy()

    def get_equity(self) -> float:
        mktval = sum(p.market_value for p in self._positions.values())
        return self.cash + mktval

    def get_trade_log(self) -> pd.DataFrame:
        """Return all executed orders as a DataFrame."""
        return pd.DataFrame([vars(o) for o in self._orders if o.status == "filled"])


# ─── Alpaca executor ──────────────────────────────────────────────────────────

class AlpacaExecutor(BaseExecutor):
    """
    Alpaca Markets order executor.

    Credentials are read from ALPACA_API_KEY and ALPACA_SECRET_KEY.
    They are not accepted as arguments, so a config file cannot carry them.

    The live trading endpoint stays off unless ``allow_live`` is set.
    ``paper=True`` (the default) uses Alpaca's paper endpoint.

    Args:
        paper:      Use the paper trading endpoint.
        allow_live: Required when ``paper`` is False.
    """

    def __init__(
        self,
        paper: bool = True,
        allow_live: bool = False,
    ):
        if not paper and not allow_live:
            raise RuntimeError(
                "Alpaca live trading is disabled. The default route is paper. "
                "Pass allow_live=True to use the live endpoint."
            )

        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret:
            raise RuntimeError(
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment. "
                "Keys are not read from config files or constructor arguments."
            )

        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError("Install alpaca-py: pip install alpaca-py")

        self._paper = paper
        self._client = TradingClient(api_key, secret, paper=paper)
        logger.info(f"Alpaca executor initialised ({'paper' if paper else 'LIVE'})")

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        order_type: OrderType = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide as AS, TimeInForce

        alpaca_side = AS.BUY if side == "buy" else AS.SELL
        tif = TimeInForce.DAY

        if order_type == "market":
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=alpaca_side, time_in_force=tif)
        elif order_type == "limit":
            req = LimitOrderRequest(symbol=symbol, qty=qty, side=alpaca_side,
                                    limit_price=limit_price, time_in_force=tif)
        else:
            from alpaca.trading.requests import StopOrderRequest
            req = StopOrderRequest(symbol=symbol, qty=qty, side=alpaca_side,
                                   stop_price=stop_price, time_in_force=tif)

        resp = self._client.submit_order(req)
        order = Order(symbol=symbol, side=side, qty=qty, order_type=order_type,
                      order_id=str(resp.id), status=str(resp.status))
        logger.info(f"[Alpaca] Order submitted: {order.order_id} – {side} {qty} {symbol}")
        return order

    def get_positions(self) -> dict[str, Position]:
        positions = self._client.get_all_positions()
        return {
            p.symbol: Position(
                symbol=p.symbol,
                qty=int(p.qty),
                avg_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
            )
            for p in positions
        }

    def get_equity(self) -> float:
        account = self._client.get_account()
        return float(account.portfolio_value)


# ─── CCXT crypto executor ─────────────────────────────────────────────────────

class CCXTExecutor(BaseExecutor):
    """
    CCXT-based crypto exchange executor.

    Supports any CCXT-compatible exchange (Binance, Kraken, Coinbase, etc.).
    Credentials are read from CCXT_API_KEY and CCXT_SECRET only.

    Sandbox mode is the default. Turning it off requires ``allow_live``.

    Args:
        exchange_id: CCXT exchange id string.
        sandbox:     Use the exchange sandbox or testnet when available.
        allow_live:  Required when ``sandbox`` is False.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        sandbox: bool = True,
        allow_live: bool = False,
    ):
        if not sandbox and not allow_live:
            raise RuntimeError(
                "CCXT live trading is disabled. The default route is the sandbox. "
                "Pass allow_live=True to submit orders outside sandbox mode."
            )

        api_key = os.environ.get("CCXT_API_KEY", "")
        secret = os.environ.get("CCXT_SECRET", "")
        if not api_key or not secret:
            raise RuntimeError(
                "Set CCXT_API_KEY and CCXT_SECRET in the environment. "
                "Keys are not read from config files or constructor arguments."
            )

        try:
            import ccxt
        except ImportError:
            raise ImportError("Install ccxt: pip install ccxt")

        exchange_class = getattr(ccxt, exchange_id)
        self._exchange = exchange_class({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
        })
        if sandbox:
            self._exchange.set_sandbox_mode(True)
        logger.info(f"CCXT executor: {exchange_id} ({'sandbox' if sandbox else 'LIVE'})")

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        order_type: OrderType = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        price_arg = limit_price if order_type == "limit" else None
        resp = self._exchange.create_order(symbol, order_type, side, qty, price_arg)
        order = Order(symbol=symbol, side=side, qty=qty, order_type=order_type,
                      order_id=str(resp["id"]),
                      filled_price=resp.get("price"),
                      status=resp.get("status", "open"))
        logger.info(f"[CCXT] Order: {order.order_id} {side} {qty} {symbol}")
        return order

    def get_positions(self) -> dict[str, Position]:
        balance = self._exchange.fetch_balance()
        positions = {}
        for asset, amt in balance["total"].items():
            if amt and float(amt) > 0:
                ticker = self._exchange.fetch_ticker(f"{asset}/USDT")
                price = ticker["last"]
                positions[asset] = Position(
                    symbol=asset,
                    qty=int(float(amt)),
                    avg_price=price,
                    current_price=price,
                )
        return positions

    def get_equity(self) -> float:
        balance = self._exchange.fetch_balance()
        return float(balance.get("USDT", {}).get("total", 0.0))


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_executor(broker: str, **kwargs) -> BaseExecutor:
    """
    Factory: return configured executor by broker name.

    Args:
        broker: 'paper', 'alpaca', or 'ccxt'.
        **kwargs: Passed to executor constructor.
    """
    brokers = {"paper": PaperExecutor, "alpaca": AlpacaExecutor, "ccxt": CCXTExecutor}
    cls = brokers.get(broker)
    if not cls:
        raise ValueError(f"Unknown broker '{broker}'. Options: {list(brokers)}")
    return cls(**kwargs)
