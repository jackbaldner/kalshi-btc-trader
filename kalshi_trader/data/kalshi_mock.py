"""Mock Kalshi client for paper trading. Implements same interface as KalshiClient."""

import logging
import time
import uuid

logger = logging.getLogger(__name__)


class KalshiMock:
    """Simulates Kalshi API for paper trading. Fills against real orderbook depth."""

    def __init__(self, cfg: dict):
        self.event_ticker = cfg["kalshi"]["event_ticker"]
        self._balance = 10000.0  # starting paper balance
        self._positions: dict[str, dict] = {}  # ticker -> position
        self._orders: dict[str, dict] = {}  # order_id -> order
        self._fills: list[dict] = []
        self._simulated_markets: list[dict] = []
        self._live_orderbook: dict | None = None  # set by TradingSystem

    def set_live_orderbook(self, orderbook: dict):
        """Receive the real orderbook from the Kalshi reader for realistic fills."""
        self._live_orderbook = orderbook

    def set_simulated_markets(self, markets: list[dict]):
        """Set mock market data for testing."""
        self._simulated_markets = markets

    def get_markets(self, status: str = "open") -> list[dict]:
        if self._simulated_markets:
            return self._simulated_markets
        # Generate a synthetic market
        now_ms = int(time.time() * 1000)
        return [{
            "ticker": f"{self.event_ticker}-T1234",
            "event_ticker": self.event_ticker,
            "status": "open",
            "yes_bid": 50,
            "yes_ask": 52,
            "no_bid": 48,
            "no_ask": 50,
            "close_time": now_ms + 15 * 60 * 1000,
        }]

    def get_market(self, ticker: str) -> dict:
        for m in self.get_markets():
            if m["ticker"] == ticker:
                return m
        return {}

    def get_orderbook(self, ticker: str) -> dict:
        """Return the live orderbook if available, else synthetic."""
        if self._live_orderbook:
            return self._live_orderbook
        mid = 50
        return {
            "yes": [[mid - 2, 10], [mid - 1, 20]],
            "no": [[100 - mid - 2, 10], [100 - mid - 1, 20]],
        }

    def create_order(self, ticker: str, side: str, yes_price: int, count: int,
                     order_type: str = "limit") -> dict:
        """Simulate order fill by walking the real orderbook.

        Only fills contracts that are actually available at or below the
        requested price. Returns partial fill if insufficient depth.
        """
        order_id = str(uuid.uuid4())[:8]

        # Kalshi orderbook: both "yes" and "no" arrays are BIDS (resting buy orders).
        # To BUY YES: match against NO bids. A NO bid at Xc = YES ask at (100-X)c.
        # To BUY NO: match against YES bids. A YES bid at Xc = NO ask at (100-X)c.
        book = self._live_orderbook or {}
        opposite_side = "no" if side == "yes" else "yes"
        opposite_bids = book.get(opposite_side, [])

        filled_count = 0
        total_cost = 0.0

        if opposite_bids:
            # Convert opposite bids to our ask prices and sort cheapest first
            # NO bid at X → YES ask at (100-X). We want the highest NO bids first
            # (they give us the cheapest YES price).
            ask_levels = [(100 - bid_price, qty) for bid_price, qty in opposite_bids]
            ask_levels.sort(key=lambda x: x[0])  # cheapest ask first

            for ask_price, level_qty in ask_levels:
                if ask_price > yes_price:
                    break  # beyond our limit price
                can_fill = min(level_qty, count - filled_count)
                filled_count += can_fill
                total_cost += ask_price * can_fill / 100  # cents to dollars
                if filled_count >= count:
                    break
        else:
            # No orderbook available — fall back to limit price with 1c slippage
            fill_price = yes_price + 1 if side == "yes" else yes_price - 1
            fill_price = max(1, min(99, fill_price))
            filled_count = count
            total_cost = fill_price * count / 100

        if filled_count == 0:
            logger.warning(
                f"[PAPER] No fills available: {side} {count}x {ticker} @ {yes_price}c "
                f"(book has no offers at or below limit)"
            )
            return {"error": "no_liquidity"}

        avg_fill_price = (total_cost / filled_count) * 100 if filled_count > 0 else 0  # back to cents

        if total_cost > self._balance:
            logger.warning(f"Insufficient balance: need ${total_cost:.2f}, have ${self._balance:.2f}")
            return {"error": "insufficient_balance"}

        self._balance -= total_cost

        if filled_count < count:
            logger.info(
                f"[PAPER] Partial fill: {side} {filled_count}/{count}x {ticker} "
                f"@ avg {avg_fill_price:.1f}c (cost ${total_cost:.2f}) — "
                f"insufficient depth for full order"
            )
        else:
            logger.info(
                f"[PAPER] Filled {side} {filled_count}x {ticker} "
                f"@ avg {avg_fill_price:.1f}c (cost ${total_cost:.2f})"
            )

        order = {
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "yes_price": round(avg_fill_price),
            "count": filled_count,
            "requested_count": count,
            "status": "filled" if filled_count == count else "partial",
            "created_time": int(time.time() * 1000),
        }
        self._orders[order_id] = order

        # Update position
        if ticker not in self._positions:
            self._positions[ticker] = {
                "ticker": ticker,
                "yes_count": 0,
                "no_count": 0,
                "avg_yes_price": 0,
                "avg_no_price": 0,
            }
        pos = self._positions[ticker]
        if side == "yes":
            prev_cost = pos["avg_yes_price"] * pos["yes_count"]
            pos["yes_count"] += filled_count
            pos["avg_yes_price"] = (prev_cost + total_cost * 100) / pos["yes_count"] if pos["yes_count"] > 0 else 0
        else:
            prev_cost = pos["avg_no_price"] * pos["no_count"]
            pos["no_count"] += filled_count
            pos["avg_no_price"] = (prev_cost + total_cost * 100) / pos["no_count"] if pos["no_count"] > 0 else 0

        fill = {
            "trade_id": order_id,
            "ticker": ticker,
            "side": side,
            "yes_price": round(avg_fill_price),
            "count": filled_count,
            "created_time": order["created_time"],
        }
        self._fills.append(fill)

        return order

    def cancel_order(self, order_id: str) -> dict:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "canceled"
            return {"order_id": order_id, "status": "canceled"}
        return {"error": "order_not_found"}

    def get_positions(self) -> list[dict]:
        return [p for p in self._positions.values() if p["yes_count"] > 0 or p["no_count"] > 0]

    def get_fills(self, ticker: str | None = None, limit: int = 100) -> list[dict]:
        fills = self._fills
        if ticker:
            fills = [f for f in fills if f["ticker"] == ticker]
        return fills[-limit:]

    def get_balance(self) -> float:
        return self._balance

    def resolve_market(self, ticker: str, result: str):
        """Resolve a paper market. result='yes' or 'no'. Pays out $1 per winning contract."""
        pos = self._positions.get(ticker)
        if not pos:
            return 0.0

        pnl = 0.0
        if result == "yes" and pos["yes_count"] > 0:
            payout = pos["yes_count"]  # $1 per contract
            pnl = payout - (pos["avg_yes_price"] * pos["yes_count"] / 100)
            self._balance += payout
        elif result == "no" and pos["no_count"] > 0:
            payout = pos["no_count"]
            pnl = payout - (pos["avg_no_price"] * pos["no_count"] / 100)
            self._balance += payout
        else:
            # Losing side — cost was already deducted
            if result == "yes":
                pnl = -(pos["no_count"] * pos["avg_no_price"] / 100)
            else:
                pnl = -(pos["yes_count"] * pos["avg_yes_price"] / 100)

        # Clear position
        self._positions[ticker] = {
            "ticker": ticker, "yes_count": 0, "no_count": 0,
            "avg_yes_price": 0, "avg_no_price": 0,
        }
        logger.info(f"[PAPER] Resolved {ticker} -> {result}, PnL: ${pnl:.2f}")
        return pnl

    def close(self):
        pass
