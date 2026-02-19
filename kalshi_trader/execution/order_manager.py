"""Order lifecycle management: place, cancel, track orders."""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order placement, tracking, and auto-cancellation before expiry."""

    def __init__(self, kalshi_client, cfg: dict):
        self.client = kalshi_client
        self.cancel_before_expiry = cfg["risk"]["cancel_before_expiry_sec"]
        self._open_orders: dict[str, dict] = {}  # order_id -> order info
        self._cancel_tasks: dict[str, asyncio.Task] = {}

    def place_order(self, ticker: str, side: str, price_cents: int, count: int,
                    expiry_time_ms: int | None = None) -> dict | None:
        """Place an order and schedule auto-cancel before expiry.

        Args:
            ticker: Kalshi market ticker
            side: 'yes' or 'no'
            price_cents: Price in cents (1-99)
            count: Number of contracts
            expiry_time_ms: Market expiry timestamp in ms (for auto-cancel)

        Returns:
            Order response dict or None on failure
        """
        try:
            result = self.client.create_order(ticker, side, price_cents, count)
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

        if "error" in result:
            logger.warning(f"Order rejected: {result}")
            return None

        order_id = result.get("order_id") or result.get("order", {}).get("order_id")
        if order_id:
            self._open_orders[order_id] = {
                "order_id": order_id,
                "ticker": ticker,
                "side": side,
                "price": price_cents,
                "count": count,
                "status": result.get("status", "open"),
                "placed_at": int(time.time() * 1000),
            }
            logger.info(f"Placed order {order_id}: {side} {count}x {ticker} @ {price_cents}c")

        return result

    def schedule_auto_cancel(self, order_id: str, expiry_time_ms: int, loop: asyncio.AbstractEventLoop):
        """Schedule auto-cancel of an order before market expiry."""
        now_ms = int(time.time() * 1000)
        cancel_at_ms = expiry_time_ms - (self.cancel_before_expiry * 1000)
        delay_sec = max(0, (cancel_at_ms - now_ms) / 1000)

        async def _cancel_later():
            await asyncio.sleep(delay_sec)
            self.cancel_order(order_id)

        task = loop.create_task(_cancel_later())
        self._cancel_tasks[order_id] = task

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id not in self._open_orders:
            return False

        try:
            self.client.cancel_order(order_id)
            self._open_orders[order_id]["status"] = "canceled"
            logger.info(f"Canceled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all(self):
        """Cancel all open orders."""
        for order_id in list(self._open_orders.keys()):
            if self._open_orders[order_id].get("status") in ("open", "pending"):
                self.cancel_order(order_id)

    def get_open_orders(self) -> list[dict]:
        return [o for o in self._open_orders.values() if o.get("status") in ("open", "pending")]

    def mark_filled(self, order_id: str):
        if order_id in self._open_orders:
            self._open_orders[order_id]["status"] = "filled"

    def cleanup(self):
        """Cancel all pending auto-cancel tasks."""
        for task in self._cancel_tasks.values():
            task.cancel()
        self._cancel_tasks.clear()
