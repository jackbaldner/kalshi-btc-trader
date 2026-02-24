"""Order lifecycle management: place, cancel, track orders."""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Max time (seconds) before orphaned order state is cleaned up (safety net only)
ORDER_TIMEOUT_SEC = 300


class OrderManager:
    """Manages order placement, tracking, and auto-cancellation before expiry."""

    def __init__(self, kalshi_client, cfg: dict):
        self.client = kalshi_client
        self.cancel_before_expiry = cfg["risk"]["cancel_before_expiry_sec"]
        self._open_orders: dict[str, dict] = {}  # order_id -> order info
        self._cancel_tasks: dict[str, asyncio.Task] = {}

    def place_order(self, ticker: str, side: str, price_cents: int, count: int,
                    expiry_time_ms: int | None = None, max_retries: int = 2) -> dict | None:
        """Place an order with retry logic for transient failures.

        Args:
            ticker: Kalshi market ticker
            side: 'yes' or 'no'
            price_cents: Price in cents (1-99)
            count: Number of contracts
            expiry_time_ms: Market expiry timestamp in ms (for auto-cancel)
            max_retries: Number of retries on transient errors

        Returns:
            Order response dict or None on failure
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = self.client.create_order(ticker, side, price_cents, count)
            except Exception as e:
                last_error = e
                logger.warning(f"Order attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)  # backoff
                continue

            if "error" in result:
                error_msg = result.get("error", "")
                # Don't retry on permanent rejections
                if any(msg in str(error_msg).lower() for msg in
                       ["insufficient", "market closed", "invalid", "not found"]):
                    logger.warning(f"Order permanently rejected: {result}")
                    return None
                # Retry on transient errors
                logger.warning(f"Order attempt {attempt}/{max_retries} rejected: {result}")
                if attempt < max_retries:
                    time.sleep(0.5 * attempt)
                continue

            # Success — track the order
            order_id = result.get("order_id") or result.get("order", {}).get("order_id")
            filled_count = result.get("count", count)
            if order_id:
                self._open_orders[order_id] = {
                    "order_id": order_id,
                    "ticker": ticker,
                    "side": side,
                    "price": price_cents,
                    "requested_count": count,
                    "filled_count": filled_count,
                    "status": result.get("status", "open"),
                    "placed_at": int(time.time() * 1000),
                }
                if filled_count < count:
                    logger.info(
                        f"Partial fill: {order_id} {side} {filled_count}/{count}x {ticker} @ {price_cents}c"
                    )
                else:
                    logger.info(f"Placed order {order_id}: {side} {count}x {ticker} @ {price_cents}c")

            return result

        logger.error(f"Order failed after {max_retries} attempts: {last_error}")
        return None

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

    def cancel_stale_orders(self):
        """Cancel orders that have been open longer than ORDER_TIMEOUT_SEC."""
        now_ms = int(time.time() * 1000)
        timeout_ms = ORDER_TIMEOUT_SEC * 1000
        for order_id, info in list(self._open_orders.items()):
            if info.get("status") in ("open", "pending"):
                age_ms = now_ms - info.get("placed_at", now_ms)
                if age_ms > timeout_ms:
                    logger.warning(
                        f"Order {order_id} stale ({age_ms / 1000:.0f}s old), canceling"
                    )
                    self.cancel_order(order_id)

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
