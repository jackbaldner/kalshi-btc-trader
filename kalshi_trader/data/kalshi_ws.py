"""Kalshi WebSocket client for real-time orderbook updates."""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path

import websockets

logger = logging.getLogger(__name__)

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
SIGN_PATH = "/trade-api/ws/v2"


class KalshiWebSocket:
    """Maintains a live local orderbook via Kalshi's WebSocket feed.

    Subscribes to orderbook_delta channel for a given market ticker.
    Receives an initial snapshot then applies incremental deltas.
    """

    def __init__(self, cfg: dict):
        self.api_key_id = cfg["kalshi"]["api_key_id"]
        self._private_key = None

        pk_path = cfg["kalshi"]["private_key_path"]
        if pk_path and Path(pk_path).exists():
            from cryptography.hazmat.primitives import serialization
            with open(pk_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)

        self._ws = None
        self._running = False
        self._market_ticker: str | None = None

        # Local orderbook: {side: {price: quantity}}
        self._book: dict[str, dict[int, int]] = {"yes": {}, "no": {}}
        self._book_ready = False
        self._last_update_ts: float = 0
        self._reconnect_delay = 1

    @property
    def orderbook(self) -> dict:
        """Return orderbook in same format as REST API: {yes: [[price, qty], ...], no: [...]}."""
        return {
            "yes": sorted(self._book["yes"].items()),
            "no": sorted(self._book["no"].items()),
        }

    @property
    def is_ready(self) -> bool:
        return self._book_ready

    @property
    def age_seconds(self) -> float:
        """Seconds since last orderbook update."""
        if self._last_update_ts == 0:
            return float("inf")
        return time.time() - self._last_update_ts

    def _sign(self, timestamp_ms: int) -> str:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        message = f"{timestamp_ms}GET{SIGN_PATH}"
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self) -> dict:
        ts = int(time.time() * 1000)
        sig = self._sign(ts)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
        }

    async def connect(self, market_ticker: str):
        """Start WebSocket connection and subscribe to orderbook for a market."""
        self._market_ticker = market_ticker
        self._running = True
        self._reconnect_delay = 1

        while self._running:
            try:
                await self._run_connection()
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if not self._running:
                    break
                logger.warning(
                    f"WebSocket disconnected: {e}. Reconnecting in {self._reconnect_delay}s..."
                )
                self._book_ready = False
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"WebSocket error: {e}. Reconnecting in {self._reconnect_delay}s...")
                self._book_ready = False
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)

    async def _run_connection(self):
        headers = self._auth_headers()
        async with websockets.connect(WS_URL, additional_headers=headers) as ws:
            self._ws = ws
            logger.info(f"WebSocket connected, subscribing to orderbook for {self._market_ticker}")

            # Subscribe to orderbook delta channel
            sub_msg = {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": [self._market_ticker],
                },
            }
            await ws.send(json.dumps(sub_msg))
            self._reconnect_delay = 1  # reset on successful connect

            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                    self._handle_message(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from WebSocket: {raw[:100]}")

    def _handle_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "orderbook_snapshot":
            self._handle_snapshot(msg.get("msg", {}))
        elif msg_type == "orderbook_delta":
            self._handle_delta(msg.get("msg", {}))
        elif msg_type == "error":
            logger.error(f"WebSocket error: {msg.get('msg', {})}")
        elif msg_type == "subscribed":
            logger.info(f"Subscribed to channel: {msg}")

    def _handle_snapshot(self, data: dict):
        """Replace local book with full snapshot."""
        self._book = {"yes": {}, "no": {}}

        for side in ("yes", "no"):
            levels = data.get(side, [])
            for price, qty in levels:
                if qty > 0:
                    self._book[side][price] = qty

        self._book_ready = True
        self._last_update_ts = time.time()

        yes_levels = len(self._book["yes"])
        no_levels = len(self._book["no"])
        logger.info(f"Orderbook snapshot: {yes_levels} YES levels, {no_levels} NO levels")

    def _handle_delta(self, data: dict):
        """Apply incremental update to local book."""
        side = data.get("side")
        price = data.get("price")
        delta = data.get("delta", 0)

        if side not in ("yes", "no") or price is None:
            return

        current = self._book[side].get(price, 0)
        new_qty = current + delta

        if new_qty <= 0:
            self._book[side].pop(price, None)
        else:
            self._book[side][price] = new_qty

        self._last_update_ts = time.time()

    async def switch_market(self, new_ticker: str):
        """Unsubscribe from current market and subscribe to a new one."""
        if new_ticker == self._market_ticker:
            return

        old_ticker = self._market_ticker
        self._market_ticker = new_ticker
        self._book = {"yes": {}, "no": {}}
        self._book_ready = False

        if self._is_ws_open():
            # Unsubscribe from old
            if old_ticker:
                unsub = {
                    "id": 2,
                    "cmd": "unsubscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_tickers": [old_ticker],
                    },
                }
                await self._ws.send(json.dumps(unsub))

            # Subscribe to new
            sub = {
                "id": 3,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": [new_ticker],
                },
            }
            await self._ws.send(json.dumps(sub))
            logger.info(f"Switched orderbook subscription: {old_ticker} -> {new_ticker}")

    def _is_ws_open(self) -> bool:
        """Check if WebSocket is open, compatible across websockets versions."""
        if not self._ws:
            return False
        try:
            return not self._ws.closed
        except AttributeError:
            # Newer websockets versions use close_code
            return self._ws.close_code is None

    async def close(self):
        """Gracefully shut down."""
        self._running = False
        try:
            if self._is_ws_open():
                await self._ws.close()
        except Exception:
            pass  # never crash on shutdown
