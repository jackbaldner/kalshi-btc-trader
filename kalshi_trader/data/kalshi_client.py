"""Kalshi REST API client with RSA-PSS authentication."""

import base64
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)


class KalshiClient:
    """Thin wrapper around Kalshi REST API v2."""

    def __init__(self, cfg: dict):
        self.base_url = cfg["kalshi"]["base_url"].rstrip("/")
        self.api_key_id = cfg["kalshi"]["api_key_id"]
        self.event_ticker = cfg["kalshi"]["event_ticker"]
        self.depth = cfg["kalshi"]["orderbook_depth"]
        self._private_key = None

        # Extract the path prefix from base_url for signing
        # e.g. "https://api.elections.kalshi.com/trade-api/v2" -> "/trade-api/v2"
        from urllib.parse import urlparse
        parsed = urlparse(self.base_url)
        self._path_prefix = parsed.path.rstrip("/")

        pk_path = cfg["kalshi"]["private_key_path"]
        if pk_path and Path(pk_path).exists():
            with open(pk_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
            logger.info("Loaded Kalshi private key")
        else:
            logger.warning(f"No private key found at: {pk_path}")

        self._client = httpx.Client(timeout=10.0)

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Create RSA-PSS signature for Kalshi API auth.

        Message format: "{timestamp}{METHOD}{path_without_query_params}"
        Salt length must be DIGEST_LENGTH per Kalshi docs.
        """
        if self._private_key is None:
            raise RuntimeError("No private key loaded — cannot sign requests")

        # Strip query params from path for signing
        path_for_signing = path.split("?")[0]
        message = f"{timestamp_ms}{method}{path_for_signing}"
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path: str) -> dict:
        ts = int(time.time() * 1000)
        sig = self._sign_request(method, path, ts)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        # Sign with full path including /trade-api/v2 prefix
        full_path = f"{self._path_prefix}{path}"
        headers = self._headers(method.upper(), full_path)
        resp = self._client.request(method, url, headers=headers, **kwargs)
        resp.raise_for_status()
        return resp.json()

    # -- Public / Market Data --

    def get_events(self, status: str = "open") -> list[dict]:
        """List open events for the KXBTC series."""
        path = f"/events?status={status}&series_ticker={self.event_ticker}"
        data = self._request("GET", path)
        return data.get("events", [])

    def get_nearest_event_ticker(self, max_hours: float = 24) -> tuple[str | None, str | None]:
        """Find the nearest expiring KXBTC event by checking market close times.

        Args:
            max_hours: Only consider events closing within this many hours.

        Returns:
            (event_ticker, close_time_iso) or (None, None)
        """
        from datetime import datetime, timezone
        events = self.get_events()
        if not events:
            return None, None

        now = datetime.now(timezone.utc)
        best_ticker = None
        best_close = None

        for event in events:
            eticker = event.get("event_ticker", "")
            # Get one market to check close_time
            markets = self.get_markets(event_ticker=eticker)
            if not markets:
                continue
            close_str = markets[0].get("close_time", "")
            if not close_str:
                continue
            try:
                close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            # Skip already closed or too far out
            hours_remaining = (close_dt - now).total_seconds() / 3600
            if hours_remaining <= 0 or hours_remaining > max_hours:
                continue
            if best_close is None or close_dt < best_close:
                best_close = close_dt
                best_ticker = eticker

        if best_ticker:
            logger.info(
                f"Selected event {best_ticker} closing at {best_close.isoformat()} "
                f"({(best_close - now).total_seconds() / 3600:.1f}h remaining)"
            )
            return best_ticker, best_close.isoformat() if best_close else None

        logger.warning(f"No KXBTC events closing within {max_hours}h")
        return None, None

    def get_markets(self, status: str = "open", event_ticker: str | None = None) -> list[dict]:
        """List markets for a BTC price range event."""
        ticker = event_ticker or self.event_ticker
        path = f"/markets?event_ticker={ticker}&status={status}"
        data = self._request("GET", path)
        return data.get("markets", [])

    def get_market(self, ticker: str) -> dict:
        """Get single market details."""
        path = f"/markets/{ticker}"
        return self._request("GET", path).get("market", {})

    def get_orderbook(self, ticker: str) -> dict:
        """Get orderbook for a market."""
        path = f"/markets/{ticker}/orderbook?depth={self.depth}"
        return self._request("GET", path).get("orderbook", {})

    # -- Trading --

    def create_order(self, ticker: str, side: str, yes_price: int, count: int,
                     order_type: str = "limit") -> dict:
        """Place an order. side='yes'|'no', yes_price in cents (1-99), count=number of contracts."""
        path = "/portfolio/orders"
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": order_type,
            "yes_price": yes_price,
            "count": count,
        }
        return self._request("POST", path, json=body)

    def get_order(self, order_id: str) -> dict:
        """Get order status by ID."""
        path = f"/portfolio/orders/{order_id}"
        data = self._request("GET", path)
        return data.get("order", data)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        path = f"/portfolio/orders/{order_id}"
        return self._request("DELETE", path)

    def get_positions(self) -> list[dict]:
        """Get current positions."""
        path = "/portfolio/positions"
        data = self._request("GET", path)
        return data.get("market_positions", [])

    def get_fills(self, ticker: str | None = None, limit: int = 100) -> list[dict]:
        """Get trade history."""
        path = f"/portfolio/fills?limit={limit}"
        if ticker:
            path += f"&ticker={ticker}"
        data = self._request("GET", path)
        return data.get("fills", [])

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        path = "/portfolio/balance"
        data = self._request("GET", path)
        return data.get("balance", 0) / 100  # cents -> dollars

    def close(self):
        self._client.close()
