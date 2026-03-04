"""Binance public API client for BTC price data (async).

Uses Binance.us for US-based users (Binance.com returns 451 for US IPs).
Falls back gracefully if futures endpoint is unavailable.
"""

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# Binance endpoints to try in order (US-accessible first)
_SPOT_ENDPOINTS = [
    "https://api.binance.us",
    "https://api.binance.com",
]


class BinanceClient:
    """Fetches BTC spot OHLCV, current price, and funding rate from Binance."""

    def __init__(self, cfg: dict):
        self.base_url = cfg["binance"]["base_url"]
        self.futures_url = cfg["binance"]["futures_url"]
        self.symbol = cfg["binance"]["symbol"]
        self.interval = cfg["binance"]["interval"]
        self._session: aiohttp.ClientSession | None = None
        self._working_base_url: str | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _resolve_base_url(self) -> str:
        """Find a working Binance spot endpoint (handles US geo-blocking)."""
        if self._working_base_url:
            return self._working_base_url

        session = await self._get_session()
        candidates = [self.base_url] + [u for u in _SPOT_ENDPOINTS if u != self.base_url]
        for url in candidates:
            try:
                test = f"{url}/api/v3/ping"
                async with session.get(test, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        self._working_base_url = url
                        if url != self.base_url:
                            logger.info(f"Switched to accessible endpoint: {url}")
                        return url
            except Exception:
                continue

        # Fallback to configured URL
        self._working_base_url = self.base_url
        return self.base_url

    async def get_klines(self, limit: int = 100, start_time: int | None = None) -> pd.DataFrame:
        """Fetch 15m OHLCV candles. Returns DataFrame with columns:
        open_time, open, high, low, close, volume, close_time."""
        session = await self._get_session()
        base = await self._resolve_base_url()
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time

        url = f"{base}/api/v3/klines"
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        rows = []
        for k in data:
            rows.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df

    async def get_historical_klines(self, days: int = 90) -> pd.DataFrame:
        """Fetch historical klines by paginating backwards."""
        all_frames = []
        end_time = None
        candles_needed = days * 24 * 4  # 4 candles per hour for 15m
        fetched = 0

        while fetched < candles_needed:
            batch_limit = min(1000, candles_needed - fetched)
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "limit": batch_limit,
            }
            if end_time is not None:
                params["endTime"] = end_time

            session = await self._get_session()
            base = await self._resolve_base_url()
            url = f"{base}/api/v3/klines"
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            if not data:
                break

            rows = []
            for k in data:
                rows.append({
                    "open_time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": int(k[6]),
                })

            df = pd.DataFrame(rows)
            all_frames.append(df)
            fetched += len(df)
            end_time = int(data[0][0]) - 1  # before earliest candle in batch

            logger.info(f"Fetched {fetched}/{candles_needed} candles")
            await asyncio.sleep(0.2)  # respect rate limits

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        result = result.sort_values("open_time").drop_duplicates(subset="open_time").reset_index(drop=True)
        result["open_time_dt"] = pd.to_datetime(result["open_time"], unit="ms", utc=True)
        return result

    async def get_price(self) -> float:
        """Get current BTC spot price."""
        session = await self._get_session()
        base = await self._resolve_base_url()
        url = f"{base}/api/v3/ticker/price"
        async with session.get(url, params={"symbol": self.symbol}) as resp:
            resp.raise_for_status()
            data = await resp.json()
        return float(data["price"])

    async def get_funding_rate(self) -> float:
        """Get latest Binance perpetual funding rate.
        Falls back to 0.0 if futures endpoint is unavailable (e.g. Binance.us)."""
        session = await self._get_session()
        # Try global futures first, then gracefully degrade
        for futures_url in [self.futures_url, "https://fapi.binance.com"]:
            try:
                url = f"{futures_url}/fapi/v1/fundingRate"
                async with session.get(url, params={"symbol": self.symbol, "limit": 1},
                                       timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            return float(data[0]["fundingRate"])
            except Exception:
                continue
        logger.debug("Funding rate unavailable, defaulting to 0.0")
        return 0.0
