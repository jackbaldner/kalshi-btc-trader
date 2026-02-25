"""Deribit public API client for BTC options/vol data (async).

Fetches DVOL index, BTC-PERPETUAL ticker, and nearest-expiry ATM options.
All endpoints are public — no authentication required.
"""

import logging
import time

import aiohttp

logger = logging.getLogger(__name__)


class DeribitClient:
    """Fetches BTC volatility and options data from Deribit public API."""

    def __init__(self, cfg: dict):
        self.base_url = cfg.get("deribit", {}).get(
            "base_url", "https://www.deribit.com/api/v2"
        )
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_dvol(self) -> dict:
        """Fetch latest DVOL (BTC volatility index) 1h candle.

        Returns dict with open/high/low/close/timestamp, or {} on error.
        """
        try:
            session = await self._get_session()
            now_ms = int(time.time() * 1000)
            two_hours_ago = now_ms - 2 * 3600 * 1000
            params = {
                "currency": "BTC",
                "resolution": 3600,
                "start_timestamp": two_hours_ago,
                "end_timestamp": now_ms,
            }
            url = f"{self.base_url}/public/get_volatility_index_data"
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            # Response: {"result": {"data": [[ts, open, high, low, close], ...], ...}}
            candles = data.get("result", {}).get("data", [])
            if not candles:
                return {}
            latest = candles[-1]
            return {
                "timestamp": int(latest[0]),
                "open": float(latest[1]),
                "high": float(latest[2]),
                "low": float(latest[3]),
                "close": float(latest[4]),
            }
        except Exception as e:
            logger.warning(f"Deribit DVOL fetch failed: {e}")
            return {}

    async def get_perpetual_ticker(self) -> dict:
        """Fetch BTC-PERPETUAL ticker (funding rate, OI, mark/index price).

        Returns dict with funding_8h, current_funding, open_interest,
        mark_price, index_price, or {} on error.
        """
        try:
            session = await self._get_session()
            params = {"instrument_name": "BTC-PERPETUAL"}
            url = f"{self.base_url}/public/ticker"
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            result = data.get("result", {})
            return {
                "funding_8h": float(result.get("funding_8h", 0)),
                "current_funding": float(result.get("current_funding", 0)),
                "open_interest": float(result.get("open_interest", 0)),
                "mark_price": float(result.get("mark_price", 0)),
                "index_price": float(result.get("index_price", 0)),
            }
        except Exception as e:
            logger.warning(f"Deribit perpetual ticker fetch failed: {e}")
            return {}

    async def get_options_summary(self) -> dict:
        """Fetch nearest-expiry ATM call/put implied vol.

        Finds the nearest expiry, picks the strike closest to index price,
        and returns call/put IV with skew.

        Returns dict with atm_call_iv, atm_put_iv, put_call_skew,
        nearest_expiry, strike, or {} on error.
        """
        try:
            session = await self._get_session()
            params = {"currency": "BTC", "kind": "option"}
            url = f"{self.base_url}/public/get_book_summary_by_currency"
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            instruments = data.get("result", [])
            if not instruments:
                return {}

            # Parse instrument names: e.g. "BTC-28FEB25-100000-C"
            # Filter to instruments with valid mark_iv
            options = []
            for inst in instruments:
                name = inst.get("instrument_name", "")
                parts = name.split("-")
                if len(parts) != 4:
                    continue
                mark_iv = inst.get("mark_iv")
                if mark_iv is None or mark_iv == 0:
                    continue
                options.append({
                    "name": name,
                    "expiry": parts[1],
                    "strike": float(parts[2]),
                    "type": parts[3],  # C or P
                    "mark_iv": float(mark_iv),
                    "bid_iv": float(inst.get("bid_iv", 0) or 0),
                    "ask_iv": float(inst.get("ask_iv", 0) or 0),
                    "underlying_index": inst.get("underlying_index", ""),
                    "underlying_price": float(inst.get("underlying_price", 0) or 0),
                })

            if not options:
                return {}

            # Find the nearest expiry
            expiries = sorted(set(o["expiry"] for o in options))
            nearest_expiry = expiries[0]

            # Filter to nearest expiry only
            nearest = [o for o in options if o["expiry"] == nearest_expiry]
            if not nearest:
                return {}

            # Get index price from any option
            index_price = nearest[0].get("underlying_price", 0)
            if index_price == 0:
                return {}

            # Find the strike closest to index price
            strikes = sorted(set(o["strike"] for o in nearest))
            atm_strike = min(strikes, key=lambda s: abs(s - index_price))

            # Get call and put at that strike
            atm_call = [o for o in nearest if o["strike"] == atm_strike and o["type"] == "C"]
            atm_put = [o for o in nearest if o["strike"] == atm_strike and o["type"] == "P"]

            call_iv = atm_call[0]["mark_iv"] if atm_call else 0.0
            put_iv = atm_put[0]["mark_iv"] if atm_put else 0.0
            skew = put_iv - call_iv

            return {
                "atm_call_iv": call_iv,
                "atm_put_iv": put_iv,
                "put_call_skew": skew,
                "nearest_expiry": nearest_expiry,
                "strike": atm_strike,
            }
        except Exception as e:
            logger.warning(f"Deribit options summary fetch failed: {e}")
            return {}
