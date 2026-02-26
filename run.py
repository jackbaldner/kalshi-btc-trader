"""Main entry point for the Kalshi BTC 15-minute binary trading system."""

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

from kalshi_trader.config import load_config
from kalshi_trader.dashboard.cli import Dashboard
from kalshi_trader.data.binance import BinanceClient
from kalshi_trader.data.database import Database
from kalshi_trader.data.deribit import DeribitClient
from kalshi_trader.data.kalshi_client import KalshiClient
from kalshi_trader.data.kalshi_mock import KalshiMock
from kalshi_trader.data.kalshi_ws import KalshiWebSocket
from kalshi_trader.execution.order_manager import OrderManager
from kalshi_trader.execution.risk import RiskManager
from kalshi_trader.execution.trade_logger import TradeLogger
import re

import numpy as np

from kalshi_trader.models.bracket_prob import (
    estimate_bracket_prob_from_vol,
    implied_vol_from_bracket_price,
    parse_bracket_bounds,
)
from kalshi_trader.models.vol_model import VolModel
from kalshi_trader.notifications.discord import (
    configure as configure_discord,
    send_startup_alert,
    send_trade_alert,
    send_settlement_alert,
)
from kalshi_trader.strategies.ensemble import EnsembleStrategy

logger = logging.getLogger(__name__)


class TradingSystem:
    """Main trading loop that orchestrates all components."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg["mode"]
        self.running = False

        # Data
        self.binance = BinanceClient(cfg)
        self.deribit = DeribitClient(cfg)
        self.db = Database(cfg["database"]["path"])

        # Kalshi clients
        # Paper mode: real API for market data, mock for order execution
        # Live mode: real API for everything
        self.kalshi_reader = KalshiClient(cfg)  # always use real API for market data
        if self.mode == "live":
            self.kalshi = KalshiClient(cfg)
        else:
            self.kalshi = KalshiMock(cfg)

        # Strategy
        self.ensemble = EnsembleStrategy(cfg)
        self.vol_model = VolModel(
            lookback=cfg["strategy"].get("vol_model_lookback", 20),
            vol_floor_ratio=cfg["strategy"].get("vol_floor_ratio", 0.5),
        )

        # Execution
        self.order_manager = OrderManager(self.kalshi, cfg)
        self.risk = RiskManager(cfg, self.db)
        self.trade_logger = TradeLogger(self.db, cfg["logging"]["trade_csv"])

        # WebSocket for real-time orderbook
        self.kalshi_ws = KalshiWebSocket(cfg)

        # Dashboard
        self.dashboard = Dashboard()

        # Settlement
        self._settlement_interval = 60  # check every 60 seconds
        self._last_settlement_check = 0

        # Retraining
        self._retrain_interval = 6 * 3600  # retrain every 6 hours
        self._last_retrain_time = 0

        # State
        self._last_binance_poll = 0
        self._last_kalshi_poll = 0
        self._btc_price = 0.0
        self._prev_btc_price = 0.0
        self._price_unchanged_count = 0
        self._funding_rate = 0.0
        self._current_market = None
        self._current_event_ticker = None
        self._current_bracket = None  # (low, high) bounds of selected bracket
        self._all_brackets = []  # sorted list of (market, (low, high))
        self._traded_brackets_this_window = set()  # per-bracket dedup within a window
        self._orderbook = None
        self._candles = None
        self._trained = False
        self._last_trade_window = ""  # dedup key: event close time
        self._event_close_time = None  # ISO timestamp of current event close
        self._last_deribit_poll = 0

    async def start(self):
        """Initialize and start the trading loop."""
        logger.info(f"Starting trading system in {self.mode} mode")

        # Load candles: try DB first (from backtest), then fetch fresh
        self._candles = self.db.get_candles(limit=10000)
        logger.info(f"Loaded {len(self._candles)} candles from database")

        if len(self._candles) < 100:
            logger.info("Fetching candle data from Binance...")
            fresh = await self.binance.get_klines(limit=100)
            if not fresh.empty:
                self.db.upsert_candles(fresh)
                self._candles = self.db.get_candles(limit=10000)

        # One-time backfill: fetch 180 days of history if DB is sparse
        if len(self._candles) < 17000:
            logger.info(
                f"Backfilling historical candles (have {len(self._candles)}, "
                f"target ~17,000 for 180 days)..."
            )
            try:
                historical = await self.binance.get_historical_klines(days=180)
                if not historical.empty:
                    self.db.upsert_candles(historical)
                    self._candles = self.db.get_candles(limit=10000)
                    logger.info(f"Backfilled to {len(self._candles)} candles in DB")
            except Exception as e:
                logger.warning(f"Historical backfill failed (non-fatal): {e}")

        # Train models
        if len(self._candles) >= 50:
            self.ensemble.fair_value_model.train(self._candles, self._funding_rate)
            self.vol_model.train(self._candles, self._funding_rate)
            self._trained = True
            self._last_retrain_time = time.time()
            logger.info(f"Models trained on {len(self._candles)} candles")

        # Configure Discord notifications
        configure_discord(self.cfg.get("discord", {}).get("webhook_url"))

        # Get initial price
        self._btc_price = await self.binance.get_price()
        logger.info(f"BTC price: ${self._btc_price:,.2f}")

        # Send startup alert
        try:
            balance = self.kalshi.get_balance()
            send_startup_alert(
                mode=self.mode,
                balance=balance,
                event_ticker=self.cfg["kalshi"].get("event_ticker", "KXBTC"),
            )
        except Exception as e:
            logger.warning(f"Failed to send startup alert: {e}")

        # Main loop
        self.running = True
        await self._run_loop()

    async def _run_loop(self):
        """Main async event loop."""
        binance_interval = self.cfg["polling"]["binance_interval_sec"]
        kalshi_interval = self.cfg["polling"]["kalshi_interval_sec"]

        # Start dashboard and WebSocket in background
        dashboard_task = asyncio.create_task(self._dashboard_loop())
        ws_task = asyncio.create_task(self._ws_loop())

        try:
            while self.running:
                now = time.time()

                # Poll Binance
                if now - self._last_binance_poll >= binance_interval:
                    await self._poll_binance()
                    self._last_binance_poll = now

                # Poll Kalshi
                if now - self._last_kalshi_poll >= kalshi_interval:
                    await self._poll_kalshi()
                    self._last_kalshi_poll = now

                # Poll Deribit (options/vol data, 60s interval)
                if now - self._last_deribit_poll >= 60:
                    await self._poll_deribit()
                    self._last_deribit_poll = now

                # Settle expired trades and resolve evaluations
                if now - self._last_settlement_check >= self._settlement_interval:
                    await self._settle_trades()
                    await self._resolve_evaluations()
                    self.order_manager.cancel_stale_orders()
                    self._last_settlement_check = now

                # Periodic model retraining
                if now - self._last_retrain_time >= self._retrain_interval:
                    await self._retrain_models()
                    self._last_retrain_time = now

                # Trade when ~15 minutes remain until market close
                # This aligns with the vol model's 15-minute training window
                if self._event_close_time:
                    try:
                        close_dt = datetime.fromisoformat(
                            self._event_close_time.replace("Z", "+00:00")
                        )
                        utc_now = datetime.now(timezone.utc)
                        secs_until_close = (close_dt - utc_now).total_seconds()
                        trade_before_sec = self.cfg["strategy"].get(
                            "trade_before_close_sec", 900
                        )  # 900s = 15 min
                        # Trigger in a 30-second window around the target time
                        if (trade_before_sec - 30) <= secs_until_close <= trade_before_sec:
                            try:
                                await self._evaluate_and_trade()
                            except Exception as e:
                                logger.error(f"Evaluate-and-trade failed: {e}", exc_info=True)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Could not parse close time: {e}")

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        finally:
            dashboard_task.cancel()
            ws_task.cancel()
            await self._shutdown()

    async def _ws_loop(self):
        """Run WebSocket orderbook feed. Waits for a market to be selected, then connects."""
        try:
            # Wait until we have a market to subscribe to
            while self.running and not self._current_market:
                await asyncio.sleep(1)

            if self._current_market:
                ticker = self._current_market["ticker"]
                await self.kalshi_ws.connect(ticker)
        except asyncio.CancelledError:
            try:
                await self.kalshi_ws.close()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"WebSocket loop crashed: {e}")
            try:
                await self.kalshi_ws.close()
            except Exception:
                pass

    async def _poll_binance(self):
        """Fetch latest BTC price and candles."""
        try:
            self._btc_price = await self.binance.get_price()
            logger.info(f"BTC price: ${self._btc_price:,.2f}")

            # Staleness detection
            if self._prev_btc_price != 0.0 and self._btc_price == self._prev_btc_price:
                self._price_unchanged_count += 1
                if self._price_unchanged_count >= 5:
                    logger.warning(
                        f"BTC price unchanged for {self._price_unchanged_count} polls "
                        f"(${self._btc_price:,.2f}) — possible stale feed"
                    )
            else:
                self._price_unchanged_count = 0
            self._prev_btc_price = self._btc_price

            new_candles = await self.binance.get_klines(limit=5)
            if not new_candles.empty:
                self.db.upsert_candles(new_candles)
                self._candles = self.db.get_candles(limit=10000)

            self._funding_rate = await self.binance.get_funding_rate()

        except Exception as e:
            logger.error(f"Binance poll error: {e}")

    async def _poll_kalshi(self):
        """Fetch latest Kalshi market data (always uses real API for reads).

        For price-range bracket markets (KXBTC), finds the bracket closest
        to the current BTC price — that's the most actionable market.
        """
        try:
            # Dynamically find the nearest expiring KXBTC event (within 24h)
            event_ticker, close_time = self.kalshi_reader.get_nearest_event_ticker(max_hours=24)
            if event_ticker and event_ticker != self._current_event_ticker:
                self._current_event_ticker = event_ticker
                self._event_close_time = close_time
                logger.info(f"Tracking event: {event_ticker} (closes {close_time})")
            elif not event_ticker:
                if self._current_event_ticker:
                    logger.warning("No events within 24h — waiting for new event")
                    self._current_event_ticker = None
                    self._all_brackets = []
                return

            markets = self.kalshi_reader.get_markets(event_ticker=self._current_event_ticker)
            if not markets:
                logger.debug("No open KXBTC markets found")
                return

            # Parse bracket bounds for all markets and select the one containing current price
            parsed = []
            for m in markets:
                bounds = parse_bracket_bounds(m)
                if bounds:
                    parsed.append((m, bounds))

            if not parsed:
                logger.warning("Could not parse bracket bounds from any market")
                self._current_market = markets[0]
                self._current_bracket = None
                self._all_brackets = []
            else:
                self._all_brackets = sorted(parsed, key=lambda mb: mb[1][0])
                # Prefer the bracket that contains the current BTC price
                containing = [
                    (m, b) for m, b in parsed
                    if b[0] <= self._btc_price < b[1]
                ]
                if containing:
                    self._current_market, self._current_bracket = containing[0]
                else:
                    # Fallback: nearest bracket to current price
                    nearest = min(
                        parsed,
                        key=lambda mb: min(abs(self._btc_price - mb[1][0]),
                                           abs(self._btc_price - mb[1][1])),
                    )
                    self._current_market, self._current_bracket = nearest

            ticker = self._current_market["ticker"]

            # Use WebSocket orderbook if available and fresh, else REST fallback
            if self.kalshi_ws.is_ready and self.kalshi_ws.age_seconds < 5:
                self._orderbook = self.kalshi_ws.orderbook
            else:
                self._orderbook = self.kalshi_reader.get_orderbook(ticker)

            # Switch WS subscription if market changed
            await self.kalshi_ws.switch_market(ticker)

            # Feed real orderbook to mock for realistic fill simulation
            if hasattr(self.kalshi, "set_live_orderbook"):
                self.kalshi.set_live_orderbook(self._orderbook)

            subtitle = self._current_market.get("subtitle", ticker)
            yes_bid = self._current_market.get("yes_bid", 0)
            yes_ask = self._current_market.get("yes_ask", 0)
            bracket_str = f"[{self._current_bracket[0]:,.0f}-{self._current_bracket[1]:,.0f}]" if self._current_bracket else "?"

            logger.info(
                f"Kalshi bracket: {bracket_str} {subtitle} | yes={yes_bid}/{yes_ask} | "
                f"{len(parsed)} parsed / {len(markets)} total brackets"
            )

            # Log time until next evaluation
            if self._event_close_time:
                try:
                    close_dt = datetime.fromisoformat(self._event_close_time.replace("Z", "+00:00"))
                    utc_now = datetime.now(timezone.utc)
                    secs_until_close = (close_dt - utc_now).total_seconds()
                    trade_before_sec = self.cfg["strategy"].get("trade_before_close_sec", 900)
                    secs_until_eval = secs_until_close - trade_before_sec
                    if secs_until_eval > 0:
                        h, remainder = divmod(int(secs_until_eval), 3600)
                        m, s = divmod(remainder, 60)
                        eval_time = close_dt - timedelta(seconds=trade_before_sec)
                        logger.info(
                            f"Next eval in {h}h{m:02d}m{s:02d}s "
                            f"(at {eval_time.strftime('%H:%M:%S UTC')}, "
                            f"{trade_before_sec // 60}min before close)"
                        )
                    elif secs_until_close > 0:
                        logger.info(f"Eval window active — {secs_until_close:.0f}s until market close")
                except (ValueError, TypeError):
                    pass

            # Snapshot to DB
            ts = int(time.time() * 1000)
            self.db.insert_snapshot(
                ts, ticker, yes_bid, yes_ask,
                self._current_market.get("no_bid", 0),
                self._current_market.get("no_ask", 0),
                self._orderbook or {},
            )

            # Update exposure from mock positions
            positions = self.kalshi.get_positions()
            self.risk.update_exposure(positions)

        except Exception as e:
            logger.error(f"Kalshi poll error: {e}")

    async def _evaluate_and_trade(self):
        """Run ensemble strategy across current bracket ± 2 adjacent brackets."""
        # Dedup: only evaluate once per market close event
        close_key = self._event_close_time or ""
        if close_key == self._last_trade_window:
            return
        # Mark dedup AFTER successful evaluation (set at end of method)
        self._traded_brackets_this_window = set()

        if self._candles is None or self._candles.empty:
            return

        if not self._trained:
            if len(self._candles) >= 50:
                self.ensemble.fair_value_model.train(self._candles, self._funding_rate)
                self.vol_model.train(self._candles, self._funding_rate)
                self._trained = True
            else:
                return

        if not self._all_brackets:
            if self._current_market and self._current_bracket:
                try:
                    bal = self.kalshi.get_balance()
                except Exception as e:
                    logger.error(f"Failed to fetch balance: {e}")
                    return
                await self._evaluate_single_bracket(
                    self._current_market, self._current_bracket, bal,
                )
                self._last_trade_window = close_key
            return

        # Find index of the bracket containing current price
        center_idx = None
        for i, (m, b) in enumerate(self._all_brackets):
            if b[0] <= self._btc_price < b[1]:
                center_idx = i
                break

        if center_idx is None:
            # Price not in any bracket — find nearest
            center_idx = min(
                range(len(self._all_brackets)),
                key=lambda i: min(abs(self._btc_price - self._all_brackets[i][1][0]),
                                  abs(self._btc_price - self._all_brackets[i][1][1])),
            )

        # Evaluate center ± 2 adjacent brackets
        start_idx = max(0, center_idx - 2)
        end_idx = min(len(self._all_brackets), center_idx + 3)  # +3 because range is exclusive
        brackets_to_eval = self._all_brackets[start_idx:end_idx]

        logger.info(
            f"Evaluating {len(brackets_to_eval)} brackets "
            f"(indices {start_idx}-{end_idx - 1} of {len(self._all_brackets)}, "
            f"center={center_idx})"
        )

        try:
            balance = self.kalshi.get_balance()
        except Exception as e:
            logger.error(f"Failed to fetch balance, skipping evaluation: {e}")
            return
        for market, bracket in brackets_to_eval:
            await self._evaluate_single_bracket(market, bracket, balance)
            # Refresh balance after each trade so Kelly sizing adapts
            try:
                balance = self.kalshi.get_balance()
            except Exception as e:
                logger.warning(f"Failed to refresh balance: {e}")
                continue

        # Mark this window as evaluated only after successful completion
        self._last_trade_window = close_key

    async def _evaluate_single_bracket(self, market: dict, bracket: tuple, balance: float):
        """Evaluate and potentially trade a single bracket."""
        ticker = market["ticker"]

        # Per-bracket dedup: don't re-trade same ticker within an hour
        if ticker in self._traded_brackets_this_window:
            return
        one_hour_ago_ms = int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp() * 1000)
        existing = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM trades WHERE timestamp >= ? AND ticker = ?",
            (one_hour_ago_ms, ticker),
        ).fetchone()
        if existing["c"] > 0:
            logger.debug(f"Already traded {ticker} within the last hour, skipping")
            return

        # Per-bracket daily limit: avoid overconcentration on one ticker
        max_daily_per_bracket = self.cfg["risk"].get("max_daily_trades_per_bracket", 4)
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_ms = int(day_start.timestamp() * 1000)
        daily_count = self.db.conn.execute(
            "SELECT COUNT(*) as c FROM trades WHERE timestamp >= ? AND ticker = ?",
            (day_start_ms, ticker),
        ).fetchone()
        if daily_count["c"] >= max_daily_per_bracket:
            logger.info(
                f"Daily limit reached for {ticker}: {daily_count['c']}/{max_daily_per_bracket} trades today, skipping"
            )
            return

        # Build market data for this bracket
        yes_bid = market.get("yes_bid", 50)
        yes_ask = market.get("yes_ask", 50)
        implied_prob = (yes_bid + yes_ask) / 200  # midpoint in [0,1]

        # Vol-based bracket probability
        bracket_prob = None
        predicted_vol = None
        implied_vol = None
        if self._btc_price > 0:
            model_prob = self.ensemble.fair_value_model.predict(self._candles, self._funding_rate)
            raw_pred_vol = self.vol_model.predict(self._candles, self._funding_rate)
            implied_vol = implied_vol_from_bracket_price(
                self._btc_price, bracket[0], bracket[1], implied_prob,
            )
            predicted_vol = self.vol_model.blend_with_implied(raw_pred_vol, implied_vol)
            bracket_prob = estimate_bracket_prob_from_vol(
                self._btc_price, bracket[0], bracket[1], predicted_vol, model_prob,
                raw_model_vol=raw_pred_vol,
            )
            logger.info(
                f"Vol: raw={raw_pred_vol:.5f} blended={predicted_vol:.5f} "
                f"impl={implied_vol:.5f} | "
                f"bracket_prob={bracket_prob:.4f} mkt={implied_prob:.4f} "
                f"edge={bracket_prob - implied_prob:.4f} | "
                f"bracket: [{bracket[0]:,.0f}-{bracket[1]:,.0f}]"
            )

        market_data = {
            "orderbook": self._orderbook or {},
            "funding_rate": self._funding_rate,
            "implied_prob": implied_prob,
            "bracket_prob": bracket_prob,
            "predicted_vol": predicted_vol,
            "implied_vol": implied_vol,
        }

        result = self.ensemble.evaluate(self._candles, market_data, balance)

        # Update dashboard with latest evaluation
        self.dashboard.update(
            current_edge=result.edge,
            regime=result.regime,
            last_signal=f"{result.direction.value} ({result.ensemble_prob:.2f})",
        )

        # Log every evaluation to DB (even if we don't trade)
        try:
            edge = (bracket_prob - implied_prob) if bracket_prob is not None else 0.0
            self.db.insert_evaluation(
                timestamp=int(time.time() * 1000),
                event_close_time=self._event_close_time or "",
                ticker=ticker,
                bracket_low=bracket[0],
                bracket_high=bracket[1],
                btc_price=self._btc_price,
                raw_model_vol=raw_pred_vol if self._btc_price > 0 else 0.0,
                predicted_vol=predicted_vol or 0.0,
                implied_vol=implied_vol or 0.0,
                blended_vol=predicted_vol or 0.0,
                shrinkage=self.vol_model.shrinkage,
                bracket_prob=bracket_prob or 0.0,
                market_prob=implied_prob,
                edge=edge,
                should_trade=result.should_trade,
            )
        except Exception as e:
            logger.warning(f"Failed to log evaluation: {e}")

        if not result.should_trade:
            logger.info(f"No edge for [{bracket[0]:,.0f}-{bracket[1]:,.0f}] — skipping")
            return

        # Use freshest orderbook available before trading
        if self.kalshi_ws.is_ready and self.kalshi_ws.age_seconds < 2:
            fresh_book = self.kalshi_ws.orderbook
            logger.info(f"Using WebSocket orderbook ({self.kalshi_ws.age_seconds:.1f}s old)")
        else:
            try:
                fresh_book = self.kalshi_reader.get_orderbook(ticker)
                logger.info(f"Refreshed orderbook via REST for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to refresh orderbook: {e}")
                fresh_book = self._orderbook

        if fresh_book and hasattr(self.kalshi, "set_live_orderbook"):
            self.kalshi.set_live_orderbook(fresh_book)
            self._orderbook = fresh_book

        # Smart pricing
        price_cents, available_depth = self._get_realistic_price(
            fresh_book or self._orderbook, result.side, result.edge, result.ensemble_prob,
        )
        if price_cents is None:
            logger.info(f"No fillable price with edge for {result.side} on {ticker} — skipping")
            return

        # Cap NO price — expensive NOs have terrible risk/reward
        max_no_price = self.cfg["risk"].get("max_no_price_cents", 50)
        if result.side == "no" and price_cents > max_no_price:
            logger.info(f"NO price {price_cents}c exceeds cap {max_no_price}c — skipping")
            return

        order_size = result.contracts
        if order_size <= 0:
            return

        # Liquidity check: cap order size to 50% of available book depth
        max_liquidity_pct = self.cfg["risk"].get("max_book_take_pct", 0.5)
        if available_depth > 0 and order_size > int(available_depth * max_liquidity_pct):
            capped_size = max(1, int(available_depth * max_liquidity_pct))
            logger.info(
                f"Liquidity cap: {order_size} -> {capped_size} contracts "
                f"(book has {available_depth}, taking max {max_liquidity_pct:.0%})"
            )
            order_size = capped_size

        if order_size <= 0:
            return

        logger.info(
            f"Order plan: {result.side} {order_size}x {ticker} @ {price_cents}c "
            f"(book has {available_depth})"
        )

        # Risk check
        allowed, reason = self.risk.check_order(result.side, price_cents, order_size, balance)
        if not allowed:
            logger.info(f"Trade blocked by risk: {reason}")
            return

        # Calculate fill confidence before placing order
        fill_confidence = self._estimate_fill_confidence(
            fresh_book or self._orderbook, result.side, price_cents, order_size,
        )

        # Place order
        try:
            order = self.order_manager.place_order(
                ticker, result.side, price_cents, order_size,
                expiry_time_ms=market.get("close_time"),
            )
        except Exception as e:
            logger.error(f"Order placement failed for {ticker}: {e}")
            return

        if order and "error" not in order:
            order_id = order.get("order_id") or order.get("order", {}).get("order_id")
            order_obj = order.get("order", order)
            status = order_obj.get("status", "")
            filled_count = order_obj.get("fill_count", 0)
            remaining = order_obj.get("remaining_count", order_size)

            if filled_count == 0 and order_id and status not in ("canceled", "cancelled"):
                # Order not immediately filled — wait briefly and recheck
                await asyncio.sleep(3)
                try:
                    order_status = self.kalshi.get_order(order_id)
                    filled_count = order_status.get("fill_count", 0)
                    remaining = order_status.get("remaining_count", order_size)
                    status = order_status.get("status", status)
                    logger.info(f"Recheck order {order_id}: fill_count={filled_count} status={status}")
                except Exception as e:
                    logger.warning(f"Failed to recheck order {order_id}: {e}")

            if filled_count > 0:
                # Order filled (fully or partially) — log the trade
                fill_price = order_obj.get("yes_price", price_cents)
                logger.info(
                    f"Order filled: {order_id} {result.side} {filled_count}x {ticker} "
                    f"@ {fill_price}c (status={status})"
                )
                self.trade_logger.log_trade(
                    ticker, result.side, fill_price, filled_count,
                    "ensemble", result.edge, result.ensemble_prob, implied_prob,
                    predicted_vol=predicted_vol, implied_vol=implied_vol,
                    fill_confidence=fill_confidence,
                    bracket_low=bracket[0], bracket_high=bracket[1],
                )
                try:
                    post_balance = self.kalshi.get_balance()
                except Exception:
                    post_balance = balance
                try:
                    send_trade_alert(
                        ticker=ticker,
                        side=result.side,
                        price=fill_price,
                        size=filled_count,
                        edge=result.edge,
                        bracket_low=bracket[0],
                        bracket_high=bracket[1],
                        balance=post_balance,
                        predicted_vol=predicted_vol,
                        implied_vol=implied_vol,
                    )
                except Exception as e:
                    logger.warning(f"Discord trade alert failed: {e}")
            else:
                # Order accepted but not filled after recheck
                logger.info(
                    f"Order resting (unfilled): {order_id} {result.side} {order_size}x {ticker} "
                    f"@ {price_cents}c (status={status}, remaining={remaining})"
                )

            # Schedule auto-cancel for any unfilled portion
            close_time = market.get("close_time")
            if order_id and close_time and self.mode == "live" and remaining > 0:
                try:
                    # Convert ISO string to milliseconds for schedule_auto_cancel
                    if isinstance(close_time, str):
                        close_dt_parsed = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                        close_time_ms = int(close_dt_parsed.timestamp() * 1000)
                    else:
                        close_time_ms = int(close_time)
                    loop = asyncio.get_running_loop()
                    self.order_manager.schedule_auto_cancel(order_id, close_time_ms, loop)
                except Exception as e:
                    logger.warning(f"Failed to schedule auto-cancel for {order_id}: {e}")
            self._traded_brackets_this_window.add(ticker)
            # Update exposure so risk checks work across multi-bracket trades
            try:
                positions = self.kalshi.get_positions()
                self.risk.update_exposure(positions)
            except Exception as e:
                logger.warning(f"Failed to update exposure after trade: {e}")

    def _get_realistic_price(self, book: dict, side: str, edge: float,
                               model_prob: float) -> tuple[int | None, int]:
        """Determine an aggressive fillable price from the orderbook.

        Prices 1c above the best ask to guarantee crossing the spread.
        Only proceeds if we still have edge after the aggressive pricing.

        Returns (price_cents, available_depth) or (None, 0) if no viable price.
        """
        if not book:
            return None, 0

        # Get ask levels we'd cross (opposite side bids converted)
        opposite_side = "no" if side == "yes" else "yes"
        opposite_bids = book.get(opposite_side, [])

        if not opposite_bids:
            return None, 0

        # Convert to ask prices, sorted cheapest first
        ask_levels = sorted(
            [(100 - p, q) for p, q in opposite_bids],
            key=lambda x: x[0],
        )

        edge_threshold = self.cfg["strategy"]["edge_threshold"]
        aggression = self.cfg["strategy"].get("price_aggression_cents", 1)

        for ask_price, depth in ask_levels:
            # Pay aggression cents above the ask to guarantee fill
            fill_price = min(ask_price + aggression, 99)

            if side == "yes":
                effective_implied = fill_price / 100.0
                remaining_edge = model_prob - effective_implied
            else:
                effective_implied = fill_price / 100.0
                remaining_edge = model_prob - effective_implied

            if remaining_edge >= edge_threshold:
                total_depth = sum(q for p, q in ask_levels if p <= fill_price)
                logger.info(
                    f"Aggressive price: {side} @ {fill_price}c (ask={ask_price}c +{aggression}c) | "
                    f"edge_at_fill={remaining_edge:.4f} depth={total_depth}"
                )
                return fill_price, total_depth

        # No ask level has enough edge even without aggression
        best_ask = ask_levels[0][0] if ask_levels else 0
        if side == "yes":
            best_edge = model_prob - best_ask / 100.0
        else:
            best_edge = model_prob - best_ask / 100.0
        logger.info(
            f"No edge at any ask: best_ask={best_ask}c edge_at_best={best_edge:.4f} "
            f"(need {edge_threshold})"
        )
        return None, 0

    def _estimate_fill_confidence(self, book: dict, side: str, limit_price: int,
                                    order_size: int) -> float:
        """Estimate the probability (0-100%) this order would fill in a live environment.

        Factors:
        - Depth ratio: how much of the available liquidity we're consuming
        - Book staleness: how old the orderbook data is
        - Spread: wider spread = less liquid = lower confidence
        - Fill completeness: could we even fill the full order in simulation?
        """
        if not book:
            return 10.0  # no book data = very low confidence

        # 1. Depth analysis: walk the book to see available liquidity
        opposite_side = "no" if side == "yes" else "yes"
        opposite_bids = book.get(opposite_side, [])

        if not opposite_bids:
            return 5.0  # no liquidity on the other side

        # Convert to ask levels we'd cross
        ask_levels = sorted(
            [(100 - p, q) for p, q in opposite_bids],
            key=lambda x: x[0],
        )

        # Total available at or below our limit
        available = sum(q for price, q in ask_levels if price <= limit_price)

        if available == 0:
            return 5.0  # nothing available at our price

        # Depth ratio: what fraction of available liquidity are we taking?
        # Taking <10% = great, >50% = risky (our order would move the market)
        depth_ratio = order_size / available
        if depth_ratio <= 0.05:
            depth_score = 100.0
        elif depth_ratio <= 0.10:
            depth_score = 90.0
        elif depth_ratio <= 0.25:
            depth_score = 70.0
        elif depth_ratio <= 0.50:
            depth_score = 45.0
        elif depth_ratio <= 1.0:
            depth_score = 25.0
        else:
            depth_score = 10.0  # can't even fill the order

        # 2. Staleness penalty
        book_age = self.kalshi_ws.age_seconds if self.kalshi_ws.is_ready else 30.0
        if book_age < 1:
            staleness_score = 100.0   # real-time WebSocket
        elif book_age < 3:
            staleness_score = 90.0    # very fresh
        elif book_age < 10:
            staleness_score = 70.0    # acceptable
        elif book_age < 30:
            staleness_score = 45.0    # stale
        else:
            staleness_score = 20.0    # very stale

        # 3. Spread score
        same_bids = book.get(side, [])
        if same_bids and opposite_bids:
            best_bid = max(p for p, q in same_bids) if same_bids else 0
            best_ask = min(100 - p for p, q in opposite_bids)
            spread = best_ask - best_bid
            if spread <= 2:
                spread_score = 95.0
            elif spread <= 5:
                spread_score = 80.0
            elif spread <= 10:
                spread_score = 60.0
            elif spread <= 20:
                spread_score = 35.0
            else:
                spread_score = 15.0
        else:
            spread_score = 20.0

        # 4. Price impact: how many levels do we eat through?
        levels_consumed = sum(1 for p, q in ask_levels if p <= limit_price and q > 0)
        total_levels = len(ask_levels)
        if levels_consumed <= 1:
            impact_score = 95.0
        elif levels_consumed <= 2:
            impact_score = 80.0
        elif levels_consumed <= 3:
            impact_score = 60.0
        else:
            impact_score = 35.0

        # Weighted composite
        confidence = (
            depth_score * 0.35 +
            staleness_score * 0.30 +
            spread_score * 0.20 +
            impact_score * 0.15
        )

        confidence = max(0.0, min(100.0, confidence))

        logger.info(
            f"Fill confidence: {confidence:.1f}% "
            f"(depth={depth_score:.0f} stale={staleness_score:.0f} "
            f"spread={spread_score:.0f} impact={impact_score:.0f} | "
            f"available={available} need={order_size} ratio={depth_ratio:.2f} "
            f"book_age={book_age:.1f}s)"
        )

        return confidence

    async def _retrain_models(self):
        """Retrain vol and fair value models on all available candles.

        Safety: if the new model's median prediction changes by more than 3x
        compared to the old model, log a warning and keep the old model.
        """
        candles = self.db.get_candles(limit=10000)
        if len(candles) < 200:
            return

        # Capture old model's median prediction for safety check
        old_median = None
        if self.vol_model._fitted and self._candles is not None and not self._candles.empty:
            try:
                X = self.vol_model._compute_features(self._candles, self._funding_rate)
                last = X[-1:]
                if not np.isnan(last).any():
                    X_scaled = self.vol_model.scaler.transform(last)
                    old_median = float(self.vol_model.model.predict(X_scaled)[0])
            except Exception:
                pass

        # Retrain
        try:
            self.vol_model.train(candles, self._funding_rate)
            self.ensemble.fair_value_model.train(candles, self._funding_rate)
        except Exception as e:
            logger.error(f"Model retraining failed: {e}", exc_info=True)
            return

        # Safety check: compare new median prediction
        if old_median and old_median > 0 and self._candles is not None and not self._candles.empty:
            try:
                X = self.vol_model._compute_features(self._candles, self._funding_rate)
                last = X[-1:]
                if not np.isnan(last).any():
                    X_scaled = self.vol_model.scaler.transform(last)
                    new_median = float(self.vol_model.model.predict(X_scaled)[0])
                    ratio = new_median / old_median if old_median > 0 else 1.0
                    if ratio > 3.0 or ratio < 1 / 3.0:
                        logger.warning(
                            f"Retrained model prediction shifted {ratio:.2f}x "
                            f"(old={old_median:.6f} new={new_median:.6f}) — "
                            f"this is suspicious but keeping new model"
                        )
            except Exception:
                pass

        self._candles = candles
        self._trained = True
        logger.info(f"Models retrained on {len(candles)} candles")

    async def _resolve_evaluations(self):
        """Resolve evaluations whose event windows have closed.

        For each unresolved evaluation, looks up the actual BTC price at the event close
        time and computes realized return/vol. This creates the (predicted, implied, realized)
        triples needed for future model calibration.
        """
        unresolved = self.db.get_unresolved_evaluations()
        if not unresolved:
            return

        now_ms = int(time.time() * 1000)

        for ev in unresolved:
            # Parse event close time to determine if the window has elapsed
            close_time_str = ev.get("event_close_time", "")
            if not close_time_str:
                continue

            try:
                close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                close_time_ms = int(close_dt.timestamp() * 1000)
            except (ValueError, TypeError):
                continue

            # Only resolve after the event has closed (+ 60s buffer for candle availability)
            if now_ms < close_time_ms + 60_000:
                continue

            # Look up the candle that covers the event close time
            row = self.db.conn.execute(
                "SELECT close FROM candles WHERE open_time <= ? ORDER BY open_time DESC LIMIT 1",
                (close_time_ms,),
            ).fetchone()
            if not row:
                continue

            price_at_close = float(row["close"])
            btc_price = ev["btc_price"]
            if btc_price <= 0:
                continue

            realized_return = (price_at_close - btc_price) / btc_price
            realized_vol = abs(realized_return)
            in_bracket = ev["bracket_low"] <= price_at_close < ev["bracket_high"]

            self.db.resolve_evaluation(
                eval_id=ev["id"],
                realized_vol=realized_vol,
                realized_return=realized_return,
                price_at_close=price_at_close,
                in_bracket=in_bracket,
            )

        resolved_count = len([e for e in unresolved
                              if e.get("event_close_time")
                              and now_ms >= int(datetime.fromisoformat(
                                  e["event_close_time"].replace("Z", "+00:00")
                              ).timestamp() * 1000) + 60_000])
        if resolved_count > 0:
            logger.info(f"Resolved {resolved_count} evaluations with realized vol data")

    async def _settle_trades(self):
        """Check unresolved trades and settle.

        Paper mode: simulate settlement using BTC price 15 min after trade.
        Live mode: query Kalshi API for actual position/fill status.
        """
        if self.mode == "live":
            await self._settle_trades_live()
            return

        unresolved = self.db.get_unresolved_trades()
        if not unresolved:
            return

        now_ms = int(time.time() * 1000)
        settle_delay_ms = 15 * 60 * 1000  # 15 minutes

        for trade in unresolved:
            # Only settle after the 15-min window has elapsed
            if now_ms - trade["timestamp"] < settle_delay_ms:
                continue

            ticker = trade["ticker"]

            # Determine bracket bounds (priority: stored → cache → API → ticker fallback)
            bracket_low = trade.get("bracket_low")
            bracket_high = trade.get("bracket_high")

            if bracket_low is None or bracket_high is None:
                # Try cached brackets from current polling
                cached = next(
                    ((m_, b) for m_, b in self._all_brackets if m_["ticker"] == ticker),
                    None,
                )
                if cached:
                    bracket_low, bracket_high = cached[1]
                    logger.info(f"Settlement bounds from cache for {ticker}: [{bracket_low:,.0f}-{bracket_high:,.0f}]")
                else:
                    # Try fetching from Kalshi API
                    try:
                        api_market = self.kalshi_reader.get_market(ticker)
                        if api_market:
                            bounds = parse_bracket_bounds(api_market)
                            if bounds:
                                bracket_low, bracket_high = bounds
                                logger.info(f"Settlement bounds from API for {ticker}: [{bracket_low:,.0f}-{bracket_high:,.0f}]")
                    except Exception as e:
                        logger.debug(f"API lookup failed for {ticker}: {e}")

            if bracket_low is None or bracket_high is None:
                # Last resort: ticker parse with $500 width
                m = re.search(r"-B(\d+)$", ticker)
                if not m:
                    logger.warning(f"Cannot parse bracket from ticker {ticker}, skipping settlement")
                    continue
                bracket_low = float(m.group(1))
                bracket_high = bracket_low + 500.0
                logger.warning(
                    f"Using hardcoded $500 bracket width for settlement of {ticker} — "
                    f"actual width may differ"
                )

            # Find the BTC price at settlement time (trade time + 15 min)
            settle_time_ms = trade["timestamp"] + settle_delay_ms
            # Get the candle that covers the settlement time
            row = self.db.conn.execute(
                "SELECT close FROM candles WHERE open_time <= ? ORDER BY open_time DESC LIMIT 1",
                (settle_time_ms,),
            ).fetchone()

            if not row:
                logger.debug(f"No candle data yet for trade {trade['id']} settlement")
                continue

            settle_price = float(row["close"])
            in_bracket = bracket_low <= settle_price < bracket_high
            result = "yes" if in_bracket else "no"

            # Calculate P&L
            side = trade["side"]
            price_cents = trade["price"]
            size = trade["size"]
            cost = price_cents * size / 100  # what we paid

            if side == result:
                # Won: receive $1 per contract, minus cost
                pnl = size - cost
            else:
                # Lost: lose the cost
                pnl = -cost

            # Resolve in mock client and DB
            if hasattr(self.kalshi, "resolve_market"):
                self.kalshi.resolve_market(ticker, result)

            self.trade_logger.update_pnl(trade["id"], pnl)
            logger.info(
                f"Settled trade {trade['id']}: {ticker} {side} @ {price_cents}c x{int(size)} | "
                f"BTC={settle_price:,.2f} bracket=[{bracket_low:,.0f}-{bracket_high:,.0f}] "
                f"result={result} PnL=${pnl:.2f}"
            )
            try:
                send_settlement_alert(
                    ticker=ticker,
                    side=side,
                    result=result,
                    pnl=pnl,
                    price=price_cents,
                    size=int(size),
                )
            except Exception as e:
                logger.warning(f"Discord settlement alert failed: {e}")

    async def _settle_trades_live(self):
        """Settle trades using actual Kalshi position data.

        Queries the API for fills and resolved positions to determine real P&L.
        """
        unresolved = self.db.get_unresolved_trades()
        if not unresolved:
            return

        try:
            fills = self.kalshi_reader.get_fills(limit=200)
        except Exception as e:
            logger.error(f"Failed to fetch fills for settlement: {e}")
            return

        # Build a map of ticker -> total filled info
        fill_map = {}
        for fill in fills:
            ticker = fill.get("ticker", "")
            if ticker not in fill_map:
                fill_map[ticker] = []
            fill_map[ticker].append(fill)

        for trade in unresolved:
            ticker = trade["ticker"]

            # Check if this market has settled by looking at the market status
            try:
                market = self.kalshi_reader.get_market(ticker)
                status = market.get("status", "")
                result = market.get("result", "")

                if status not in ("settled", "finalized"):
                    continue  # market still open, skip

                # Market is settled — calculate real P&L
                side = trade["side"]
                price_cents = trade["price"]
                size = trade["size"]
                cost = price_cents * size / 100

                if side == result:
                    pnl = size - cost  # won
                else:
                    pnl = -cost  # lost

                self.trade_logger.update_pnl(trade["id"], pnl)
                logger.info(
                    f"Live settlement: trade {trade['id']} {ticker} {side} @ {price_cents}c x{int(size)} | "
                    f"result={result} PnL=${pnl:.2f}"
                )
                try:
                    send_settlement_alert(
                        ticker=ticker,
                        side=side,
                        result=result,
                        pnl=pnl,
                        price=price_cents,
                        size=int(size),
                    )
                except Exception as e:
                    logger.warning(f"Discord settlement alert failed: {e}")

            except Exception as e:
                logger.debug(f"Cannot check settlement for {ticker}: {e}")
                continue

    async def _dashboard_loop(self):
        """Periodically update dashboard data."""
        while self.running:
            try:
                trades_df = self.db.get_recent_trades(limit=100)
                total = len(trades_df)
                wins = len(trades_df[trades_df["pnl"] > 0]) if not trades_df.empty and "pnl" in trades_df else 0

                day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                daily_pnl = self.db.get_daily_pnl(int(day_start.timestamp() * 1000))

                recent = []
                if not trades_df.empty:
                    for _, row in trades_df.head(10).iterrows():
                        recent.append({
                            "time": datetime.fromtimestamp(row["timestamp"] / 1000, tz=timezone.utc).strftime("%H:%M:%S"),
                            "side": row["side"],
                            "price": row["price"],
                            "size": row["size"],
                            "edge": row.get("edge", 0),
                            "pnl": row.get("pnl"),
                        })

                # Next expiry — close_time is ISO string from Kalshi API
                next_expiry = ""
                if self._current_market:
                    close_time_str = self._current_market.get("close_time", "")
                    if close_time_str and isinstance(close_time_str, str):
                        try:
                            exp_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                            remaining = (exp_dt - datetime.now(timezone.utc)).total_seconds()
                            if remaining > 0:
                                next_expiry = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                            else:
                                next_expiry = "EXPIRED"
                        except (ValueError, TypeError):
                            pass

                self.dashboard.update(
                    mode=self.mode,
                    balance=self.kalshi.get_balance(),
                    btc_price=self._btc_price,
                    positions=self.kalshi.get_positions(),
                    total_trades=total,
                    win_rate=wins / total if total > 0 else 0.0,
                    daily_pnl=daily_pnl,
                    exposure=self.risk.current_exposure,
                    next_expiry=next_expiry,
                    recent_trades=recent,
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Dashboard loop error: {e}", exc_info=True)

            await asyncio.sleep(5)

    async def _poll_deribit(self):
        """Fetch Deribit options/vol data and log to DB."""
        try:
            dvol, perpetual, options = await asyncio.gather(
                self.deribit.get_dvol(),
                self.deribit.get_perpetual_ticker(),
                self.deribit.get_options_summary(),
            )

            # Log summary
            dvol_val = dvol.get("close", 0)
            funding = perpetual.get("current_funding", 0)
            oi = perpetual.get("open_interest", 0)
            skew = options.get("put_call_skew", 0)
            logger.info(
                f"Deribit: DVOL={dvol_val:.1f}% "
                f"perp_funding={funding:.4f}% "
                f"OI=${oi / 1e6:.1f}M "
                f"skew={skew:.2f}%"
            )

            # Store snapshot
            ts = int(time.time() * 1000)
            self.db.insert_deribit_snapshot(ts, dvol, perpetual, options)
        except Exception as e:
            logger.warning(f"Deribit poll failed (non-fatal): {e}")

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        self.order_manager.cancel_all()
        self.order_manager.cleanup()
        await self.kalshi_ws.close()
        await self.binance.close()
        await self.deribit.close()
        if hasattr(self.kalshi, "close"):
            self.kalshi.close()
        if hasattr(self.kalshi_reader, "close"):
            self.kalshi_reader.close()
        self.db.close()
        logger.info("Shutdown complete")


async def run_with_dashboard(system: TradingSystem):
    """Run trading system with dashboard display."""
    # Start trading in background
    trade_task = asyncio.create_task(system.start())

    # Dashboard renders synchronously in main thread via Rich Live
    # The dashboard_loop inside TradingSystem handles async updates
    try:
        from rich.live import Live
        with Live(system.dashboard.render(), refresh_per_second=0.5) as live:
            while system.running:
                live.update(system.dashboard.render())
                await asyncio.sleep(2)
    except KeyboardInterrupt:
        system.running = False

    await trade_task


def main():
    parser = argparse.ArgumentParser(description="Kalshi BTC 15-Min Binary Trader")
    parser.add_argument("--mode", choices=["paper", "live"], default=None,
                        help="Trading mode (overrides config)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Run without dashboard UI")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    # Also write to log file directly (unbuffered)
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    file_handler = logging.FileHandler("logs/live.log", mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
        if args.mode == "live":
            cfg["kalshi"]["base_url"] = cfg["kalshi"]["live_url"]

    # Live mode safety checks
    if cfg["mode"] == "live":
        from pathlib import Path
        api_key = cfg["kalshi"].get("api_key_id", "")
        pk_path = cfg["kalshi"].get("private_key_path", "")
        if not api_key or api_key.startswith("your-"):
            print("ERROR: Live mode requires real API credentials in .env")
            sys.exit(1)
        if not pk_path or not Path(pk_path).exists():
            print(f"ERROR: Private key not found at: {pk_path}")
            sys.exit(1)
        # Confirm with user before starting live trading
        print(f"\n*** LIVE TRADING MODE ***")
        print(f"  API Key: {api_key[:8]}...{api_key[-4:]}")
        print(f"  Max position: ${cfg['risk']['max_position_size']}")
        print(f"  Max exposure: ${cfg['risk']['max_total_exposure']}")
        print(f"  Max daily loss: ${cfg['risk']['max_daily_loss']}")
        print(f"  Edge threshold: {cfg['strategy']['edge_threshold']:.0%}")
        print(f"  Kelly fraction: {cfg['strategy']['kelly_fraction']}")
        confirm = input("\nType 'YES' to confirm live trading: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            sys.exit(0)

    system = TradingSystem(cfg)

    # Handle Ctrl+C
    def handle_sigint(sig, frame):
        system.running = False

    signal.signal(signal.SIGINT, handle_sigint)

    if args.no_dashboard:
        asyncio.run(system.start())
    else:
        asyncio.run(run_with_dashboard(system))


if __name__ == "__main__":
    main()
