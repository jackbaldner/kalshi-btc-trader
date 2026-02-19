"""Main entry point for the Kalshi BTC 15-minute binary trading system."""

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from kalshi_trader.config import load_config
from kalshi_trader.dashboard.cli import Dashboard
from kalshi_trader.data.binance import BinanceClient
from kalshi_trader.data.database import Database
from kalshi_trader.data.kalshi_client import KalshiClient
from kalshi_trader.data.kalshi_mock import KalshiMock
from kalshi_trader.execution.order_manager import OrderManager
from kalshi_trader.execution.risk import RiskManager
from kalshi_trader.execution.trade_logger import TradeLogger
from kalshi_trader.models.bracket_prob import estimate_bracket_prob, parse_bracket_bounds
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

        # Execution
        self.order_manager = OrderManager(self.kalshi, cfg)
        self.risk = RiskManager(cfg, self.db)
        self.trade_logger = TradeLogger(self.db, cfg["logging"]["trade_csv"])

        # Dashboard
        self.dashboard = Dashboard()

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
        self._orderbook = None
        self._candles = None
        self._trained = False
        self._last_trade_window = -1  # duplicate trade guard

    async def start(self):
        """Initialize and start the trading loop."""
        logger.info(f"Starting trading system in {self.mode} mode")

        # Load candles: try DB first (from backtest), then fetch fresh
        self._candles = self.db.get_candles(limit=500)
        logger.info(f"Loaded {len(self._candles)} candles from database")

        if len(self._candles) < 100:
            logger.info("Fetching candle data from Binance...")
            fresh = await self.binance.get_klines(limit=100)
            if not fresh.empty:
                self.db.upsert_candles(fresh)
                self._candles = self.db.get_candles(limit=500)

        # Train fair value model
        if len(self._candles) >= 50:
            self.ensemble.fair_value_model.train(self._candles, self._funding_rate)
            self._trained = True
            logger.info(f"Fair value model trained on {len(self._candles)} candles")

        # Get initial price
        self._btc_price = await self.binance.get_price()
        logger.info(f"BTC price: ${self._btc_price:,.2f}")

        # Main loop
        self.running = True
        await self._run_loop()

    async def _run_loop(self):
        """Main async event loop."""
        binance_interval = self.cfg["polling"]["binance_interval_sec"]
        kalshi_interval = self.cfg["polling"]["kalshi_interval_sec"]

        # Start dashboard in background
        dashboard_task = asyncio.create_task(self._dashboard_loop())

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

                # Check if we're on a 15-min boundary (within 30 sec)
                utc_now = datetime.now(timezone.utc)
                if utc_now.minute % 15 < 1 and utc_now.second < 30:
                    await self._evaluate_and_trade()

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        finally:
            dashboard_task.cancel()
            await self._shutdown()

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
                self._candles = self.db.get_candles(limit=100)

            self._funding_rate = await self.binance.get_funding_rate()

        except Exception as e:
            logger.error(f"Binance poll error: {e}")

    async def _poll_kalshi(self):
        """Fetch latest Kalshi market data (always uses real API for reads).

        For price-range bracket markets (KXBTC), finds the bracket closest
        to the current BTC price — that's the most actionable market.
        """
        try:
            # Dynamically find the nearest expiring KXBTC event
            event_ticker = self.kalshi_reader.get_nearest_event_ticker()
            if event_ticker and event_ticker != self._current_event_ticker:
                self._current_event_ticker = event_ticker
                logger.info(f"Tracking event: {event_ticker}")

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
            else:
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
            self._orderbook = self.kalshi_reader.get_orderbook(ticker)

            subtitle = self._current_market.get("subtitle", ticker)
            yes_bid = self._current_market.get("yes_bid", 0)
            yes_ask = self._current_market.get("yes_ask", 0)
            bracket_str = f"[{self._current_bracket[0]:,.0f}-{self._current_bracket[1]:,.0f}]" if self._current_bracket else "?"

            logger.info(
                f"Kalshi bracket: {bracket_str} {subtitle} | yes={yes_bid}/{yes_ask} | "
                f"{len(parsed)} parsed / {len(markets)} total brackets"
            )

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
        """Run ensemble strategy and place trade if warranted."""
        # Duplicate evaluation guard: only evaluate once per 15-min window
        utc_now = datetime.now(timezone.utc)
        window_id = utc_now.hour * 4 + utc_now.minute // 15
        if window_id == self._last_trade_window:
            return
        self._last_trade_window = window_id

        if self._candles is None or self._candles.empty:
            return

        if not self._trained:
            if len(self._candles) >= 50:
                self.ensemble.fair_value_model.train(self._candles, self._funding_rate)
                self._trained = True
            else:
                return

        # Build market data
        implied_prob = 0.5
        if self._current_market:
            yes_bid = self._current_market.get("yes_bid", 50)
            yes_ask = self._current_market.get("yes_ask", 50)
            implied_prob = (yes_bid + yes_ask) / 200  # midpoint in [0,1]

        # Compute bracket probability calibrated from market price
        bracket_prob = None
        if self._current_bracket and self._btc_price > 0:
            model_prob = self.ensemble.fair_value_model.predict(self._candles, self._funding_rate)
            bracket_prob = estimate_bracket_prob(
                self._btc_price,
                self._current_bracket[0],
                self._current_bracket[1],
                model_prob,
                implied_prob,
            )
            logger.info(
                f"Bracket prob: {bracket_prob:.4f} | market implied: {implied_prob:.4f} | "
                f"model P(up): {model_prob:.4f} | edge: {bracket_prob - implied_prob:.4f} | "
                f"bracket: [{self._current_bracket[0]:,.0f}-{self._current_bracket[1]:,.0f}]"
            )

        market_data = {
            "orderbook": self._orderbook or {},
            "funding_rate": self._funding_rate,
            "implied_prob": implied_prob,
            "bracket_prob": bracket_prob,
        }

        balance = self.kalshi.get_balance()
        result = self.ensemble.evaluate(self._candles, market_data, balance)

        # Update dashboard
        self.dashboard.update(
            current_edge=result.edge,
            regime=result.regime,
            last_signal=f"{result.direction.value} ({result.ensemble_prob:.2f})",
        )

        if not result.should_trade:
            return

        if not self._current_market:
            logger.warning("No active market to trade")
            return

        ticker = self._current_market["ticker"]
        price_cents = int(implied_prob * 100) if result.side == "yes" else int((1 - implied_prob) * 100)
        price_cents = max(1, min(99, price_cents))

        # Risk check
        allowed, reason = self.risk.check_order(result.side, price_cents, result.contracts, balance)
        if not allowed:
            logger.info(f"Trade blocked by risk: {reason}")
            return

        # Place order
        order = self.order_manager.place_order(
            ticker, result.side, price_cents, result.contracts,
            expiry_time_ms=self._current_market.get("close_time"),
        )

        if order and "error" not in order:
            self.trade_logger.log_trade(
                ticker, result.side, price_cents, result.contracts,
                "ensemble", result.edge, result.ensemble_prob, implied_prob,
            )

    async def _dashboard_loop(self):
        """Periodically update dashboard data."""
        try:
            while self.running:
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

                # Next expiry
                next_expiry = ""
                if self._current_market:
                    exp_ms = self._current_market.get("close_time", 0)
                    if exp_ms:
                        exp_dt = datetime.fromtimestamp(exp_ms / 1000, tz=timezone.utc)
                        remaining = (exp_dt - datetime.now(timezone.utc)).total_seconds()
                        next_expiry = f"{int(remaining // 60)}m {int(remaining % 60)}s"

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

                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        self.order_manager.cancel_all()
        self.order_manager.cleanup()
        await self.binance.close()
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
        if args.mode == "live":
            cfg["kalshi"]["base_url"] = cfg["kalshi"]["live_url"]

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
