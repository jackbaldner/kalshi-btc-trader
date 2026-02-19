"""Backtest engine: runs strategies against historical BTC data with simulated Kalshi markets."""

import asyncio
import logging
import sys

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.binance import BinanceClient
from ..data.database import Database
from ..models.bracket_prob import estimate_bracket_prob
from ..strategies.ensemble import EnsembleStrategy
from .market_simulator import MarketSimulator
from .metrics import calculate_metrics, print_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Runs backtest of ensemble strategy on historical data."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.simulator = MarketSimulator(
            noise_std=0.08,
            lookback=6,
            efficiency=0.3,
        )
        self.ensemble = EnsembleStrategy(cfg)

    def run(self, candles: pd.DataFrame, seed: int = 42) -> dict:
        """Run backtest on historical candles.

        Args:
            candles: DataFrame with OHLCV columns, sorted by time
            seed: Random seed for reproducibility

        Returns:
            Dict of performance metrics
        """
        rng = np.random.default_rng(seed)

        if len(candles) < 100:
            logger.error("Need at least 100 candles for backtest")
            return calculate_metrics([])

        # Train fair value model on first 60% of data
        split_idx = int(len(candles) * 0.6)
        train_data = candles.iloc[:split_idx].copy()
        test_data = candles.iloc[split_idx:].copy()

        logger.info(f"Training on {len(train_data)} candles, testing on {len(test_data)} candles")
        self.ensemble.fair_value_model.train(train_data)

        trades = []
        balance = 10000.0
        lookback_window = 50  # candles to feed strategies

        for i in range(lookback_window, len(test_data) - 1):
            # Slice window of candles for strategies
            window = test_data.iloc[i - lookback_window : i + 1].copy().reset_index(drop=True)

            # Simulate Kalshi bracket market
            global_idx = split_idx + i
            current_price = test_data.iloc[i]["close"]
            bracket_low, bracket_high = self.simulator.get_bracket_bounds(current_price)

            implied_prob = self.simulator.simulate_implied_prob(candles, global_idx, rng)
            orderbook = self.simulator.simulate_orderbook(implied_prob, rng)

            # Calibrate vol from market bracket price so edge comes from
            # model signal, not vol mismatch
            model_prob = self.ensemble.fair_value_model.predict(window, 0.0)
            bracket_prob = estimate_bracket_prob(
                current_price, bracket_low, bracket_high, model_prob, implied_prob,
            )

            market_data = {
                "orderbook": orderbook,
                "funding_rate": 0.0,
                "implied_prob": implied_prob,
                "bracket_prob": bracket_prob,
            }

            # Run ensemble
            result = self.ensemble.evaluate(window, market_data, balance)

            if not result.should_trade:
                continue

            # Simulate trade execution (bracket markets: buy yes on bracket)
            price_cents = int(implied_prob * 100)
            price_cents = max(1, min(99, price_cents))

            # Cap contracts so we don't exceed balance
            max_affordable = int(balance / (price_cents / 100))
            contracts = min(result.contracts, max_affordable)
            if contracts <= 0:
                continue

            cost = price_cents * contracts / 100  # total cost in dollars
            balance -= cost

            # Resolve: did next candle's price land in the bracket?
            actual_result = self.simulator.get_market_result(test_data, i)
            won = (result.side == "yes" and actual_result == "yes")

            if won:
                payout = contracts * 1.0  # $1 per contract
                pnl = payout - cost
                balance += payout
            else:
                pnl = -cost

            trades.append({
                "pnl": pnl,
                "edge": result.edge,
                "model_prob": bracket_prob,
                "implied_prob": implied_prob,
                "strategy": "ensemble",
                "side": result.side,
                "contracts": contracts,
                "price": price_cents,
            })

        metrics = calculate_metrics(trades)
        metrics["final_balance"] = balance
        return metrics


async def fetch_and_run():
    """Fetch historical data from Binance and run backtest."""
    cfg = load_config()
    db = Database(cfg["database"]["path"])

    # Try loading from DB first
    candles = db.get_candles(limit=10000)

    if len(candles) < 500:
        logger.info("Fetching historical candles from Binance...")
        client = BinanceClient(cfg)
        try:
            candles = await client.get_historical_klines(days=90)
            db.upsert_candles(candles)
            logger.info(f"Saved {len(candles)} candles to database")
        finally:
            await client.close()

    logger.info(f"Running backtest on {len(candles)} candles")
    engine = BacktestEngine(cfg)
    metrics = engine.run(candles)

    print_metrics(metrics)
    print(f"\nFinal balance: ${metrics.get('final_balance', 0):.2f}")

    db.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(fetch_and_run())


if __name__ == "__main__":
    main()
