"""Backtest engine: runs strategies against historical BTC data with simulated Kalshi markets."""

import asyncio
import logging
import sys

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.binance import BinanceClient
from ..data.database import Database
from ..models.bracket_prob import estimate_bracket_prob_from_vol, implied_vol_from_bracket_price
from ..models.vol_model import VolModel
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
            lookback=20,
            efficiency=0.05,
        )
        self.ensemble = EnsembleStrategy(cfg)
        self.vol_model = VolModel(lookback=cfg["strategy"].get("vol_model_lookback", 20))

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

        # Train models on first 60% of data
        split_idx = int(len(candles) * 0.6)
        train_data = candles.iloc[:split_idx].copy()
        test_data = candles.iloc[split_idx:].copy()

        logger.info(f"Training on {len(train_data)} candles, testing on {len(test_data)} candles")
        self.ensemble.fair_value_model.train(train_data)
        self.vol_model.train(train_data)

        trades = []
        balance = 10000.0
        lookback_window = 50  # candles to feed strategies

        # Track vol prediction accuracy
        vol_preds = []
        vol_actuals = []

        for i in range(lookback_window, len(test_data) - 1):
            # Slice window of candles for strategies
            window = test_data.iloc[i - lookback_window : i + 1].copy().reset_index(drop=True)

            # Simulate Kalshi bracket market
            global_idx = split_idx + i
            current_price = test_data.iloc[i]["close"]
            bracket_low, bracket_high = self.simulator.get_bracket_bounds(current_price)

            implied_prob = self.simulator.simulate_implied_prob(candles, global_idx, rng)
            orderbook = self.simulator.simulate_orderbook(implied_prob, rng)

            # Vol prediction: our model vs market implied
            raw_pred_vol = self.vol_model.predict(window, 0.0)
            implied_vol = implied_vol_from_bracket_price(
                current_price, bracket_low, bracket_high, implied_prob,
            )
            predicted_vol = self.vol_model.blend_with_implied(raw_pred_vol, implied_vol)

            # Track prediction accuracy
            actual_vol = abs(
                (test_data.iloc[i + 1]["close"] - current_price) / current_price
            )
            vol_preds.append(raw_pred_vol)
            vol_actuals.append(actual_vol)

            # Bracket prob from our blended vol prediction
            model_prob = self.ensemble.fair_value_model.predict(window, 0.0)
            bracket_prob = estimate_bracket_prob_from_vol(
                current_price, bracket_low, bracket_high, predicted_vol, model_prob,
            )

            market_data = {
                "orderbook": orderbook,
                "funding_rate": 0.0,
                "implied_prob": implied_prob,
                "bracket_prob": bracket_prob,
                "predicted_vol": predicted_vol,
                "implied_vol": implied_vol,
            }

            # Run ensemble
            result = self.ensemble.evaluate(window, market_data, balance)

            if not result.should_trade:
                continue

            # Simulate trade execution
            if result.side == "yes":
                price_cents = int(implied_prob * 100)
            else:
                price_cents = int((1 - implied_prob) * 100)
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
            if result.side == "yes":
                won = actual_result == "yes"
            else:
                won = actual_result == "no"

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
                "predicted_vol": predicted_vol,
                "implied_vol": implied_vol,
            })

        # Vol prediction accuracy
        if vol_preds:
            vp = np.array(vol_preds)
            va = np.array(vol_actuals)
            corr = float(np.corrcoef(vp, va)[0, 1])
            mae = float(np.mean(np.abs(vp - va)))
            logger.info(
                f"Vol prediction: corr={corr:.3f}, MAE={mae:.5f}, "
                f"median_pred={np.median(vp):.5f}, median_actual={np.median(va):.5f}"
            )

        metrics = calculate_metrics(trades)
        metrics["final_balance"] = balance
        metrics["trades"] = trades
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
