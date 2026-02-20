"""Simulate Kalshi bracket market prices from BTC candle data.

The simulator models a semi-efficient bracket market: the implied probability
represents P(price lands in a $500 bracket), not P(up). Edge must be genuinely
earned by the model seeing something the market doesn't fully price in.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


class MarketSimulator:
    """Models Kalshi bracket implied probability from BTC price data for backtesting.

    Simulates a $500 bracket market centered on the current price.
    The simulated market is partially efficient:
    - It uses a normal model with the actual next-candle outcome as a lean
    - It adds calibrated noise so the market isn't perfectly efficient
    - Bracket probabilities are typically 5-15% (realistic for $500 ranges)
    """

    # Minimum realistic 15-min BTC vol (~0.3%).  Real markets price in jump
    # risk and fat tails, so observed vol from calm periods under-states the
    # effective vol the market uses.  This floor prevents bracket_prob from
    # approaching 1.0 during low-vol stretches.
    MIN_VOL = 0.003

    def __init__(self, noise_std: float = 0.08, lookback: int = 20, efficiency: float = 0.05,
                 bracket_width: float = 500.0):
        self.noise_std = noise_std
        self.lookback = lookback
        self.efficiency = efficiency
        self.bracket_width = bracket_width

    def estimate_vol(self, candles: pd.DataFrame, idx: int) -> float:
        """Estimate return volatility from the last `lookback` candles at idx."""
        start = max(0, idx - self.lookback)
        closes = candles.iloc[start : idx + 1]["close"].values
        if len(closes) < 3:
            return self.MIN_VOL
        returns = np.diff(closes) / closes[:-1]
        vol = float(np.std(returns))
        return max(vol, self.MIN_VOL)

    def get_bracket_bounds(self, current_price: float) -> tuple[float, float]:
        """Return (low, high) for a $500 bracket centered on current price."""
        center = round(current_price / self.bracket_width) * self.bracket_width
        return (center - self.bracket_width / 2, center + self.bracket_width / 2)

    def simulate_implied_prob(self, candles: pd.DataFrame, idx: int,
                              rng: np.random.Generator | None = None) -> float:
        """Simulate Kalshi implied P(price in bracket) at a given candle index.

        The market view is a blend of:
        1. Recent volatility and price level (public information)
        2. A small lean toward the actual outcome (smart money / efficiency)
        3. Noise (market microstructure, disagreement)
        """
        if rng is None:
            rng = np.random.default_rng()

        if idx < self.lookback + 1:
            return 0.08  # default bracket prob

        current_price = candles.iloc[idx]["close"]
        bracket_low, bracket_high = self.get_bracket_bounds(current_price)

        # Estimate recent volatility from returns (same window as engine)
        vol = self.estimate_vol(candles, idx)
        if vol < self.MIN_VOL:
            vol = self.MIN_VOL

        # Efficiency component: market partially knows where price ends up
        drift = 0.0
        if idx + 1 < len(candles):
            next_price = candles.iloc[idx + 1]["close"]
            actual_move = (next_price - current_price) / current_price
            drift = actual_move * self.efficiency

        # Market's estimate of next-candle price distribution
        mu = current_price * (1 + drift)
        sigma = current_price * vol

        # Base bracket probability from normal CDF
        base_prob = norm.cdf(bracket_high, mu, sigma) - norm.cdf(bracket_low, mu, sigma)

        # Add noise (market disagreement)
        noise = rng.normal(0, self.noise_std * 0.1)  # scaled down for bracket probs
        noisy_prob = base_prob + noise

        return float(np.clip(noisy_prob, 0.01, 0.99))

    def simulate_orderbook(self, implied_prob: float,
                           rng: np.random.Generator | None = None) -> dict:
        """Generate a synthetic orderbook around the implied probability."""
        if rng is None:
            rng = np.random.default_rng()

        yes_price = int(implied_prob * 100)
        spread = rng.integers(2, 5)

        yes_bid = max(1, yes_price - spread)
        yes_ask = min(99, yes_price + spread)

        # Random depth
        bid_depth = [
            [yes_bid, int(rng.integers(5, 30))],
            [max(1, yes_bid - 1), int(rng.integers(10, 50))],
        ]
        ask_depth = [
            [100 - yes_ask, int(rng.integers(5, 30))],
            [min(99, 100 - yes_ask + 1), int(rng.integers(10, 50))],
        ]

        return {"yes": bid_depth, "no": ask_depth}

    def get_market_result(self, candles: pd.DataFrame, idx: int) -> str:
        """Determine if the next candle's price lands in the bracket."""
        if idx < 1 or idx + 1 >= len(candles):
            return "no"
        current_price = candles.iloc[idx]["close"]
        next_price = candles.iloc[idx + 1]["close"]
        bracket_low, bracket_high = self.get_bracket_bounds(current_price)
        if bracket_low <= next_price < bracket_high:
            return "yes"
        return "no"
