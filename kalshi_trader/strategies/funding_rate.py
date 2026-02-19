"""Binance perpetual funding rate signal strategy."""

import pandas as pd

from .base import Direction, Signal, Strategy


class FundingRateStrategy(Strategy):
    """Uses Binance perp funding rate as a contrarian signal.

    High positive funding -> market over-levered long -> slight bearish signal
    High negative funding -> market over-levered short -> slight bullish signal
    """

    name = "funding_rate"

    def __init__(self, threshold: float = 0.0005):
        self.threshold = threshold  # funding rate threshold to trigger signal

    def generate_signal(self, candles: pd.DataFrame, market_data: dict | None = None) -> Signal:
        if not market_data or "funding_rate" not in market_data:
            return Signal(Direction.NONE, 0.0, self.name)

        funding = market_data["funding_rate"]

        features = {
            "funding_rate": float(funding),
            "threshold": self.threshold,
        }

        # High positive funding -> contrarian bearish
        if funding > self.threshold:
            confidence = min(0.3 + abs(funding) * 200, 0.65)
            return Signal(Direction.DOWN, confidence, self.name, features)

        # High negative funding -> contrarian bullish
        if funding < -self.threshold:
            confidence = min(0.3 + abs(funding) * 200, 0.65)
            return Signal(Direction.UP, confidence, self.name, features)

        return Signal(Direction.NONE, 0.0, self.name, features)
