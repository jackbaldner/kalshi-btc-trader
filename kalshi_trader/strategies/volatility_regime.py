"""Volatility regime detector: switches between momentum and mean-reversion weighting."""

import numpy as np
import pandas as pd

from .base import Direction, Signal, Strategy


class VolatilityRegime:
    """Classifies the current volatility regime to weight strategies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def __init__(self, window: int = 20):
        self.window = window

    def classify(self, candles: pd.DataFrame) -> str:
        """Classify current vol regime based on rolling percentile."""
        if len(candles) < self.window * 2:
            return self.MEDIUM

        returns = candles["close"].pct_change().dropna()
        current_vol = returns.tail(self.window).std()
        long_vol = returns.std()

        if long_vol == 0:
            return self.MEDIUM

        ratio = current_vol / long_vol

        if ratio < 0.7:
            return self.LOW
        elif ratio > 1.3:
            return self.HIGH
        return self.MEDIUM

    def strategy_weights(self, regime: str) -> dict[str, float]:
        """Return strategy weights based on vol regime.

        Low vol -> favor momentum (trends persist)
        High vol -> favor mean reversion (overreactions)
        """
        if regime == self.LOW:
            return {
                "momentum": 0.4,
                "mean_reversion": 0.15,
                "orderbook_imbalance": 0.25,
                "funding_rate": 0.2,
            }
        elif regime == self.HIGH:
            return {
                "momentum": 0.15,
                "mean_reversion": 0.4,
                "orderbook_imbalance": 0.25,
                "funding_rate": 0.2,
            }
        else:  # MEDIUM
            return {
                "momentum": 0.25,
                "mean_reversion": 0.25,
                "orderbook_imbalance": 0.3,
                "funding_rate": 0.2,
            }


class VolatilityRegimeStrategy(Strategy):
    """Strategy that adapts based on volatility regime."""

    name = "volatility_regime"

    def __init__(self, window: int = 20):
        self.regime = VolatilityRegime(window)

    def generate_signal(self, candles: pd.DataFrame, market_data: dict | None = None) -> Signal:
        """Returns regime classification as signal metadata (no direct trades)."""
        current_regime = self.regime.classify(candles)
        returns = candles["close"].pct_change().dropna()
        current_vol = float(returns.tail(self.regime.window).std()) if len(returns) >= self.regime.window else 0.0

        features = {
            "regime": current_regime,
            "current_vol": current_vol,
            "weights": self.regime.strategy_weights(current_regime),
        }

        return Signal(Direction.NONE, 0.0, self.name, features)
