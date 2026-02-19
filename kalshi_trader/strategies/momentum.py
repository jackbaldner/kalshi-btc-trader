"""N-candle momentum continuation strategy."""

import numpy as np
import pandas as pd

from .base import Direction, Signal, Strategy


class MomentumStrategy(Strategy):
    """Bets on continuation when ALL N recent candles move in the same direction.

    Only fires on strong consensus (all candles agree) — no partial matches.
    """

    name = "momentum"

    def __init__(self, lookback: int = 4):
        self.lookback = lookback

    def generate_signal(self, candles: pd.DataFrame, market_data: dict | None = None) -> Signal:
        if len(candles) < self.lookback + 1:
            return Signal(Direction.NONE, 0.0, self.name)

        recent = candles.tail(self.lookback + 1)
        returns = recent["close"].pct_change().dropna()

        if len(returns) < self.lookback:
            return Signal(Direction.NONE, 0.0, self.name)

        up_count = (returns > 0).sum()
        down_count = (returns < 0).sum()
        total = len(returns)
        avg_ret = float(returns.mean())

        features = {
            "up_count": int(up_count),
            "down_count": int(down_count),
            "avg_return": avg_ret,
            "cum_return": float(returns.sum()),
        }

        # Only fire on UNANIMOUS direction (all candles agree)
        if up_count == total:
            confidence = min(0.54 + abs(avg_ret) * 30, 0.62)
            return Signal(Direction.UP, confidence, self.name, features)
        elif down_count == total:
            confidence = min(0.54 + abs(avg_ret) * 30, 0.62)
            return Signal(Direction.DOWN, confidence, self.name, features)

        return Signal(Direction.NONE, 0.0, self.name, features)
