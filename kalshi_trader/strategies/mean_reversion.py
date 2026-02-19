"""Mean reversion strategy: fade large moves expecting reversal."""

import numpy as np
import pandas as pd

from .base import Direction, Signal, Strategy


class MeanReversionStrategy(Strategy):
    """Bets against large single-candle moves, expecting reversion to mean."""

    name = "mean_reversion"

    def __init__(self, threshold: float = 0.005, vol_window: int = 20):
        self.threshold = threshold  # minimum move to trigger (0.5%)
        self.vol_window = vol_window

    def generate_signal(self, candles: pd.DataFrame, market_data: dict | None = None) -> Signal:
        if len(candles) < self.vol_window + 1:
            return Signal(Direction.NONE, 0.0, self.name)

        returns = candles["close"].pct_change()
        last_return = returns.iloc[-1]
        rolling_vol = returns.tail(self.vol_window).std()

        if rolling_vol == 0 or np.isnan(rolling_vol):
            return Signal(Direction.NONE, 0.0, self.name)

        z_score = last_return / rolling_vol

        features = {
            "last_return": float(last_return),
            "rolling_vol": float(rolling_vol),
            "z_score": float(z_score),
        }

        # Only trigger on genuinely extreme moves (z > 2.0)
        if abs(z_score) < 2.0:
            return Signal(Direction.NONE, 0.0, self.name, features)

        # Large move up -> fade (bet DOWN for reversion)
        if last_return > self.threshold:
            confidence = min(0.52 + abs(z_score) * 0.03, 0.62)
            return Signal(Direction.DOWN, confidence, self.name, features)

        # Large move down -> fade (bet UP for reversion)
        if last_return < -self.threshold:
            confidence = min(0.52 + abs(z_score) * 0.03, 0.62)
            return Signal(Direction.UP, confidence, self.name, features)

        return Signal(Direction.NONE, 0.0, self.name, features)
