"""Orderbook imbalance strategy: pure market microstructure signal."""

import pandas as pd

from .base import Direction, Signal, Strategy


class OrderbookImbalanceStrategy(Strategy):
    """Detects directional signal from orderbook volume imbalance.

    This strategy is purely based on market microstructure — it does NOT
    use the model's fair_value, ensuring it provides an independent signal
    for the ensemble's confirmation filter.
    """

    name = "orderbook_imbalance"

    def __init__(self, imbalance_threshold: float = 0.15):
        self.imbalance_threshold = imbalance_threshold

    def generate_signal(self, candles: pd.DataFrame, market_data: dict | None = None) -> Signal:
        if not market_data or "orderbook" not in market_data:
            return Signal(Direction.NONE, 0.0, self.name)

        orderbook = market_data["orderbook"]

        yes_bids = orderbook.get("yes", [])
        no_bids = orderbook.get("no", [])

        if not yes_bids and not no_bids:
            return Signal(Direction.NONE, 0.0, self.name)

        # Volume imbalance: positive = more buying interest on YES side
        yes_volume = sum(level[1] for level in yes_bids) if yes_bids else 0
        no_volume = sum(level[1] for level in no_bids) if no_bids else 0
        total_volume = yes_volume + no_volume

        if total_volume == 0:
            return Signal(Direction.NONE, 0.0, self.name)

        imbalance = (yes_volume - no_volume) / total_volume

        # Also look at implied prob deviation from 50/50
        implied_prob = market_data.get("implied_prob", 0.5)
        price_lean = implied_prob - 0.5  # positive = market leans UP

        features = {
            "imbalance": float(imbalance),
            "price_lean": float(price_lean),
            "yes_volume": int(yes_volume),
            "no_volume": int(no_volume),
        }

        # Strong YES-side imbalance AND price leans up -> UP signal
        if imbalance > self.imbalance_threshold and price_lean > 0.01:
            confidence = min(0.52 + abs(imbalance) * 0.15, 0.60)
            return Signal(Direction.UP, confidence, self.name, features)

        # Strong NO-side imbalance AND price leans down -> DOWN signal
        if imbalance < -self.imbalance_threshold and price_lean < -0.01:
            confidence = min(0.52 + abs(imbalance) * 0.15, 0.60)
            return Signal(Direction.DOWN, confidence, self.name, features)

        return Signal(Direction.NONE, 0.0, self.name, features)
