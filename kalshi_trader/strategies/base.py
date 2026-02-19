"""Abstract strategy interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class Signal:
    direction: Direction
    confidence: float  # 0 to 1
    strategy: str
    features: dict | None = None

    @property
    def is_trade(self) -> bool:
        return self.direction != Direction.NONE and self.confidence > 0


class Strategy(ABC):
    """Base class for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signal(self, candles: pd.DataFrame, market_data: dict | None = None) -> Signal:
        """Generate a trading signal from candle data and optional market data.

        Args:
            candles: DataFrame with columns: open, high, low, close, volume
            market_data: Optional dict with kalshi orderbook, funding rate, etc.

        Returns:
            Signal with direction, confidence, and strategy name.
        """
        ...
