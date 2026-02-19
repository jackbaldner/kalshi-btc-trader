"""Ensemble strategy: model direction confirmed by at least one strategy signal."""

import logging

import pandas as pd

from ..models.fair_value import FairValueModel
from ..models.kelly import kelly_fraction, size_order
from .base import Direction, Signal, Strategy
from .funding_rate import FundingRateStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .orderbook_imbalance import OrderbookImbalanceStrategy
from .volatility_regime import VolatilityRegime

logger = logging.getLogger(__name__)

STRATEGY_CLASSES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "orderbook_imbalance": OrderbookImbalanceStrategy,
    "funding_rate": FundingRateStrategy,
}


class EnsembleResult:
    def __init__(self):
        self.direction: Direction = Direction.NONE
        self.ensemble_prob: float = 0.5
        self.model_prob: float = 0.5
        self.implied_prob: float = 0.5
        self.edge: float = 0.0
        self.kelly_f: float = 0.0
        self.contracts: int = 0
        self.side: str = ""
        self.signals: list[Signal] = []
        self.regime: str = "medium"
        self.should_trade: bool = False
        self.signal_count: int = 0
        self.confirming_strategies: list[str] = []


class EnsembleStrategy:
    """Model-driven with strategy confirmation.

    1. Model determines the directional lean (P(up) > 0.5 → UP, else DOWN)
    2. At least one strategy must independently confirm the direction
    3. The ensemble prob is the model prob boosted slightly by confirmations
    4. Only trade when edge vs market implied prob exceeds threshold
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.edge_threshold = cfg["strategy"]["edge_threshold"]
        self.kelly_multiplier = cfg["strategy"]["kelly_fraction"]
        self.max_position = cfg["risk"]["max_position_size"]
        self.min_confirmations = cfg["strategy"].get("min_agreeing_signals", 1)

        self.strategies: dict[str, Strategy] = {}
        active = cfg["strategy"]["active_strategies"]
        for name in active:
            if name in STRATEGY_CLASSES:
                if name == "momentum":
                    self.strategies[name] = STRATEGY_CLASSES[name](cfg["strategy"]["momentum_lookback"])
                elif name == "mean_reversion":
                    self.strategies[name] = STRATEGY_CLASSES[name](cfg["strategy"]["mean_reversion_threshold"])
                else:
                    self.strategies[name] = STRATEGY_CLASSES[name]()

        self.vol_regime = VolatilityRegime(cfg["strategy"]["vol_regime_window"])
        self.fair_value_model = FairValueModel(lookback=cfg["strategy"]["momentum_lookback"])

    def evaluate(self, candles: pd.DataFrame, market_data: dict, balance: float) -> EnsembleResult:
        result = EnsembleResult()

        regime = self.vol_regime.classify(candles)
        result.regime = regime

        # Model prediction
        funding = market_data.get("funding_rate", 0.0)
        result.model_prob = self.fair_value_model.predict(candles, funding)
        market_data["fair_value"] = result.model_prob

        # Model direction
        if result.model_prob > 0.5:
            model_dir = Direction.UP
        elif result.model_prob < 0.5:
            model_dir = Direction.DOWN
        else:
            result.should_trade = False
            return result

        # Collect strategy signals and find confirmations
        confirmations = []
        for name, strategy in self.strategies.items():
            signal = strategy.generate_signal(candles, market_data)
            result.signals.append(signal)

            # Strategy confirms model if it points same direction
            if signal.direction == model_dir and signal.confidence > 0:
                confirmations.append(signal)

        result.signal_count = len(confirmations)
        result.confirming_strategies = [s.strategy for s in confirmations]

        # Require minimum confirmations
        if len(confirmations) < self.min_confirmations:
            result.should_trade = False
            return result

        # Ensemble probability: model prob + confirmation boost
        # Each confirming signal adds a small nudge (1% per signal)
        confirmation_boost = sum(
            0.01 * (s.confidence - 0.5) * 2  # scale confidence to [-1, 1] range
            for s in confirmations
        )

        result.ensemble_prob = result.model_prob + confirmation_boost
        result.ensemble_prob = max(0.35, min(0.65, result.ensemble_prob))
        result.direction = model_dir

        # Market implied probability (bracket price from Kalshi)
        result.implied_prob = market_data.get("implied_prob", 0.5)

        # Bracket-based edge calculation
        # bracket_prob = our estimate of P(price in bracket) from normal CDF
        # implied_prob = market's price for that bracket
        bracket_prob = market_data.get("bracket_prob")

        if bracket_prob is not None:
            # Bracket mode: compare our bracket probability vs market bracket price
            our_p = bracket_prob
            market_p = result.implied_prob
            result.edge = our_p - market_p

            if result.edge > 0:
                result.side = "yes"  # our prob > market → buy yes on this bracket
            else:
                result.should_trade = False
                return result
        else:
            # Fallback: directional mode (no bracket info available)
            if result.direction == Direction.UP:
                result.side = "yes"
                our_p = result.ensemble_prob
                market_p = result.implied_prob
            else:
                result.side = "no"
                our_p = 1 - result.ensemble_prob
                market_p = 1 - result.implied_prob

            result.edge = our_p - market_p

        if result.edge < self.edge_threshold:
            result.should_trade = False
            return result

        # Kelly sizing
        result.kelly_f = kelly_fraction(our_p, market_p, self.kelly_multiplier)
        if result.kelly_f <= 0:
            result.should_trade = False
            return result

        result.contracts = size_order(result.kelly_f, balance, self.max_position, market_p)
        result.should_trade = result.contracts > 0

        if result.should_trade:
            logger.info(
                f"Ensemble: dir={result.direction.value} prob={result.ensemble_prob:.3f} "
                f"implied={result.implied_prob:.3f} edge={result.edge:.3f} "
                f"kelly={result.kelly_f:.3f} contracts={result.contracts} "
                f"confirmed_by={result.confirming_strategies} regime={regime}"
            )

        return result
