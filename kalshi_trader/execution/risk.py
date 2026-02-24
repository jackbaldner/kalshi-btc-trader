"""Risk management: position limits, exposure caps, daily loss limit."""

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces position limits, max exposure, and daily loss limits."""

    def __init__(self, cfg: dict, database):
        self.max_position_size = cfg["risk"]["max_position_size"]
        self.max_total_exposure = cfg["risk"]["max_total_exposure"]
        self.max_daily_loss = cfg["risk"]["max_daily_loss"]
        self.db = database
        self._current_exposure = 0.0

    def check_order(self, side: str, price_cents: int, count: int, balance: float) -> tuple[bool, str]:
        """Check if an order passes risk checks.

        Returns:
            (allowed, reason) tuple
        """
        cost = price_cents * count / 100  # cents -> dollars

        # Check position size
        if cost > self.max_position_size:
            return False, f"Position size ${cost:.2f} exceeds max ${self.max_position_size}"

        # Check total exposure
        if self._current_exposure + cost > self.max_total_exposure:
            return False, (
                f"Total exposure ${self._current_exposure + cost:.2f} "
                f"exceeds max ${self.max_total_exposure}"
            )

        # Check daily loss limit
        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_ts = int(day_start.timestamp() * 1000)
        daily_pnl = self.db.get_daily_pnl(day_start_ts)
        if daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss ${abs(daily_pnl):.2f} exceeds limit ${self.max_daily_loss}"

        # Check balance
        if cost > balance:
            return False, f"Insufficient balance: need ${cost:.2f}, have ${balance:.2f}"

        return True, "ok"

    def update_exposure(self, positions: list[dict]):
        """Recalculate current exposure from positions.

        Handles both live Kalshi API format (market_exposure in cents)
        and mock client format (yes_count/avg_yes_price).
        """
        total = 0.0
        for pos in positions:
            # Live API: market_exposure is in cents
            if "market_exposure" in pos:
                total += pos["market_exposure"] / 100
            else:
                # Mock client format
                yes_cost = pos.get("yes_count", 0) * pos.get("avg_yes_price", 0) / 100
                no_cost = pos.get("no_count", 0) * pos.get("avg_no_price", 0) / 100
                total += yes_cost + no_cost
        self._current_exposure = total

    @property
    def current_exposure(self) -> float:
        return self._current_exposure
