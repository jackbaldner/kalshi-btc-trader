"""Fractional Kelly criterion for position sizing."""

import logging

logger = logging.getLogger(__name__)


def kelly_ramp(edge: float, edge_threshold: float = 0.02, full_kelly_edge: float = 0.05) -> float:
    """Scale Kelly fraction based on edge size.

    Quarter Kelly at edge_threshold, linearly ramping to full Kelly at full_kelly_edge.
    This prevents over-betting on thin edges while allowing full sizing on strong ones.

    Returns a multiplier in [0.25, 1.0].
    """
    if edge <= 0:
        return 0.0
    if edge >= full_kelly_edge:
        return 1.0
    if edge <= edge_threshold:
        return 0.25
    # Linear ramp from 0.25 at edge_threshold to 1.0 at full_kelly_edge
    t = (edge - edge_threshold) / (full_kelly_edge - edge_threshold)
    return 0.25 + 0.75 * t


def kelly_fraction(model_prob: float, implied_prob: float, fraction: float = 1.0) -> float:
    """Calculate fractional Kelly bet size.

    Args:
        model_prob: Our estimated probability of winning (P(up) or P(down) depending on side)
        implied_prob: Market implied probability (from Kalshi price in cents / 100)
        fraction: Kelly fraction multiplier (default 1.0 = full Kelly)

    Returns:
        Fraction of bankroll to bet (0 to 1). Returns 0 if no edge.
    """
    if implied_prob <= 0 or implied_prob >= 1:
        return 0.0
    if model_prob <= 0 or model_prob >= 1:
        return 0.0

    # Binary contract: pay implied_prob to win (1 - implied_prob) if correct
    # Payout odds: b = (1 - implied_prob) / implied_prob
    b = (1.0 - implied_prob) / implied_prob
    p = model_prob
    q = 1.0 - p

    # Kelly: f* = (p * b - q) / b
    f_star = (p * b - q) / b

    if f_star <= 0:
        return 0.0

    return min(f_star * fraction, 1.0)


def size_order(kelly_f: float, balance: float, max_position: float, contract_price: float) -> int:
    """Convert Kelly fraction to number of contracts.

    Args:
        kelly_f: Kelly fraction (0 to 1)
        balance: Current account balance in dollars
        max_position: Maximum position size in dollars
        contract_price: Price per contract in dollars (cents/100)

    Returns:
        Number of contracts to buy (integer, >= 0)
    """
    if kelly_f <= 0 or contract_price <= 0:
        return 0

    dollar_amount = kelly_f * balance
    if dollar_amount > max_position:
        logger.info(
            f"Kelly size ${dollar_amount:.2f} capped by max_position ${max_position:.2f}"
        )
        dollar_amount = max_position

    contracts = int(dollar_amount / contract_price)
    return max(contracts, 0)
