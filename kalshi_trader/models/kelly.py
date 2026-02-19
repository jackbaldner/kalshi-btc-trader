"""Fractional Kelly criterion for position sizing."""

import logging

logger = logging.getLogger(__name__)


def kelly_fraction(model_prob: float, implied_prob: float, fraction: float = 0.25) -> float:
    """Calculate fractional Kelly bet size.

    Args:
        model_prob: Our estimated probability of winning (P(up) or P(down) depending on side)
        implied_prob: Market implied probability (from Kalshi price in cents / 100)
        fraction: Kelly fraction multiplier (default 0.25 = quarter Kelly)

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
    dollar_amount = min(dollar_amount, max_position)

    contracts = int(dollar_amount / contract_price)
    return max(contracts, 0)
