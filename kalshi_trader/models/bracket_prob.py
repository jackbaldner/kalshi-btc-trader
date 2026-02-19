"""Convert directional P(up) model output to bracket probability P(price in range).

Kalshi KXBTC markets are price-range brackets (e.g. "$68,750 to $69,249.99"),
not directional up/down. The model outputs P(BTC up) which must be mapped to
P(price lands in a specific $500 bracket) using a normal distribution.
"""

import logging
import re

from scipy.stats import norm

logger = logging.getLogger(__name__)


def parse_bracket_bounds(market: dict) -> tuple[float, float] | None:
    """Extract (low, high) price bounds from a Kalshi bracket market.

    Tries subtitle first (e.g. "$68,750 to $69,249.99"),
    then falls back to ticker parsing (e.g. "KXBTC-26FEB2017-B69000").

    Returns None if bounds cannot be parsed.
    """
    # Try subtitle: "$68,750 to $69,249.99" or "$68,750.00 to $69,249.99"
    subtitle = market.get("subtitle", "")
    if subtitle:
        pattern = r"\$?([\d,]+(?:\.\d+)?)\s+to\s+\$?([\d,]+(?:\.\d+)?)"
        m = re.search(pattern, subtitle, re.IGNORECASE)
        if m:
            low = float(m.group(1).replace(",", ""))
            high = float(m.group(2).replace(",", ""))
            return (low, high)

    # Try ticker: KXBTC-26FEB2017-B69000
    ticker = market.get("ticker", "")
    if ticker:
        m = re.search(r"-B(\d+)$", ticker)
        if m:
            low = float(m.group(1))
            high = low + 500.0  # standard $500 bracket
            return (low, high)

    return None


def estimate_bracket_prob(
    current_price: float,
    bracket_low: float,
    bracket_high: float,
    model_p_up: float,
    vol_15m: float,
) -> float:
    """Estimate probability that BTC price lands in [bracket_low, bracket_high].

    Converts the directional P(up) signal into an implied drift, then uses a
    normal distribution to compute the probability of landing in the bracket.

    Args:
        current_price: Current BTC spot price.
        bracket_low: Lower bound of the bracket.
        bracket_high: Upper bound of the bracket.
        model_p_up: Model's P(BTC goes up) in [0, 1].
        vol_15m: 15-minute return volatility (std dev of log returns).

    Returns:
        Probability in [0, 1] that price lands in the bracket.
    """
    if vol_15m < 0.003:
        vol_15m = 0.003  # floor ~0.3% per 15min (realistic for BTC)

    # Convert P(up) to implied drift
    # If model_p_up == 0.5, drift is 0 (no directional view)
    # norm.ppf(0.53) ≈ 0.075, so small edge → small drift
    drift = norm.ppf(max(0.01, min(0.99, model_p_up))) * vol_15m

    # Expected price and std dev under normal model
    mu = current_price * (1 + drift)
    sigma = current_price * vol_15m

    if sigma <= 0:
        sigma = current_price * 0.003

    prob = norm.cdf(bracket_high, mu, sigma) - norm.cdf(bracket_low, mu, sigma)
    return float(max(0.0, min(1.0, prob)))
