"""Convert directional P(up) model output to bracket probability P(price in range).

Kalshi KXBTC markets are price-range brackets (e.g. "$68,750 to $69,249.99"),
not directional up/down. The model outputs P(BTC up) which must be mapped to
P(price lands in a specific $500 bracket) using a normal distribution.

Key insight: we calibrate our vol from the market's bracket price so that edge
comes from our directional signal, not from a vol mismatch.
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
            logger.warning(
                f"Using hardcoded $500 bracket width for {ticker} — "
                f"subtitle parse failed, actual width may differ"
            )
            return (low, high)

    return None


def implied_vol_from_bracket_price(
    current_price: float,
    bracket_low: float,
    bracket_high: float,
    market_prob: float,
) -> float:
    """Back-calculate the implied vol from the market's bracket price.

    Uses bisection to find sigma such that:
        norm.cdf(high, price, sigma) - norm.cdf(low, price, sigma) = market_prob

    Returns vol as a fraction (e.g. 0.027 for 2.7%).
    """
    if market_prob <= 0.01 or market_prob >= 0.99:
        return 0.03  # fallback ~3%

    # Bisection on sigma (in dollar terms)
    lo_sigma = current_price * 0.0005  # 0.05% vol
    hi_sigma = current_price * 0.10    # 10% vol

    for _ in range(50):
        mid_sigma = (lo_sigma + hi_sigma) / 2
        p = norm.cdf(bracket_high, current_price, mid_sigma) - norm.cdf(bracket_low, current_price, mid_sigma)
        if p > market_prob:
            lo_sigma = mid_sigma  # sigma too small → prob too high → increase sigma
        else:
            hi_sigma = mid_sigma

    return (lo_sigma + hi_sigma) / 2 / current_price


def estimate_bracket_prob(
    current_price: float,
    bracket_low: float,
    bracket_high: float,
    model_p_up: float,
    market_prob: float,
) -> float:
    """Estimate probability that BTC price lands in [bracket_low, bracket_high].

    Calibrates vol from the market's bracket price, then applies the model's
    directional view as a drift. Edge comes from signal, not vol mismatch.

    Args:
        current_price: Current BTC spot price.
        bracket_low: Lower bound of the bracket.
        bracket_high: Upper bound of the bracket.
        model_p_up: Model's P(BTC goes up) in [0, 1].
        market_prob: Market's bracket price as probability (e.g. 0.11).

    Returns:
        Probability in [0, 1] that price lands in the bracket.
    """
    # Calibrate vol from market price
    vol = implied_vol_from_bracket_price(current_price, bracket_low, bracket_high, market_prob)

    # Convert P(up) to implied drift
    # If model_p_up == 0.5, drift is 0 → our prob ≈ market_prob (no edge)
    # Small deviations from 0.5 create small drift → small edge
    drift = norm.ppf(max(0.01, min(0.99, model_p_up))) * vol

    # Expected price and std dev under our model
    mu = current_price * (1 + drift)
    sigma = current_price * vol

    if sigma <= 0:
        return market_prob

    prob = norm.cdf(bracket_high, mu, sigma) - norm.cdf(bracket_low, mu, sigma)
    return float(max(0.0, min(1.0, prob)))


def estimate_bracket_prob_from_vol(
    current_price: float,
    bracket_low: float,
    bracket_high: float,
    predicted_vol: float,
    model_p_up: float = 0.5,
) -> float:
    """Estimate bracket probability using our predicted volatility.

    Unlike estimate_bracket_prob() which calibrates from market price,
    this uses our own vol prediction. If predicted_vol < implied_vol,
    we'll get a higher bracket prob → positive edge → buy YES.

    Applies a tail-risk adjustment: far-OTM brackets get their probability
    shrunk toward the market because the normal distribution underestimates
    tail risk (BTC has fat tails), making far-OTM brackets look falsely cheap.

    Args:
        current_price: Current BTC spot price.
        bracket_low: Lower bound of the bracket.
        bracket_high: Upper bound of the bracket.
        predicted_vol: Our predicted next-candle vol (absolute return).
        model_p_up: Optional directional lean from P(up) model.

    Returns:
        Probability in [0, 1] that price lands in the bracket.
    """
    if predicted_vol <= 0:
        predicted_vol = 0.003

    # Small directional drift from P(up) model
    drift = norm.ppf(max(0.01, min(0.99, model_p_up))) * predicted_vol

    mu = current_price * (1 + drift)
    sigma = current_price * predicted_vol

    if sigma <= 0:
        return 0.5

    prob = norm.cdf(bracket_high, mu, sigma) - norm.cdf(bracket_low, mu, sigma)
    prob = float(max(0.0, min(1.0, prob)))

    # Tail-risk adjustment: penalize confidence on far-OTM brackets.
    # The normal distribution underestimates tail probability for BTC.
    # For brackets far from current price, shrink our prob estimate toward
    # a conservative baseline (reduces false edge on cheap OTM brackets).
    bracket_mid = (bracket_low + bracket_high) / 2
    distance = abs(bracket_mid - current_price) / current_price  # as fraction
    # distance_in_sigmas: how many predicted-vol units away is this bracket
    distance_in_sigmas = distance / max(predicted_vol, 0.0005)

    if distance_in_sigmas > 1.0:
        # Shrink our prob toward zero for far-out brackets.
        # At 1 sigma: no penalty. At 2+ sigma: heavy penalty.
        # trust_factor decays from 1.0 at 1σ to ~0.3 at 3σ
        trust_factor = 1.0 / (1.0 + 0.5 * (distance_in_sigmas - 1.0))
        adjusted_prob = prob * trust_factor
        if adjusted_prob != prob:
            logger.debug(
                f"Tail-risk adjustment: bracket [{bracket_low:,.0f}-{bracket_high:,.0f}] "
                f"distance={distance:.4f} ({distance_in_sigmas:.1f}σ) "
                f"prob {prob:.4f} -> {adjusted_prob:.4f} (trust={trust_factor:.2f})"
            )
        prob = adjusted_prob

    return prob
