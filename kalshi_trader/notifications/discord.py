"""Discord webhook notifications for trade alerts.

All functions are fire-and-forget: they log errors but never raise,
so they cannot crash or block the trading bot.
"""

import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Module-level webhook URL, set once at startup via configure()
_webhook_url: str | None = None


def configure(webhook_url: str | None) -> None:
    """Set the Discord webhook URL. Call once at startup."""
    global _webhook_url
    _webhook_url = webhook_url
    if _webhook_url:
        logger.info("Discord notifications enabled")
    else:
        logger.info("Discord notifications disabled (no webhook URL configured)")


def _post_embed(embed: dict) -> None:
    """POST a single embed to the Discord webhook. Fire-and-forget."""
    if not _webhook_url:
        return
    try:
        payload = {"embeds": [embed]}
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(_webhook_url, json=payload)
            if resp.status_code not in (200, 204):
                logger.warning(f"Discord webhook returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Discord webhook failed: {e}")


def send_trade_alert(
    *,
    ticker: str,
    side: str,
    price: int,
    size: int,
    edge: float,
    bracket_low: float,
    bracket_high: float,
    balance: float,
    predicted_vol: float | None = None,
    implied_vol: float | None = None,
) -> None:
    """Send a rich embed for a filled trade."""
    if not _webhook_url:
        return
    try:
        color = 0x2ECC71 if side == "yes" else 0xE74C3C  # green / red
        cost = price * size / 100.0
        if side == "yes":
            potential_profit = size * (100 - price) / 100.0
        else:
            potential_profit = size * (100 - (100 - price)) / 100.0

        vol_str = ""
        if predicted_vol is not None and implied_vol is not None:
            vol_str = f"{predicted_vol:.5f} vs {implied_vol:.5f}"
        elif predicted_vol is not None:
            vol_str = f"{predicted_vol:.5f}"

        embed = {
            "title": f"Trade Filled — {side.upper()}",
            "color": color,
            "fields": [
                {"name": "Bracket", "value": f"[{bracket_low:,.0f} - {bracket_high:,.0f}]", "inline": True},
                {"name": "Ticker", "value": ticker, "inline": True},
                {"name": "Price", "value": f"{price}c", "inline": True},
                {"name": "Contracts", "value": str(size), "inline": True},
                {"name": "Cost / Risk", "value": f"${cost:.2f}", "inline": True},
                {"name": "Potential Profit", "value": f"${potential_profit:+.2f}", "inline": True},
                {"name": "Edge", "value": f"{edge:.2%}", "inline": True},
                {"name": "Vol (pred vs impl)", "value": vol_str or "N/A", "inline": True},
                {"name": "Balance After", "value": f"${balance:.2f}", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _post_embed(embed)
    except Exception as e:
        logger.warning(f"Discord trade alert failed: {e}")


def send_settlement_alert(
    *,
    ticker: str,
    side: str,
    result: str,
    pnl: float,
    price: int,
    size: int,
) -> None:
    """Send an embed when a trade settles (win or loss)."""
    if not _webhook_url:
        return
    try:
        won = side == result
        color = 0x2ECC71 if won else 0xE74C3C
        outcome = "WIN" if won else "LOSS"

        embed = {
            "title": f"Settlement — {outcome}",
            "color": color,
            "fields": [
                {"name": "Ticker", "value": ticker, "inline": True},
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Result", "value": result.upper(), "inline": True},
                {"name": "Price", "value": f"{price}c", "inline": True},
                {"name": "Contracts", "value": str(size), "inline": True},
                {"name": "P&L", "value": f"${pnl:+.2f}", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _post_embed(embed)
    except Exception as e:
        logger.warning(f"Discord settlement alert failed: {e}")


def send_eval_summary(
    *,
    btc_price: float,
    event_ticker: str,
    brackets_evaluated: int,
    best_edge: float,
    best_edge_bracket: str,
    trades_attempted: int,
    trades_filled: int,
) -> None:
    """Send a summary embed after each eval cycle (hourly heartbeat)."""
    if not _webhook_url:
        return
    try:
        if trades_filled > 0:
            color = 0x2ECC71  # green — trade filled
        elif trades_attempted > 0:
            color = 0xF1C40F  # yellow — edge found, no fill
        else:
            color = 0x3498DB  # blue — no action

        if trades_filled > 0:
            outcome = f"{trades_filled} filled"
        elif trades_attempted > 0:
            outcome = f"{trades_attempted} attempted, 0 filled"
        else:
            outcome = "No edge found"

        embed = {
            "title": "Eval Summary",
            "color": color,
            "fields": [
                {"name": "BTC Price", "value": f"${btc_price:,.0f}", "inline": True},
                {"name": "Event", "value": event_ticker or "N/A", "inline": True},
                {"name": "Brackets", "value": str(brackets_evaluated), "inline": True},
                {"name": "Best Edge", "value": f"{best_edge:.2%}" if best_edge else "N/A", "inline": True},
                {"name": "Best Bracket", "value": best_edge_bracket or "N/A", "inline": True},
                {"name": "Outcome", "value": outcome, "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _post_embed(embed)
    except Exception as e:
        logger.warning(f"Discord eval summary failed: {e}")


def send_startup_alert(
    *,
    mode: str,
    balance: float,
    event_ticker: str,
) -> None:
    """Send an embed when the bot starts up."""
    if not _webhook_url:
        return
    try:
        embed = {
            "title": "Bot Started",
            "color": 0x3498DB,  # blue
            "fields": [
                {"name": "Mode", "value": mode.upper(), "inline": True},
                {"name": "Balance", "value": f"${balance:.2f}", "inline": True},
                {"name": "Event", "value": event_ticker or "N/A", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _post_embed(embed)
    except Exception as e:
        logger.warning(f"Discord startup alert failed: {e}")
