"""Backtest edge threshold using resolved evaluations with full algo simulation."""

import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import norm

# ========= CURRENT ALGO PARAMETERS =========
SHRINKAGE = 0.4
VOL_FLOOR_RATIO = 0.5
KELLY_FRACTION = 0.5
FULL_KELLY_EDGE = 0.10
MAX_POSITION = 50       # dollars per trade
MAX_EXPOSURE = 150      # dollars total
MAX_DAILY_LOSS = 50     # dollars
MAX_NO_PRICE_CENTS = 50 # NO price cap
BALANCE = 500           # starting balance
MAX_DAILY_TRADES_PER_BRACKET = 3
OTM_DECAY = 1.0         # trust factor decay rate


def blend_vol(raw, implied):
    blended = implied + SHRINKAGE * (raw - implied)
    floor = implied * VOL_FLOOR_RATIO
    return max(blended, floor, 0.0005)


def bracket_prob_calc(price, low, high, blended_vol, raw_vol, p_up=0.5):
    if blended_vol <= 0:
        blended_vol = 0.003
    drift = norm.ppf(max(0.01, min(0.99, p_up))) * blended_vol
    mu = price * (1 + drift)
    sigma = price * blended_vol
    if sigma <= 0:
        return 0.5
    prob = norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma)
    prob = float(max(0.0, min(1.0, prob)))

    # OTM trust factor using raw vol
    bracket_mid = (low + high) / 2
    distance = abs(bracket_mid - price) / price
    vol_for_dist = raw_vol if raw_vol > 0 else blended_vol
    dist_sigmas = distance / max(vol_for_dist, 0.0005)
    if dist_sigmas > 1.0:
        trust = 1.0 / (1.0 + OTM_DECAY * (dist_sigmas - 1.0))
        prob *= trust
    return prob


def kelly_ramp(edge, threshold):
    if edge <= 0:
        return 0.0
    if edge >= FULL_KELLY_EDGE:
        return 1.0
    if edge <= threshold:
        return 0.25
    t = (edge - threshold) / (FULL_KELLY_EDGE - threshold)
    return 0.25 + 0.75 * t


def kelly_f(model_p, market_p, ramp_mult):
    if market_p <= 0 or market_p >= 1 or model_p <= 0 or model_p >= 1:
        return 0.0
    b = (1.0 - market_p) / market_p
    p = model_p
    q = 1.0 - p
    f_star = (p * b - q) / b
    if f_star <= 0:
        return 0.0
    return min(f_star * KELLY_FRACTION * ramp_mult, 1.0)


def size_order(kf, balance, fill_price):
    if kf <= 0 or fill_price <= 0:
        return 0
    dollars = min(kf * balance, MAX_POSITION)
    return max(int(dollars / fill_price), 0)


def simulate(evals, edge_threshold):
    trades = []
    balance = BALANCE
    daily_loss = {}
    daily_bracket_count = {}
    exposure = 0

    for _, e in evals.sort_values("timestamp").iterrows():
        day = pd.to_datetime(e["timestamp"], unit="ms").date()
        ticker = e["ticker"]

        # Recompute vol blend with current settings
        raw = e["raw_model_vol"]
        impl = e["implied_vol"]
        blended = blend_vol(raw, impl)

        # Recompute bracket probability with current OTM logic
        bp = bracket_prob_calc(
            e["btc_price"], e["bracket_low"], e["bracket_high"], blended, raw
        )
        mkt = e["market_prob"]
        edge = bp - mkt

        # Determine side and check threshold
        if edge > 0:
            side = "yes"
            abs_edge = edge
            our_p = bp
            market_p = mkt
        elif edge < -edge_threshold:
            side = "no"
            abs_edge = abs(edge)
            our_p = 1 - bp
            market_p = 1 - mkt
        else:
            continue

        if abs_edge < edge_threshold:
            continue

        # NO price cap
        if side == "no":
            no_price_cents = int((1 - mkt) * 100)
            if no_price_cents > MAX_NO_PRICE_CENTS:
                continue

        # Daily loss check
        day_loss = daily_loss.get(day, 0)
        if day_loss <= -MAX_DAILY_LOSS:
            continue

        # Per-bracket daily limit
        day_bracket_key = (day, ticker)
        if daily_bracket_count.get(day_bracket_key, 0) >= MAX_DAILY_TRADES_PER_BRACKET:
            continue

        # Kelly sizing
        ramp = kelly_ramp(abs_edge, edge_threshold)
        kf = kelly_f(our_p, market_p, ramp)
        if kf <= 0:
            continue

        # Fill price = market mid (no orderbook data available)
        if side == "yes":
            fill_price = mkt
        else:
            fill_price = 1 - mkt

        # NO pricing edge check (fixed version)
        if side == "no":
            remaining_edge = (1 - bp) - fill_price
            if remaining_edge < edge_threshold:
                continue

        contracts = size_order(kf, balance, fill_price)
        if contracts <= 0:
            continue

        # Exposure check
        trade_cost = contracts * fill_price
        if exposure + trade_cost > MAX_EXPOSURE:
            contracts = max(1, int((MAX_EXPOSURE - exposure) / fill_price))
            trade_cost = contracts * fill_price
        if contracts <= 0:
            continue

        # Compute PnL
        if side == "yes":
            if e["in_bracket"]:
                pnl = contracts * (1 - fill_price)
            else:
                pnl = contracts * (-fill_price)
        else:
            if not e["in_bracket"]:
                pnl = contracts * (1 - fill_price)
            else:
                pnl = contracts * (-fill_price)

        # Update state
        balance += pnl
        daily_loss[day] = daily_loss.get(day, 0) + min(pnl, 0)
        daily_bracket_count[day_bracket_key] = (
            daily_bracket_count.get(day_bracket_key, 0) + 1
        )

        trades.append(
            {
                "day": day,
                "ticker": ticker,
                "side": side,
                "contracts": contracts,
                "fill_price": fill_price,
                "edge": abs_edge,
                "pnl": pnl,
                "balance": balance,
                "bracket_prob": bp,
                "market_prob": mkt,
                "blended": blended,
                "raw": raw,
                "implied": impl,
                "in_bracket": e["in_bracket"],
            }
        )

    return pd.DataFrame(trades)


def main():
    conn = sqlite3.connect("data/trader.db")
    evals = pd.read_sql_query("SELECT * FROM evaluations WHERE resolved=1", conn)
    conn.close()

    print(f"Resolved evals: {len(evals)}")
    date_min = pd.to_datetime(evals["timestamp"].min(), unit="ms").date()
    date_max = pd.to_datetime(evals["timestamp"].max(), unit="ms").date()
    print(f"Date range: {date_min} to {date_max}")
    print(f"Brackets hit: {evals['in_bracket'].sum()}/{len(evals)} ({evals['in_bracket'].mean():.1%})")
    print()

    # ========= SUMMARY TABLE =========
    thresholds = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

    header = (
        f"{'Thresh':>7s} {'Trades':>7s} {'YES':>5s} {'NO':>5s} "
        f"{'Win%':>6s} {'Total PnL':>10s} {'Final Bal':>10s} "
        f"{'Max DD':>8s} {'Avg Size':>9s} {'PnL/Trade':>10s}"
    )
    print(header)
    print("-" * len(header))

    for t in thresholds:
        df = simulate(evals, t)
        if len(df) == 0:
            print(f"{t:>6.1%} {0:>7}       (no trades)")
            continue

        yes_c = (df["side"] == "yes").sum()
        no_c = (df["side"] == "no").sum()
        wins = (df["pnl"] > 0).mean()

        cumulative = df["pnl"].cumsum()
        running_max = cumulative.cummax()
        max_dd = (cumulative - running_max).min()

        marker = " <-- current" if t == 0.04 else ""
        print(
            f"{t:>6.1%} {len(df):>7} {yes_c:>5} {no_c:>5} {wins:>5.0%} "
            f"${df['pnl'].sum():>+9.2f} ${df['balance'].iloc[-1]:>9.2f} "
            f"${max_dd:>+7.2f} {df['contracts'].mean():>8.0f}x "
            f"${df['pnl'].mean():>+9.3f}{marker}"
        )

    # ========= DETAILED TRADE LOG =========
    for t in [0.03, 0.04]:
        df = simulate(evals, t)
        if len(df) == 0:
            continue
        label = "(current)" if t == 0.04 else ""
        print(f"\n=== {t:.1%} threshold {label} -- all {len(df)} trades ===")
        for _, r in df.iterrows():
            hit = "WIN " if r["pnl"] > 0 else "LOSS"
            bracket = r["ticker"].split("-")[-1] if "-" in r["ticker"] else r["ticker"]
            print(
                f"  {r['day']} {r['side']:>3s} {r['contracts']:>4.0f}x "
                f"@{r['fill_price']*100:>5.1f}c  edge={r['edge']:.3f}  "
                f"raw={r['raw']:.5f} blend={r['blended']:.5f} impl={r['implied']:.5f}  "
                f"pnl=${r['pnl']:>+8.2f}  bal=${r['balance']:>8.2f}  {hit}"
            )

        yes_df = df[df["side"] == "yes"]
        no_df = df[df["side"] == "no"]
        print(
            f"\n  YES: {len(yes_df)} trades, "
            f"{(yes_df['pnl']>0).mean():.0%} win, "
            f"PnL=${yes_df['pnl'].sum():+.2f}, "
            f"avg cost={yes_df['fill_price'].mean()*100:.0f}c, "
            f"avg size={yes_df['contracts'].mean():.0f}x"
            if len(yes_df) > 0
            else "\n  YES: 0 trades"
        )
        print(
            f"  NO:  {len(no_df)} trades, "
            f"{(no_df['pnl']>0).mean():.0%} win, "
            f"PnL=${no_df['pnl'].sum():+.2f}, "
            f"avg cost={no_df['fill_price'].mean()*100:.0f}c, "
            f"avg size={no_df['contracts'].mean():.0f}x"
            if len(no_df) > 0
            else "  NO:  0 trades"
        )

        # Daily breakdown
        by_day = df.groupby("day").agg(
            trades=("pnl", "count"), pnl=("pnl", "sum")
        )
        print(f"\n  Daily:")
        for day, row in by_day.iterrows():
            print(f"    {day}: {int(row['trades'])} trades, PnL=${row['pnl']:+.2f}")


if __name__ == "__main__":
    main()
