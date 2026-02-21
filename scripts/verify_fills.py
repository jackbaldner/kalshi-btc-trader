"""Verify orderbook fill logic against real snapshots."""
import sqlite3
import json

conn = sqlite3.connect("data/trader.db")
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT yes_bid, yes_ask, no_bid, no_ask, orderbook_json, ticker "
    "FROM kalshi_snapshots WHERE orderbook_json != '{}' ORDER BY timestamp DESC LIMIT 20"
).fetchall()

checked = 0
for r in rows:
    ob = json.loads(r["orderbook_json"])
    yes_bids = ob.get("yes", [])
    no_bids = ob.get("no", [])
    if not yes_bids or not no_bids:
        continue

    # To buy YES: cross NO bids. NO bid at X = YES ask at (100-X)
    yes_asks = sorted([(100 - p, q) for p, q in no_bids], key=lambda x: x[0])
    # To buy NO: cross YES bids. YES bid at X = NO ask at (100-X)
    no_asks = sorted([(100 - p, q) for p, q in yes_bids], key=lambda x: x[0])

    best_yes_ask = yes_asks[0][0] if yes_asks else None
    best_no_ask = no_asks[0][0] if no_asks else None

    yes_ok = best_yes_ask == r["yes_ask"]
    no_ok = best_no_ask == r["no_ask"]

    total_yes_depth = sum(q for _, q in yes_asks)

    status = "OK" if yes_ok and no_ok else "MISMATCH"
    print(
        f"[{status}] {r['ticker']}: "
        f"yes_ask market={r['yes_ask']} book={best_yes_ask} | "
        f"no_ask market={r['no_ask']} book={best_no_ask} | "
        f"YES depth={total_yes_depth}"
    )
    checked += 1

    # Simulate: if we tried to buy 300 YES contracts at the ask price
    if yes_asks:
        limit = int(r["yes_ask"])
        filled = 0
        cost = 0.0
        for price, qty in yes_asks:
            if price > limit:
                break
            take = min(qty, 300 - filled)
            filled += take
            cost += price * take / 100
            if filled >= 300:
                break
        avg = (cost / filled * 100) if filled > 0 else 0
        print(f"  -> Buy 300 YES @ {limit}c limit: filled {filled}/300, avg {avg:.1f}c, cost ${cost:.2f}")

print(f"\nChecked {checked} snapshots")
conn.close()
