# Changelog

## 2026-02-25 (evening)

### Fix NO-side pricing bug in `_get_realistic_price`
- **What:** NO-side edge check was using `(100 - fill_price)` instead of `fill_price` for the cost, making every NO price look like it had massive edge. Example: 87c NO showed 69% edge instead of -5%.
- **Why:** Bug caused the bot to accept any NO fill price on the book, even when paying more than the model's win probability justifies.
- **Fix:** Changed `effective_implied = (100 - fill_price) / 100.0` to `fill_price / 100.0` for NO side, mirroring the YES logic.
- **Rollback:** Change `effective_implied = fill_price / 100.0` back to `(100 - fill_price) / 100.0` in the NO branch of `_get_realistic_price` (but don't — it was wrong)

### Cap NO price at 50c
- **What:** NO orders above 50c are now rejected. Configurable via `risk.max_no_price_cents`.
- **Why:** Over 36 NO-edge evals, expensive NOs (75c+) lost money despite 75% win rate — the risk/reward is too lopsided (risk 85c to win 15c). Cheap NOs (<50c) were the only profitable bucket.
- **Data:** Expensive NO (75c+): 24 trades, 75% win, -$0.10/contract. Cheap NO (<50c): 2 trades, 50% win, +$0.14/contract.
- **Rollback:** Remove the `max_no_price_cents` check in `run.py` or set to 99 in config.yaml

## 2026-02-25

### Shrinkage 0.3 -> 0.4
- **What:** Vol model now trusts its own prediction 40% vs market's implied vol 60% (was 30/70)
- **Why:** Over 145 resolved evaluations, the raw model vol (avg 0.00238) almost perfectly matches realized vol (avg 0.00239), while implied vol (avg 0.02161) overestimates by ~9x. The model was being drowned out by the market.
- **Expected effect:** ~40% more trades, better win rate on YES side, same strong NO side performance
- **Simulated PnL (on 145 resolved evals):** 0.3 shrinkage = +$0.97 (27 trades, 33% win) vs 0.4 = +$4.29 (38 trades, 37% win)
- **Rollback:** Change default in `kalshi_trader/models/vol_model.py` `__init__` back to `shrinkage=0.3`

### OTM bracket probability fix
- **What:** `bracket_prob.py` now uses raw model vol (not blended) for distance-to-bracket calculation, and steeper trust decay (1.0x vs 0.5x)
- **Why:** High implied vol was bleeding into blended vol, making far-out brackets look artificially close, creating phantom edge
- **Rollback:** Revert `raw_model_vol` parameter in `estimate_bracket_prob_from_vol` and `trust_factor` decay back to `0.5 * (distance - 1)`

### Deribit options data logging (passive)
- **What:** New `DeribitClient` fetches DVOL index, BTC-PERPETUAL funding/OI, and ATM options IV every 60s. Logged to `deribit_snapshots` table. No model integration.
- **Why:** Collecting data to later evaluate whether options-derived vol improves the model
- **Rollback:** Remove deribit import/init/poll from `run.py`, safe to leave table in DB
