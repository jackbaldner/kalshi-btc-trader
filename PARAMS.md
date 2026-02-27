# Parameter Change Log

Quick reference for every parameter change, when it happened, and why.

| Date | Parameter | From | To | Why | Result |
|------|-----------|------|----|-----|--------|
| 2026-02-27 | `ensemble.py` bracket edge check | `> 0` (yes), `< -threshold` (no) | `>= threshold` (yes), `<= -threshold` (no) | Asymmetric check let tiny positive edges through (caught by safety net but wasted compute) | Cleaner, symmetric threshold check |
| 2026-02-26 | NO-side `_get_realistic_price` edge calc | `(1 - model_prob) - effective_implied` | `model_prob - effective_implied` | Double-flipped NO probability — blocked ALL NO fills despite real edge | NO trades now execute correctly |
| 2026-02-25 | `risk.max_no_price_cents` | _(none)_ | 50 | Expensive NOs (75c+) lost money despite 75% win rate — lopsided risk/reward | Filters out bad risk/reward NO trades |
| 2026-02-25 | NO-side `effective_implied` | `(100 - fill_price) / 100` | `fill_price / 100` | Made every NO price look like massive edge | Correct NO pricing |
| 2026-02-25 | `vol_model.shrinkage` | 0.3 | 0.4 | 145 evals: 0.4 had best win rate (37%) and PnL (+$4.29 vs +$0.97). Raw model vol matches realized; implied overestimates 9x | ~40% more trades, better quality |
| 2026-02-25 | OTM bracket trust decay | `0.5 * (distance - 1)` | `1.0 * (distance - 1)` | High implied vol bled into blended vol, making far brackets look close | Steeper decay kills phantom OTM edge |
| 2026-02-25 | OTM distance calc input | blended vol | raw model vol | Blended vol includes implied, which inflated distance denominator | Cleaner separation of model vs market |
| _initial_ | `vol_model.vol_floor_ratio` | — | 0.5 | Prevent phantom edge from systematic vol underestimation | Blended vol >= 50% of implied |
| _initial_ | `strategy.edge_threshold` | — | 0.04 | Minimum edge to trade | 4% threshold |
| _initial_ | `risk.max_position_size` | — | $50 | Per-trade risk cap | — |
| _initial_ | `risk.max_total_exposure` | — | $150 | Total open position cap | — |
| _initial_ | `risk.max_daily_loss` | — | $50 | Daily loss circuit breaker | — |
| _initial_ | `kelly.fraction` | — | 0.5 | Half-Kelly for safety | — |
| _initial_ | `kelly.full_kelly_edge` | — | 0.10 | Edge at which Kelly ramp reaches 1.0x | — |
| _initial_ | `fair_value.shrinkage` | — | 0.3 | Heavy shrinkage toward 50% base rate | — |
| _initial_ | `fair_value.clip` | — | [0.35, 0.65] | Prevent overconfident directional bets | — |
| _initial_ | `risk.max_daily_trades_per_bracket` | — | 4 | Avoid overconcentration on one ticker | — |
| _initial_ | `risk.max_book_take_pct` | — | 0.5 | Liquidity cap: max 50% of book depth | — |
| _initial_ | `strategy.price_aggression_cents` | — | 1 | Price 1c above best ask to cross spread | — |
| _initial_ | `risk.cancel_before_expiry_sec` | — | 15 | Auto-cancel resting orders before market close | — |
