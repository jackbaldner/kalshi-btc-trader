# Kalshi BTC Trader

Algorithmic trading system for Kalshi KXBTC15M binary prediction markets (BTC price-range brackets, 15-minute windows).

## Quick Reference

- **Entry point**: `run.py` — `python run.py --mode live --no-dashboard`
- **Launcher**: `start_bot.bat` or `bot_launcher.py` (detached Windows process)
- **Dashboard**: `streamlit run dashboard.py` (web UI) or remove `--no-dashboard` for Rich terminal UI
- **Config**: `config.yaml` + `.env` (API keys)
- **Database**: `data/trader.db` (SQLite)
- **Logs**: `logs/live.log`, `data/trades.csv`

## Architecture

```
Binance (price, candles, funding) ──┐
                                    ├──► EnsembleStrategy ──► OrderManager ──► Kalshi API
Kalshi (brackets, orderbook, WS) ──┘         │                    │
                                         VolModel            RiskManager
                                       FairValueModel        TradeLogger
                                        KellySizing           Database
```

### Data Layer (`kalshi_trader/data/`)
- `binance.py` — BinanceClient: spot price, 15m OHLCV candles, funding rate, historical backfill (paginated)
- `kalshi_client.py` — KalshiClient: REST API with RSA-PSS auth, markets/orders/positions
- `kalshi_ws.py` — KalshiWebSocket: real-time orderbook deltas
- `kalshi_mock.py` — KalshiMock: paper trading with realistic orderbook-based fills
- `database.py` — SQLite: candles, snapshots, trades, strategy signals, **evaluations**

### Models (`kalshi_trader/models/`)
- `fair_value.py` — FairValueModel: logistic regression, P(BTC up next 15m), heavy shrinkage (0.3x toward 50%)
- `vol_model.py` — VolModel: gradient boosting, predicts next-candle absolute return magnitude
  - **Vol floor**: blended vol ≥ `vol_floor_ratio` × implied vol (default 0.5) — prevents phantom edge from systematic underestimation
  - **Shrinkage**: fixed at 0.4 (bumped from 0.3 based on 145-eval backtest — better win rate and PnL). Will re-enable optimization with more data
  - **Retraining**: every 6 hours on all available candles, with 3x shift safety check
- `bracket_prob.py` — Converts directional prob + vol into bracket probability using normal CDF, with tail-risk adjustment
- `kelly.py` — Kelly criterion sizing with ramp (0.25x at threshold → 1.0x at full_kelly_edge)

### Strategies (`kalshi_trader/strategies/`)
- `ensemble.py` — EnsembleStrategy: orchestrates all models and strategies
  - **Bracket mode** (primary): trades vol edge (predicted_vol vs implied_vol), no confirmation needed
  - **Directional mode** (fallback): model + strategy confirmation required
- `momentum.py` — Continuation when N recent candles agree directionally
- `mean_reversion.py` — Fades extreme moves (z-score > 2.0)
- `orderbook_imbalance.py` — Microstructure: yes/no volume imbalance > 15%
- `funding_rate.py` — Contrarian signal from Binance perp funding
- `volatility_regime.py` — Classifies LOW/MEDIUM/HIGH vol, weights other strategies

### Execution (`kalshi_trader/execution/`)
- `order_manager.py` — Order placement with retries, auto-cancel before expiry
- `risk.py` — RiskManager: position ($50), exposure ($150), daily loss ($50), balance checks
- `trade_logger.py` — Logs to SQLite + CSV

### Other
- `kalshi_trader/notifications/discord.py` — Discord webhook alerts on fills
- `kalshi_trader/dashboard/cli.py` — Rich terminal dashboard
- `dashboard.py` — Streamlit web dashboard with equity curves, trade analysis
- `kalshi_trader/backtest/` — Backtesting engine, market simulator, metrics

## Trading Loop (run.py TradingSystem)

1. **Startup**: load candles from DB, backfill 180 days from Binance if <17k candles, train models
2. **Binance poll** (10s): price, candles (last 500), funding → DB
3. **Kalshi poll** (15s): find nearest event, parse brackets, subscribe orderbook
4. **Evaluate** (870-930s before close): run ensemble on center ± 2 brackets, log evaluation to DB, place orders
5. **Settle + Resolve** (60s): settle trades (paper/live), resolve evaluations with realized vol/return
6. **Retrain** (6h): retrain vol + fair value models on all candles

## Key Design Decisions

- **Edge threshold**: 4% minimum to trade
- **Kelly fraction**: 0.5 (half-Kelly for safety), ramps 0.25x→1.0x based on edge size
- **Vol floor**: blended vol ≥ 50% of implied vol (`vol_floor_ratio` in config) — prevents phantom edge
- **Model shrinkage**: Fair value 0.3x, vol model fixed 0.4 (bumped from 0.3 based on eval data)
- **Output clipping**: Model probs clipped to [0.35, 0.65]
- **Dedup**: max 1 eval per event close, 1 trade per ticker per hour, 3 per ticker per day
- **Fill pricing**: walks opposite-side orderbook, prices 1c above best ask, verifies edge remains
- **Liquidity cap**: order capped at 50% of available book depth

## Evaluations Pipeline

Every bracket evaluation is logged to the `evaluations` table (~480/day across all brackets). After the 15-min window closes, `_resolve_evaluations()` fills in realized data. This creates triples of:

- `raw_model_vol` / `predicted_vol` (blended) / `implied_vol` → what we predicted
- `realized_vol` / `realized_return` / `price_at_close` / `in_bracket` → what actually happened

**Useful queries:**
- Total evaluations: `SELECT COUNT(*) FROM evaluations`
- Resolved: `SELECT COUNT(*) FROM evaluations WHERE resolved=1`
- Model accuracy: `SELECT AVG(ABS(predicted_vol - realized_vol)) FROM evaluations WHERE resolved=1`
- Predicted/implied ratio: `SELECT AVG(predicted_vol/implied_vol) FROM evaluations WHERE implied_vol > 0`

**Planned improvements (require collected data):**
- Re-enable shrinkage optimization against real (predicted, implied, realized) triples (needs 200+ resolved evals)
- Calibration mapping: bin model predictions into deciles, correct with mean realized vol per bin (needs 500+)
- Residual model: predict (realized - implied) directly (needs 2+ weeks)

## Conventions

- Kalshi prices in **cents** (1-99), model probs in **decimals** [0,1]
- Timestamps in **ms since epoch** (UTC)
- Paper mode uses real API reads + mock writes (same KalshiClient interface)
- All strategies inherit from `Strategy` ABC, return `Signal(direction, confidence, strategy, features)`
- Python 3.11+, async/await throughout, dependencies in pyproject.toml + requirements.txt
