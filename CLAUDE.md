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
- `binance.py` — BinanceClient: spot price, 15m OHLCV candles, funding rate
- `kalshi_client.py` — KalshiClient: REST API with RSA-PSS auth, markets/orders/positions
- `kalshi_ws.py` — KalshiWebSocket: real-time orderbook deltas
- `kalshi_mock.py` — KalshiMock: paper trading with realistic orderbook-based fills
- `database.py` — SQLite: candles, snapshots, trades, strategy signals

### Models (`kalshi_trader/models/`)
- `fair_value.py` — FairValueModel: logistic regression, P(BTC up next 15m), heavy shrinkage (0.3x toward 50%)
- `vol_model.py` — VolModel: gradient boosting, predicts next-candle absolute return magnitude, optimized shrinkage
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

1. **Binance poll** (10s): price, candles, funding → DB
2. **Kalshi poll** (15s): find nearest event, parse brackets, subscribe orderbook
3. **Evaluate** (870-930s before close): run ensemble on center ± 2 brackets, place orders
4. **Settle** (60s): paper=compare price to bracket, live=query API result

## Key Design Decisions

- **Edge threshold**: 4% minimum to trade
- **Kelly fraction**: 0.5 (half-Kelly for safety), ramps 0.25x→1.0x based on edge size
- **Model shrinkage**: Fair value 0.3x, vol model optimized per training — prevents overconfident outputs
- **Output clipping**: Model probs clipped to [0.35, 0.65]
- **Dedup**: max 1 eval per event close, 1 trade per ticker per hour, 3 per ticker per day
- **Fill pricing**: walks opposite-side orderbook, prices 1c above best ask, verifies edge remains
- **Liquidity cap**: order capped at 50% of available book depth

## Conventions

- Kalshi prices in **cents** (1-99), model probs in **decimals** [0,1]
- Timestamps in **ms since epoch** (UTC)
- Paper mode uses real API reads + mock writes (same KalshiClient interface)
- All strategies inherit from `Strategy` ABC, return `Signal(direction, confidence, strategy, features)`
- Python 3.11+, async/await throughout, dependencies in pyproject.toml + requirements.txt
