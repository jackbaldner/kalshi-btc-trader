"""Trade logging to SQLite and CSV."""

import csv
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeLogger:
    """Logs trades to SQLite database and CSV file."""

    def __init__(self, database, csv_path: str):
        self.db = database
        self.csv_path = csv_path
        self._init_csv()

    def _init_csv(self):
        path = Path(self.csv_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "ticker", "side", "price", "size",
                    "strategy", "edge", "model_prob", "implied_prob", "pnl",
                ])

    def log_trade(self, ticker: str, side: str, price: float, size: int,
                  strategy: str, edge: float, model_prob: float, implied_prob: float):
        """Log a trade to both SQLite and CSV."""
        ts = int(time.time() * 1000)

        # SQLite
        self.db.insert_trade(ts, ticker, side, price, size, strategy, edge, model_prob, implied_prob)

        # CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, ticker, side, price, size, strategy, edge, model_prob, implied_prob, ""])

        logger.info(
            f"Trade logged: {side} {size}x {ticker} @ {price}c | "
            f"strategy={strategy} edge={edge:.3f}"
        )

    def update_pnl(self, trade_id: int, pnl: float):
        """Update PnL for a resolved trade."""
        self.db.update_trade_pnl(trade_id, pnl)
        logger.info(f"Trade {trade_id} resolved: PnL=${pnl:.2f}")
