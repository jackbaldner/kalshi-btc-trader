"""SQLite storage for candles, trades, market snapshots, and strategy signals."""

import sqlite3
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    open_time INTEGER PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    close_time INTEGER
);

CREATE TABLE IF NOT EXISTS kalshi_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    ticker TEXT,
    yes_bid REAL,
    yes_ask REAL,
    no_bid REAL,
    no_ask REAL,
    orderbook_json TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    ticker TEXT,
    side TEXT,
    price REAL,
    size REAL,
    strategy TEXT,
    edge REAL,
    model_prob REAL,
    implied_prob REAL,
    predicted_vol REAL,
    implied_vol REAL,
    fill_confidence REAL,
    bracket_low REAL,
    bracket_high REAL,
    pnl REAL,
    resolved INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS strategy_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    strategy TEXT,
    direction TEXT,
    confidence REAL,
    features_json TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    event_close_time TEXT,
    ticker TEXT,
    bracket_low REAL,
    bracket_high REAL,
    btc_price REAL,
    raw_model_vol REAL,
    predicted_vol REAL,
    implied_vol REAL,
    blended_vol REAL,
    shrinkage REAL,
    bracket_prob REAL,
    market_prob REAL,
    edge REAL,
    should_trade INTEGER,
    realized_vol REAL,
    realized_return REAL,
    price_at_close REAL,
    in_bracket INTEGER,
    resolved INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(open_time);
CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON kalshi_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_evaluations_time ON evaluations(timestamp);
CREATE INDEX IF NOT EXISTS idx_evaluations_resolved ON evaluations(resolved);
"""


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add columns that may be missing from older databases."""
        migrations = [
            ("trades", "predicted_vol", "REAL"),
            ("trades", "implied_vol", "REAL"),
            ("trades", "fill_confidence", "REAL"),
            ("trades", "bracket_low", "REAL"),
            ("trades", "bracket_high", "REAL"),
        ]
        for table, column, col_type in migrations:
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                self.conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    def close(self):
        self.conn.close()

    # -- Candles --
    def upsert_candles(self, df: pd.DataFrame):
        """Insert or replace candle rows from a DataFrame."""
        if df.empty:
            return
        rows = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].values.tolist()
        self.conn.executemany(
            "INSERT OR REPLACE INTO candles (open_time, open, high, low, close, volume, close_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def get_candles(self, limit: int = 100) -> pd.DataFrame:
        """Get most recent candles."""
        df = pd.read_sql_query(
            f"SELECT * FROM candles ORDER BY open_time DESC LIMIT {limit}", self.conn
        )
        if not df.empty:
            df = df.sort_values("open_time").reset_index(drop=True)
            df["open_time_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df

    # -- Kalshi Snapshots --
    def insert_snapshot(self, timestamp: int, ticker: str, yes_bid: float, yes_ask: float,
                        no_bid: float, no_ask: float, orderbook: dict):
        self.conn.execute(
            "INSERT INTO kalshi_snapshots (timestamp, ticker, yes_bid, yes_ask, no_bid, no_ask, orderbook_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, ticker, yes_bid, yes_ask, no_bid, no_ask, json.dumps(orderbook)),
        )
        self.conn.commit()

    # -- Trades --
    def insert_trade(self, timestamp: int, ticker: str, side: str, price: float, size: float,
                     strategy: str, edge: float, model_prob: float, implied_prob: float,
                     predicted_vol: float = None, implied_vol: float = None,
                     fill_confidence: float = None, bracket_low: float = None,
                     bracket_high: float = None):
        self.conn.execute(
            "INSERT INTO trades (timestamp, ticker, side, price, size, strategy, edge, model_prob, implied_prob, "
            "predicted_vol, implied_vol, fill_confidence, bracket_low, bracket_high) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, ticker, side, price, size, strategy, edge, model_prob, implied_prob,
             predicted_vol, implied_vol, fill_confidence, bracket_low, bracket_high),
        )
        self.conn.commit()

    def update_trade_pnl(self, trade_id: int, pnl: float):
        self.conn.execute(
            "UPDATE trades SET pnl = ?, resolved = 1 WHERE id = ?", (pnl, trade_id)
        )
        self.conn.commit()

    def get_recent_trades(self, limit: int = 100) -> pd.DataFrame:
        return pd.read_sql_query(
            f"SELECT * FROM trades ORDER BY timestamp DESC LIMIT {limit}", self.conn
        )

    def get_unresolved_trades(self) -> list[dict]:
        cursor = self.conn.execute("SELECT * FROM trades WHERE resolved = 0")
        return [dict(row) for row in cursor.fetchall()]

    def get_daily_pnl(self, day_start_ts: int) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE timestamp >= ? AND resolved = 1",
            (day_start_ts,),
        ).fetchone()
        return float(row["total"])

    def get_all_trades(self) -> pd.DataFrame:
        """Get all trades as a DataFrame for dashboard use."""
        return pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", self.conn)

    # -- Evaluations --
    def insert_evaluation(self, timestamp: int, event_close_time: str, ticker: str,
                          bracket_low: float, bracket_high: float, btc_price: float,
                          raw_model_vol: float, predicted_vol: float, implied_vol: float,
                          blended_vol: float, shrinkage: float, bracket_prob: float,
                          market_prob: float, edge: float, should_trade: bool):
        self.conn.execute(
            "INSERT INTO evaluations (timestamp, event_close_time, ticker, "
            "bracket_low, bracket_high, btc_price, raw_model_vol, predicted_vol, "
            "implied_vol, blended_vol, shrinkage, bracket_prob, market_prob, edge, should_trade) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, event_close_time, ticker, bracket_low, bracket_high, btc_price,
             raw_model_vol, predicted_vol, implied_vol, blended_vol, shrinkage,
             bracket_prob, market_prob, edge, int(should_trade)),
        )
        self.conn.commit()

    def get_unresolved_evaluations(self) -> list[dict]:
        cursor = self.conn.execute("SELECT * FROM evaluations WHERE resolved = 0")
        return [dict(row) for row in cursor.fetchall()]

    def resolve_evaluation(self, eval_id: int, realized_vol: float, realized_return: float,
                           price_at_close: float, in_bracket: bool):
        self.conn.execute(
            "UPDATE evaluations SET realized_vol = ?, realized_return = ?, "
            "price_at_close = ?, in_bracket = ?, resolved = 1 WHERE id = ?",
            (realized_vol, realized_return, price_at_close, int(in_bracket), eval_id),
        )
        self.conn.commit()

    # -- Strategy Signals --
    def insert_signal(self, timestamp: int, strategy: str, direction: str,
                      confidence: float, features: dict):
        self.conn.execute(
            "INSERT INTO strategy_signals (timestamp, strategy, direction, confidence, features_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, strategy, direction, confidence, json.dumps(features)),
        )
        self.conn.commit()
