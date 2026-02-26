"""Gradient boosting model for predicting next-candle realized volatility.

Volatility is more predictable than direction due to strong patterns:
clustering (high vol begets high vol), intraday seasonality (overnight < US open),
and mean reversion (extreme vol reverts to average).

Edge in bracket markets: if predicted_vol < implied_vol, the bracket is underpriced.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class VolModel:
    """Predicts next-candle realized volatility using gradient boosting.

    Target: abs(next_close_return) — the magnitude of the next 15-min move.
    This directly maps to whether price stays in a bracket or not.
    """

    def __init__(self, lookback: int = 20, shrinkage: float = 0.4, vol_floor_ratio: float = 0.5):
        self.lookback = lookback
        self.shrinkage = shrinkage  # how much to trust model vs market
        self.vol_floor_ratio = vol_floor_ratio  # blended vol can't be less than this * implied
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
        )
        self.scaler = StandardScaler()
        self._fitted = False
        self._base_vol = 0.003  # fallback ~0.3%

    def _compute_features(self, candles: pd.DataFrame, funding_rate: float = 0.0) -> np.ndarray:
        """Extract volatility-predictive features from candle data."""
        df = candles.copy()
        df["return"] = df["close"].pct_change()
        df["abs_return"] = df["return"].abs()
        df["range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

        features = []

        # Time-of-day (cyclical encoding for intraday vol seasonality)
        if "open_time_dt" in df.columns:
            hours = df["open_time_dt"].dt.hour + df["open_time_dt"].dt.minute / 60.0
        elif "open_time" in df.columns:
            # open_time is Unix ms
            hours = (df["open_time"] / 1000 % 86400) / 3600.0
        else:
            hours = pd.Series(np.zeros(len(df)), index=df.index)

        features.append(np.sin(2 * np.pi * hours / 24).values)  # hour_sin
        features.append(np.cos(2 * np.pi * hours / 24).values)  # hour_cos

        # Day of week
        if "open_time_dt" in df.columns:
            dow = df["open_time_dt"].dt.dayofweek.values.astype(float)
        elif "open_time" in df.columns:
            dow = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.dayofweek.values.astype(float)
        else:
            dow = np.zeros(len(df))
        features.append(dow / 6.0)  # normalize to [0, 1]

        # Recent realized vol (short window — vol clustering)
        features.append(df["return"].rolling(4).std().values)

        # Recent realized vol (medium window)
        features.append(df["return"].rolling(self.lookback).std().values)

        # Vol ratio: short / long (mean reversion signal)
        vol_short = df["return"].rolling(4).std()
        vol_long = df["return"].rolling(self.lookback).std()
        features.append((vol_short / vol_long.replace(0, np.nan)).fillna(1.0).values)

        # Absolute return of last candle (immediate vol proxy)
        features.append(df["abs_return"].values)

        # Mean absolute return over last 4 candles
        features.append(df["abs_return"].rolling(4).mean().values)

        # Range of last candle (Parkinson-style vol)
        features.append(df["range"].fillna(0).values)

        # Mean range over last 4 candles
        features.append(df["range"].rolling(4).mean().fillna(0).values)

        # Volume ratio (volume spike signal)
        vol_mean = df["volume"].rolling(self.lookback).mean()
        features.append((df["volume"] / vol_mean.replace(0, np.nan)).fillna(1.0).values)

        # Funding rate
        features.append(np.full(len(df), funding_rate))

        X = np.column_stack(features)
        return X

    def train(self, candles: pd.DataFrame, funding_rate: float = 0.0):
        """Train model to predict next-candle absolute return."""
        if len(candles) < self.lookback + 50:
            logger.warning("Not enough data to train vol model")
            return

        X = self._compute_features(candles, funding_rate)
        # Target: absolute return of the NEXT candle
        y = candles["close"].pct_change().shift(-1).abs().values

        # Filter valid rows
        valid = ~(np.isnan(X).any(axis=1)) & ~np.isnan(y)
        valid[-1] = False  # last row has no target
        X = X[valid]
        y = y[valid]

        if len(X) < 100:
            logger.warning("Not enough valid rows to train vol model")
            return

        self._base_vol = float(np.median(y))

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True

        # Evaluate on training data
        y_pred = self.model.predict(X_scaled)
        correlation = float(np.corrcoef(y, y_pred)[0, 1])
        mae = float(np.mean(np.abs(y - y_pred)))
        median_actual = float(np.median(y))
        median_pred = float(np.median(y_pred))

        logger.info(
            f"Vol model trained: {len(X)} samples, "
            f"correlation={correlation:.3f}, MAE={mae:.5f}, "
            f"median_actual={median_actual:.5f}, median_pred={median_pred:.5f}"
        )

        # Optimize shrinkage on this data
        self.optimize_shrinkage(candles, funding_rate)

    def optimize_shrinkage(self, candles: pd.DataFrame, funding_rate: float = 0.0):
        """Disabled: previous implementation used median(y_test) as implied vol proxy,
        which is wrong units and actively harmful. Will be re-enabled in Phase 3
        once we have real (predicted, implied, realized) triples from the evaluations table.
        """
        logger.info(f"Shrinkage optimization disabled — using fixed shrinkage={self.shrinkage}")

    def predict(self, candles: pd.DataFrame, funding_rate: float = 0.0) -> float:
        """Predict next-candle realized vol (absolute return).

        Returns:
            Predicted absolute return (e.g. 0.002 = 0.2% expected move).
        """
        if not self._fitted:
            return self._base_vol

        X = self._compute_features(candles, funding_rate)
        last_row = X[-1:]
        if np.isnan(last_row).any():
            return self._base_vol

        X_scaled = self.scaler.transform(last_row)
        pred = self.model.predict(X_scaled)[0]

        # Floor at a small positive value
        return float(max(pred, 0.0005))

    def blend_with_implied(self, predicted_vol: float, implied_vol: float) -> float:
        """Blend predicted vol toward implied vol (shrinkage).

        The market's implied vol accounts for tail risk, model uncertainty,
        etc. We only trust our deviation from the market by `shrinkage` amount.

        shrinkage=0.3 → blended = implied + 0.3 * (predicted - implied)

        A floor prevents blended vol from being less than vol_floor_ratio * implied,
        capping the maximum phantom edge from model underestimation.
        """
        blended = implied_vol + self.shrinkage * (predicted_vol - implied_vol)
        floor = implied_vol * self.vol_floor_ratio
        return float(max(blended, floor, 0.0005))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model, "scaler": self.scaler,
                "lookback": self.lookback, "base_vol": self._base_vol,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.lookback = data["lookback"]
        self._base_vol = data.get("base_vol", 0.003)
        self._fitted = True
