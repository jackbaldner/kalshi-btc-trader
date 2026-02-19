"""Logistic regression model for P(BTC up) estimation in next 15-minute window."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FairValueModel:
    """Estimates P(BTC price up in next 15 minutes) using logistic regression.

    Uses very strong regularization and output shrinkage to produce well-calibrated
    probabilities. A model with ~55% accuracy should output probabilities in the
    0.47-0.55 range, not 0.3-0.7.
    """

    def __init__(self, lookback: int = 4, shrinkage: float = 0.3):
        self.lookback = lookback
        self.shrinkage = shrinkage  # how much to compress toward base rate
        # Very strong regularization — prevents overconfident coefficients
        self.model = LogisticRegression(max_iter=1000, C=0.001)
        self.scaler = StandardScaler()
        self._fitted = False
        self._base_rate = 0.5

    def _compute_features(self, candles: pd.DataFrame, funding_rate: float = 0.0) -> np.ndarray:
        """Extract features from candle data."""
        df = candles.copy()
        df["return"] = df["close"].pct_change()

        features = []
        for i in range(self.lookback):
            features.append(df["return"].shift(i + 1).values)

        # Rolling volatility
        features.append(df["return"].rolling(self.lookback).std().values)

        # Volume change ratio
        vol_mean = df["volume"].rolling(self.lookback).mean()
        features.append((df["volume"] / vol_mean.replace(0, np.nan)).fillna(1.0).values)

        # Cumulative return over lookback
        features.append(df["close"].pct_change(self.lookback).values)

        # High-low range (intrabar volatility)
        features.append(((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0).values)

        # Funding rate
        features.append(np.full(len(df), funding_rate))

        X = np.column_stack(features)
        return X

    def train(self, candles: pd.DataFrame, funding_rate: float = 0.0):
        """Train model on historical candle data."""
        if len(candles) < self.lookback + 50:
            logger.warning("Not enough data to train fair value model")
            return

        X = self._compute_features(candles, funding_rate)
        y = (candles["close"].shift(-1) > candles["close"]).astype(int).values

        valid = ~(np.isnan(X).any(axis=1)) & ~np.isnan(y)
        valid[-1] = False
        X = X[valid]
        y = y[valid]

        if len(X) < 100:
            logger.warning("Not enough valid rows to train")
            return

        self._base_rate = float(y.mean())

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True

        # Check raw and shrunk output ranges
        raw_probs = self.model.predict_proba(X_scaled)[:, 1]
        shrunk = self._shrink(raw_probs)
        train_acc = ((raw_probs > 0.5).astype(int) == y).mean()

        logger.info(
            f"Fair value model trained: {len(X)} samples, "
            f"train accuracy={train_acc:.3f}, base rate={self._base_rate:.3f}, "
            f"raw prob range=[{raw_probs.min():.3f}, {raw_probs.max():.3f}], "
            f"shrunk prob range=[{shrunk.min():.3f}, {shrunk.max():.3f}]"
        )

    def _shrink(self, probs: np.ndarray) -> np.ndarray:
        """Shrink probabilities toward the base rate.

        This is critical: a model with 55% accuracy should NOT output probabilities
        far from 0.5. Shrinkage = 0.3 means we only use 30% of the model's
        deviation from the base rate.
        """
        return self._base_rate + self.shrinkage * (probs - self._base_rate)

    def predict(self, candles: pd.DataFrame, funding_rate: float = 0.0) -> float:
        """Predict P(BTC up). Output is shrunk toward base rate."""
        if not self._fitted:
            return self._base_rate

        X = self._compute_features(candles, funding_rate)
        last_row = X[-1:]
        if np.isnan(last_row).any():
            return self._base_rate

        X_scaled = self.scaler.transform(last_row)
        raw_prob = self.model.predict_proba(X_scaled)[0][1]
        shrunk = self._base_rate + self.shrinkage * (raw_prob - self._base_rate)
        return float(np.clip(shrunk, 0.35, 0.65))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model, "scaler": self.scaler,
                "lookback": self.lookback, "base_rate": self._base_rate,
                "shrinkage": self.shrinkage,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.lookback = data["lookback"]
        self._base_rate = data.get("base_rate", 0.5)
        self.shrinkage = data.get("shrinkage", 0.3)
        self._fitted = True
