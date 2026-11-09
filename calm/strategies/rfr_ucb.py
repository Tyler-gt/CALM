"""Strategy 3: RFR-UCB."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import AcquisitionStrategy


class RFRUCB(AcquisitionStrategy):
    """Random forest UCB with sigma from tree-prediction standard deviation (200 trees by default)."""

    name = "RFR-UCB"

    def __init__(self, alpha: float = 1.96, n_estimators: int = 200):
        super().__init__()
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.rfr: RandomForestRegressor | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.rfr = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=0,
            n_jobs=-1,
        )
        self.rfr.fit(X.values, y.ravel())

    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if self.rfr is None:
            raise RuntimeError("RFR-UCB requires fit() before select_next()")
        tree_preds = np.array([t.predict(pool_X.values) for t in self.rfr.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0, ddof=1)
        ucb = mu + self.alpha * sigma
        return np.argsort(-ucb)[:n]
