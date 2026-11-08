"""Strategy 4: bootstrap LightGBM ensemble UCB."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

from .base import AcquisitionStrategy


class GBDTUCB(AcquisitionStrategy):
    """Estimate epistemic uncertainty from independently bootstrapped GBDTs.

    Boosting-stage predictions are not exchangeable model samples: early stages
    are deliberately underfit, so their spread is not a valid predictive
    uncertainty.  This baseline instead fits several complete LightGBM models
    on bootstrap resamples and uses their between-model standard deviation.
    """

    name = "GBDT-UCB"
    needs_retrain = True

    def __init__(
        self,
        alpha: float = 1.96,
        n_estimators: int = 300,
        n_bootstrap_models: int = 5,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.n_estimators = int(n_estimators)
        self.n_bootstrap_models = max(2, int(n_bootstrap_models))
        self._models: list[LGBMRegressor] = []
        self._imputer: SimpleImputer | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._imputer = SimpleImputer(strategy="median")
        X_imp = self._imputer.fit_transform(X.values)
        target = np.asarray(y, dtype=float).ravel()
        n = len(target)
        max_leaves = min(31, max(2, n // 3))
        min_child = min(20, max(1, n // 8))

        self._models = []
        for seed in range(self.n_bootstrap_models):
            bootstrap_rng = np.random.RandomState(10_000 + seed)
            bootstrap = bootstrap_rng.choice(n, size=n, replace=True)
            model = LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                num_leaves=max_leaves,
                min_child_samples=min_child,
                random_state=42 + seed,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(X_imp[bootstrap], target[bootstrap])
            self._models.append(model)

    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if not self._models or self._imputer is None:
            raise RuntimeError("GBDT-UCB requires fit() before select_next()")

        X_imp = self._imputer.transform(pool_X.values)
        predictions = np.asarray([model.predict(X_imp) for model in self._models])
        mu = predictions.mean(axis=0)
        sigma = predictions.std(axis=0, ddof=1)
        return np.argsort(-(mu + self.alpha * sigma))[:n]
