"""Strategy 2: GPR-UCB."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .base import AcquisitionStrategy


class GPRUCB(AcquisitionStrategy):
    """Gaussian Process Regression with UCB."""

    name = "GPR-UCB"

    def __init__(self, alpha: float = 1.96):
        super().__init__()
        self.alpha = alpha
        self.gpr: GaussianProcessRegressor | None = None
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X.values)
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
        self.gpr.fit(X_scaled, y.ravel())

    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if self.gpr is None:
            raise RuntimeError("GPR-UCB requires fit() before select_next()")
        X_scaled = self.scaler.transform(pool_X.values)
        mu, sigma = self.gpr.predict(X_scaled, return_std=True)
        ucb = mu.ravel() + self.alpha * sigma
        top = np.argsort(-ucb)[:n]
        return top
