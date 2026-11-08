"""Strategy 1: Random baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AcquisitionStrategy


class RandomBaseline(AcquisitionStrategy):
    """均匀随机选择."""

    name = "Random"
    needs_retrain = False

    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        idx = rng.choice(len(pool_X), size=min(n, len(pool_X)), replace=False)
        return idx
