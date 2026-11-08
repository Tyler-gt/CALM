"""Strategy NN-UCB: fixed lightweight MC Dropout MLP + UCB.

This is the fixed-surrogate control; Adaptive-UCB is CALM's matched
data-only ablation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calm import CALMAcquisition


class NNUCB(CALMAcquisition):
    """MC Dropout MLP-UCB — 纯数据驱动, 无 LLM 知识门控.

    覆盖 fit() 强制锁定 NN 代理模型, 覆盖 select_next() 跳过 LLM 评分.
    """

    name = "NN-UCB"
    uses_llm_knowledge = False
    supports_dynamic_rescoring = False
    needs_al_feedback_context = False
    uses_soft_ensemble = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._train_nn(X, y)
        self._best_surrogate = "nn"

    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        return super().select_next(pool_X, None, n, rng)
