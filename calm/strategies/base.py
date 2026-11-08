"""Acquisition 策略抽象基类."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AcquisitionStrategy(ABC):
    """所有 acquisition 策略的统一接口."""

    name: str = "base"
    needs_retrain: bool = True
    uses_llm_knowledge: bool = False
    needs_al_feedback_context: bool = False

    @abstractmethod
    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """从 pool 中选择 n 个索引.

        Parameters
        ----------
        pool_X : 候选池特征矩阵
        pool_scores : 预计算分数 (e.g. {"s_phys": array}), 可选
        n : 要选择的数量
        rng : 随机数生成器

        Returns
        -------
        np.ndarray : pool 中被选中的索引 (int array, shape (n,))
        """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """用当前训练集训练/更新模型 (model-based strategies 重写)."""
        pass

    def export_run_state(self) -> dict:
        """Return JSON-serializable per-run state for iteration checkpoints."""
        return {}

    def import_run_state(self, state: dict) -> None:
        """Restore per-run state; fitted surrogate models are rebuilt each iteration."""
        del state

    def get_iteration_diagnostics(self) -> dict:
        """Return optional JSON-safe diagnostics after candidate selection."""
        return {}
