"""Strategy 5: LLM-rank — rank candidates by LLM physical score only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AcquisitionStrategy


class LLMRank(AcquisitionStrategy):
    """按 LLM 评分加性排序, 不依赖任何代理模型."""

    name = "LLM-rank"
    needs_retrain = False
    uses_llm_knowledge = True

    def __init__(self):
        super().__init__()

    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if pool_scores is None or "s_phys" not in pool_scores:
            raise ValueError("LLM-rank requires precomputed pool_scores with s_phys")
        phys = np.asarray(pool_scores["s_phys"], dtype=np.float64)
        tie_break = np.asarray(
            pool_scores.get("_tie_break_rank", np.arange(len(phys))),
            dtype=np.float64,
        )
        if tie_break.shape != phys.shape or not np.all(np.isfinite(tie_break)):
            raise ValueError("LLM-rank received an invalid deterministic tie order")
        # Higher LLM score is primary; higher canonical-hash rank resolves an
        # exact score tie.  This order is independent of dataframe position and
        # NumPy's default sort implementation.
        return np.lexsort((-tie_break, -phys))[:n]
