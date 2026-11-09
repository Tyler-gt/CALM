"""Abstract base class for acquisition strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AcquisitionStrategy(ABC):
    """Common interface for all acquisition strategies."""

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
        """Select n indices from the candidate pool.

        Parameters
        ----------
        pool_X : Candidate-pool feature matrix
        pool_scores : Optional precomputed scores (e.g. {"s_phys": array})
        n : Number of candidates to select
        rng : Random number generator

        Returns
        -------
        np.ndarray : Selected indices in the pool (integer array, shape (n,))
        """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Train or update on the current training set; overridden by model-based strategies."""
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
