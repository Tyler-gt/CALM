"""Pure feedback-driven LLM acquisition without a learned surrogate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calm.experiment import (
    DYNAMIC_RANK_FUSION_VERSION,
    LLM_DYNAMIC_SHORTLIST_POLICY_VERSION,
)

from .hybrid import HybridAcquisition


class LLMDynamic(HybridAcquisition):
    """Static knowledge plus feedback-updated LLM ranking, without UCB.

    The strategy intentionally reuses Hybrid's audited persistent M1, feedback,
    checkpoint, and static rank-slot machinery. It never fits or queries a
    surrogate. Shortlists are formed from static full-pool knowledge ranks plus
    a pre-registered random exploration quota, so no prediction mean,
    uncertainty, UCB, EWRC, or adaptive lambda can affect acquisition.
    """

    name = "LLM-dynamic"
    needs_retrain = False
    uses_llm_knowledge = True
    supports_dynamic_rescoring = True
    needs_al_feedback_context = True
    uses_soft_ensemble = False

    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        del pool_X
        if pool_scores is None or "s_phys" not in pool_scores:
            raise RuntimeError(
                "LLM-dynamic requires feedback-conditioned pool_scores['s_phys']"
            )

        llm_score = np.asarray(pool_scores["s_phys"], dtype=np.float64)
        tie_break = np.asarray(
            pool_scores.get("_tie_break_rank", np.arange(len(llm_score))),
            dtype=np.float64,
        )
        if (
            llm_score.ndim != 1
            or tie_break.shape != llm_score.shape
            or not np.all(np.isfinite(tie_break))
        ):
            raise ValueError(
                "LLM-dynamic received invalid knowledge scores or tie order"
            )

        base_llm_rank = self._rank_percentile(
            llm_score,
            tie_break=tie_break,
        )
        zeros = np.zeros_like(base_llm_rank)
        updated_scores = self._maybe_dynamic_rescore(
            pool_scores=pool_scores,
            mu=zeros,
            sigma=zeros,
            ucb_rank=zeros,
            static_acq=base_llm_rank,
            batch_size=n,
            rng=rng,
        )
        if updated_scores is None or "s_phys" not in updated_scores:
            raise RuntimeError("LLM-dynamic did not retain its knowledge scores")

        dynamic_score = np.asarray(updated_scores["s_phys"], dtype=np.float64)
        llm_rank = self._rerank_dynamic_subset_with_static_slots(
            static_llm_rank=base_llm_rank,
            dynamic_score=dynamic_score,
            subset_local=self._last_dynamic_subset_local,
        )
        self._last_llm_rho = 0.0
        self._last_llm_extreme_rho = 0.0
        self._last_lambda_eff = 1.0
        return np.lexsort((-tie_break, -llm_rank))[:n]

    def _select_dynamic_rescore_subset(
        self,
        *,
        mu: np.ndarray,
        sigma: np.ndarray,
        ucb_rank: np.ndarray,
        static_acq: np.ndarray,
        batch_size: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Select by knowledge rank plus random exploration, never by a model."""
        del mu, sigma, ucb_rank
        knowledge = np.asarray(static_acq, dtype=np.float64)
        n_pool = len(knowledge)
        if n_pool == 0:
            return np.array([], dtype=int)

        min_m = max(int(batch_size), self._dynamic_top_m_min_mult * int(batch_size))
        frac_m = int(np.ceil(self._dynamic_top_m_frac * n_pool))
        target_m = min(
            n_pool,
            self._dynamic_top_m_cap,
            max(min_m, frac_m),
        )
        random_quota = min(
            target_m,
            int(np.ceil(self._dynamic_random_frac * target_m)),
        )
        knowledge_quota = target_m - random_quota

        order = np.argsort(-knowledge, kind="mergesort")
        selected = [int(index) for index in order[:knowledge_quota]]
        selected_set = set(selected)
        if random_quota > 0:
            remaining = np.asarray(
                [index for index in range(n_pool) if index not in selected_set],
                dtype=int,
            )
            take = min(random_quota, len(remaining))
            if take:
                selected.extend(
                    int(index)
                    for index in rng.choice(remaining, size=take, replace=False)
                )

        if len(selected) < target_m:
            selected_set = set(selected)
            selected.extend(
                int(index)
                for index in order
                if int(index) not in selected_set
            )
        return np.asarray(selected[:target_m], dtype=int)

    def get_iteration_diagnostics(self) -> dict:
        return {
            "acquisition_mode": "llm_dynamic_only",
            "surrogate_mode": "none",
            "surrogate_weights": {},
            "oof_weighted_mae": {},
            "oof_spearman": {},
            "surrogate_sigma_scales": {},
            "oof_uncertainty_diagnostics": {},
            "weight_fit_train_size": 0,
            "ensemble_disagreement_mean": 0.0,
            "ensemble_within_sigma_mean": 0.0,
            "llm_rho": 0.0,
            "llm_rho_global": 0.0,
            "llm_rho_extreme": 0.0,
            "ewrc_gamma": None,
            "lambda_eff": 1.0,
            "data_acquisition_component": False,
            "knowledge_weight": 1.0,
            "llm_dynamic_shortlist_policy": (
                LLM_DYNAMIC_SHORTLIST_POLICY_VERSION
            ),
            "dynamic_rank_fusion": DYNAMIC_RANK_FUSION_VERSION,
            "knowledge_base": "static_full_pool",
            "dynamic_rescore_subset_size": int(
                len(self._last_dynamic_subset_local)
            ),
        }
