"""Adaptive-UCB: Hybrid's exact soft-ensemble surrogate path without LLM.

This is the matched data-only ablation for Hybrid.  It intentionally inherits
``fit`` and all ensemble weighting/prediction code from ``HybridAcquisition``;
only the knowledge-score input is removed during acquisition.
"""

from __future__ import annotations

from .hybrid import HybridAcquisition


class AdaptiveUCB(HybridAcquisition):
    """Soft GBDT/RFR/GPR/NN Ensemble-UCB with no LLM or dynamic rescoring."""

    name = "Adaptive-UCB"
    uses_llm_knowledge = False
    supports_dynamic_rescoring = False
    needs_al_feedback_context = False
