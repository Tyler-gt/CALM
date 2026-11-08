"""Adaptive-UCB: CALM's exact soft-ensemble surrogate path without LLM.

This is the matched data-only ablation for CALM.  It intentionally inherits
``fit`` and all ensemble weighting/prediction code from ``CALMAcquisition``;
only the knowledge-score input is removed during acquisition.
"""

from __future__ import annotations

from .calm import CALMAcquisition


class AdaptiveUCB(CALMAcquisition):
    """Soft GBDT/RFR/GPR/NN Ensemble-UCB with no LLM or dynamic rescoring."""

    name = "Adaptive-UCB"
    uses_llm_knowledge = False
    supports_dynamic_rescoring = False
    needs_al_feedback_context = False
