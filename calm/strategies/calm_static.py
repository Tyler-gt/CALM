"""CALM with feedback-guided shortlist reranking disabled."""

from __future__ import annotations

from .calm import CALMAcquisition


class CALMStatic(CALMAcquisition):
    """Soft Ensemble-UCB fused with the immutable full-pool LLM ranking.

    The class shares every surrogate, EWRC, lambda, seed, and acquisition
    implementation detail with ``CALM``, but keeps the original target-blind
    static full-pool ranking and disables all feedback-conditioned rescoring.
    """

    name = "CALM-static"
    supports_dynamic_rescoring = False
