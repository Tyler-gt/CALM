"""Immutable static-knowledge control for the formal Hybrid strategy."""

from __future__ import annotations

from .hybrid import HybridAcquisition


class HybridStatic(HybridAcquisition):
    """Soft Ensemble-UCB fused with the immutable full-pool LLM ranking.

    The class shares every surrogate, EWRC, lambda, seed, and acquisition
    implementation detail with ``Hybrid``, but keeps the original target-blind
    static full-pool ranking and disables all feedback-conditioned rescoring.
    """

    name = "Hybrid-static"
    supports_dynamic_rescoring = False
