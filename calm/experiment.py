"""CALM method settings, fixed-budget defaults, and run identities."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path
from common.llm.config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_LLM_MODEL,
    get_api_base_url,
    get_llm_model,
)

from calm.reproducibility import (
    CANDIDATE_INDEX_AUDIT_VERSION,
    ENVIRONMENT_FINGERPRINT_VERSION,
    IMPLEMENTATION_FINGERPRINT_VERSION,
)
from calm.objective import (
    OBJECTIVE_DIRECTION_CONTRACT_VERSION,
)


PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT / "outputs"
EXPERIMENT_VERSION = "calm_paper_v1"

# Stable scoring salt: package names and output locations must not change the
# deterministic order among equal knowledge scores used in the paper.
KNOWLEDGE_TIE_SALT = (
    "sA_v33_objsrc_raw_mlp_anchor_memory_llmdynamic_ewrc_"
    "norag_dsv4pro_1kpertask_q4_b40_10r"
)

STRATEGY_SET_VERSION = "calm_paper_main_nine_strategies_v1"

SURROGATE_POLICY_VERSION = "matched_oof_calibrated_gbdt_rfr_gpr_mlp_v3"
NN_SURROGATE_VERSION = "standardized_mlp_64_32_16_gelu_mc_dropout30_v1"
SURROGATE_SIGMA_CALIBRATION_VERSION = "oof_abs_residual_q95_over_1p96_v1"
LAMBDA_MODE_VERSION = "adaptive_utility_extreme_weighted_spearman_v2"
DYNAMIC_RANK_FUSION_VERSION = "shortlist_static_rank_slot_permutation_v2"
LLM_DYNAMIC_SHORTLIST_POLICY_VERSION = (
    "static_rank_top90_random10_no_surrogate_signal_v2"
)
KNOWLEDGE_TIE_BREAK_VERSION = (
    "canonical_candidate_sha256_task_target_unit_protocol_v1"
)
DYNAMIC_FEEDBACK_POLICY_VERSION = (
    "shared_objective_anchor_best3_worst2_nearby3_plus_candidate_memory_m1_v3"
)
CANDIDATE_MEMORY_POLICY_VERSION = (
    "target_blind_llm_select_from_xonly_nearest5_persistent_measured_m1_v1"
)
CANDIDATE_MEMORY_PREFILTER_K = 5
DYNAMIC_PREQUERY_LEDGER_VERSION = (
    "append_only_scored_candidate_set_before_selection_v2"
)
ANCHOR_BEST_K = 3
ANCHOR_WORST_K = 2
ANCHOR_NEARBY_K = 3

LLM_SCORING_PROMPT_VERSION = (
    "target_specific_objective_anchor_relative_candidate_memory_m1_v12"
)
LLM_EVIDENCE_VERSION = (
    "candidate_reason_anchor_memory_selection_prequery_norag_jsonl_v9"
)
LLM_API_BASE_URL = get_api_base_url()
LLM_SCORING_MODEL = get_llm_model()
LLM_SCORING_TEMPERATURE = 0.0
LLM_RETRIEVAL_MODE = "disabled"
LLM_RETRY_POLICY_VERSION = "retry_transient_forever_no_static_fallback_v1"
LLM_KEY_SCHEDULER_VERSION = "single_assigned_key_local_cooldown_v2"
DATASET_KEY_ALLOCATION_VERSION = (
    "one_dataset_one_explicit_key_slot_release_on_process_exit_v1"
)
RESOURCE_ALLOCATION_POLICY_VERSION = (
    "pool100_init10ceil_batch2halfup_total40ceil_finalclip_v1"
)
REGRET_AUC_VERSION = (
    "objective_direction_train_size_target_range_budget_span_normalized_v3"
)
STOPPING_POLICY_VERSION = "fixed_total_budget_no_optimum_early_stop_v2"
ITERATION_CHECKPOINT_VERSION = (
    "per_iteration_multidynamic_objective_candidate_memory_fixed_budget_state_v9"
)
DYNAMIC_RESCORING_CACHE_VERSION = (
    "strategy_scoped_static_base_exact_candidate_memory_prompt_cache_v8"
)

INITIAL_FRACTION = 0.10
BATCH_FRACTION = 0.02
TOTAL_BUDGET_FRACTION = 0.40


def resolve_resource_allocation(
    n_samples: int,
    *,
    initial_fraction: float = INITIAL_FRACTION,
    batch_fraction: float = BATCH_FRACTION,
    total_budget_fraction: float = TOTAL_BUDGET_FRACTION,
) -> dict[str, int]:
    """Resolve one dataset's percentage protocol with deterministic rounding."""
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("Active learning requires at least two candidates")

    initial_fraction_decimal = Decimal(str(initial_fraction))
    batch_fraction_decimal = Decimal(str(batch_fraction))
    total_fraction_decimal = Decimal(str(total_budget_fraction))
    if not (
        Decimal("0") < initial_fraction_decimal < total_fraction_decimal <= Decimal("1")
    ):
        raise ValueError(
            "Require 0 < initial_fraction < total_budget_fraction <= 1"
        )
    if not Decimal("0") < batch_fraction_decimal <= Decimal("1"):
        raise ValueError("Require 0 < batch_fraction <= 1")

    sample_count = Decimal(n_samples)
    n_init = int(
        (sample_count * initial_fraction_decimal).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    batch_size = max(
        1,
        int(
            (sample_count * batch_fraction_decimal).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        ),
    )
    total_budget = min(
        n_samples,
        int(
            (sample_count * total_fraction_decimal).to_integral_value(
                rounding=ROUND_CEILING
            )
        ),
    )
    if not 0 < n_init < total_budget <= n_samples:
        raise ValueError(
            f"Rounded allocation is invalid for n={n_samples}: "
            f"n_init={n_init}, total_budget={total_budget}"
        )

    queried_after_initial = total_budget - n_init
    max_iter = (queried_after_initial + batch_size - 1) // batch_size
    final_batch_size = queried_after_initial - (max_iter - 1) * batch_size
    return {
        "n_samples": n_samples,
        "n_init": n_init,
        "batch_size": batch_size,
        "total_budget": total_budget,
        "max_iter": max_iter,
        "final_batch_size": final_batch_size,
    }

# Runs use their own caches. Dynamic knowledge reranks static shortlist slots
# with measured anchors and persistent target-blind M1 references.
STRATEGY_NAMES = (
    "Random",
    "GPR-UCB",
    "RFR-UCB",
    "GBDT-UCB",
    "NN-UCB",
    "Adaptive-UCB",
    "LLM-rank",
    "LLM-dynamic",
    "CALM",
)

DYNAMIC_STRATEGY_NAMES = (
    "LLM-dynamic",
    "CALM",
)

DYNAMIC_EVIDENCE_FILENAMES = {
    "LLM-dynamic": "_llm_dynamic_evidence.jsonl",
    "CALM": "_dynamic_llm_evidence.jsonl",
}

# Default method settings. CLI overrides use a distinct output subdirectory.
DEFAULT_PROTOCOL = {
    "n_runs": 10,
    "candidate_pool_fraction": 1.0,
    "initial_fraction": INITIAL_FRACTION,
    "batch_fraction": BATCH_FRACTION,
    "total_budget_fraction": TOTAL_BUDGET_FRACTION,
    "resource_allocation_policy": RESOURCE_ALLOCATION_POLICY_VERSION,
    "objective_direction_contract": OBJECTIVE_DIRECTION_CONTRACT_VERSION,
    "regret_auc_version": REGRET_AUC_VERSION,
    "stopping_policy": STOPPING_POLICY_VERSION,
    "iteration_checkpoint": ITERATION_CHECKPOINT_VERSION,
    "implementation_fingerprint_version": IMPLEMENTATION_FINGERPRINT_VERSION,
    "environment_fingerprint_version": ENVIRONMENT_FINGERPRINT_VERSION,
    "candidate_index_audit": CANDIDATE_INDEX_AUDIT_VERSION,
    "seed": None,
    "initialization": "paper_selected_runs",
    "experiment": "benchmark",
    "strategy_names": list(STRATEGY_NAMES),
    "fixed_lambda": None,
    "feature_representation": "raw",
    "lam_llm": 0.6,
    "lambda_mode": LAMBDA_MODE_VERSION,
    "ewrc_gamma": 2.0,
    "knowledge_tie_break": KNOWLEDGE_TIE_BREAK_VERSION,
    "knowledge_tie_salt": KNOWLEDGE_TIE_SALT,
    "ucb_alpha": 1.96,
    "surrogate_policy": SURROGATE_POLICY_VERSION,
    "nn_surrogate": NN_SURROGATE_VERSION,
    "surrogate_sigma_calibration": SURROGATE_SIGMA_CALIBRATION_VERSION,
    "ensemble_min_train_samples": 40,
    "ensemble_weight_update_samples": 25,
    "ensemble_weight_floor": 0.05,
    "ensemble_weight_temperature": 0.35,
    "ensemble_top_weight_strength": 2.0,
    "gbdt_bootstrap_models": 5,
    "rescoring_interval": 1,
    "llm_scoring_prompt": LLM_SCORING_PROMPT_VERSION,
    "llm_evidence_version": LLM_EVIDENCE_VERSION,
    "llm_scoring_model": DEFAULT_LLM_MODEL,
    "llm_api_base_url": DEFAULT_API_BASE_URL,
    "llm_scoring_temperature": LLM_SCORING_TEMPERATURE,
    "llm_retrieval_mode": LLM_RETRIEVAL_MODE,
    "dynamic_top_m_frac": 0.25,
    "dynamic_top_m_min_mult": 5,
    "dynamic_top_m_cap": 200,
    "dynamic_random_frac": 0.10,
    "dynamic_rank_fusion": DYNAMIC_RANK_FUSION_VERSION,
    "llm_dynamic_shortlist_policy": LLM_DYNAMIC_SHORTLIST_POLICY_VERSION,
    "dynamic_strategy_names": list(DYNAMIC_STRATEGY_NAMES),
    "dynamic_evidence_files": dict(DYNAMIC_EVIDENCE_FILENAMES),
    "dynamic_rescoring_cache": DYNAMIC_RESCORING_CACHE_VERSION,
    "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
    "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
    "candidate_memory_prefilter_k": CANDIDATE_MEMORY_PREFILTER_K,
    "dynamic_prequery_ledger": DYNAMIC_PREQUERY_LEDGER_VERSION,
    "anchor_best_k": ANCHOR_BEST_K,
    "anchor_worst_k": ANCHOR_WORST_K,
    "anchor_nearby_k": ANCHOR_NEARBY_K,
    "llm_retry_policy": LLM_RETRY_POLICY_VERSION,
    "llm_retry_until_success": True,
    "llm_strict_scoring": True,
    "llm_max_concurrency": 10,
    "llm_api_key_count": 1,
    "llm_per_key_concurrency": 10,
    "llm_key_scheduler": LLM_KEY_SCHEDULER_VERSION,
    "dataset_key_allocation": DATASET_KEY_ALLOCATION_VERSION,
    "llm_rate_limit_batch_size": 10,
    "llm_rate_limit_sleep_seconds": 10.0,
    "llm_retry_base_seconds": 1.0,
    "llm_retry_max_seconds": 60.0,
    "dynamic_rescoring_failure_policy": "stop_never_static_fallback_v1",
}

OUTPUT_DIR = OUTPUT_ROOT / EXPERIMENT_VERSION
