"""CALM and its acquisition baselines on paired, fixed-budget alloy searches."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from calm.data import (
    COLUMN_ROLE_SCHEMA_VERSION,
    COMPLETE_COMPOSITION_POLICY,
    INITIAL_FEATURE_FIT_VERSION,
    PROCESS_FEATURE_ENCODING_VERSION,
    REPORTED_COMPOSITION_POLICY,
    _llm_detect_column_roles,
    assert_no_target_leakage,
    dataframe_fingerprint,
    load_external_dataset_initial_fit,
    normalize_mixed_composition_units,
    read_dataset_table,
    validate_model_ready_dataset,
)
from calm.experiment import (
    ANCHOR_BEST_K,
    ANCHOR_NEARBY_K,
    ANCHOR_WORST_K,
    CANDIDATE_MEMORY_POLICY_VERSION,
    CANDIDATE_MEMORY_PREFILTER_K,
    COMPATIBLE_FEATURE_CACHE_EXPERIMENTS,
    COMPATIBLE_LLM_CACHE_EXPERIMENTS,
    DYNAMIC_FEEDBACK_POLICY_VERSION,
    DYNAMIC_EVIDENCE_FILENAMES,
    DYNAMIC_PREQUERY_LEDGER_VERSION,
    DYNAMIC_RESCORING_CACHE_VERSION,
    DYNAMIC_STRATEGY_NAMES,
    DATASET_KEY_ALLOCATION_VERSION,
    DATASET_QUEUE_MAX_WORKERS,
    DATASET_QUEUE_ORDER_VERSION,
    KNOWLEDGE_SCORE_CONTROL_VERSION,
    EXPERIMENT_VERSION,
    ITERATION_CHECKPOINT_VERSION,
    KNOWLEDGE_TIE_BREAK_VERSION,
    KNOWLEDGE_TIE_SALT,
    LLM_EVIDENCE_VERSION,
    LLM_KEY_SCHEDULER_VERSION,
    LLM_DYNAMIC_SHORTLIST_POLICY_VERSION,
    LLM_RETRIEVAL_POLICY_VERSION,
    LLM_RETRIEVAL_MODE,
    LLM_RETRIEVAL_TOP_K,
    LLM_RETRY_POLICY_VERSION,
    LLM_API_BASE_URL,
    LLM_SCORING_MODEL,
    LLM_SCORING_PROMPT_VERSION,
    LLM_SCORING_TEMPERATURE,
    LLM_USE_HYDE,
    OUTPUT_DIR,
    EXPLORATORY_INFERENCE_VERSION,
    PRIMARY_INFERENCE_COMPARATORS,
    PRIMARY_INFERENCE_STRATEGY,
    PRIMARY_INFERENCE_VERSION,
    PRODUCTION_PROTOCOL,
    REGRET_AUC_VERSION,
    RESOURCE_ALLOCATION_POLICY_VERSION,
    STRATEGY_NAMES,
    STRATEGY_SET_VERSION,
    STOPPING_POLICY_VERSION,
    SURROGATE_POLICY_VERSION,
    SURROGATE_SIGMA_CALIBRATION_VERSION,
    resolve_resource_allocation,
)
from calm.metrics import compute_stats
from calm.plot import plot_convergence
from calm.reproducibility import (
    CANDIDATE_INDEX_AUDIT_VERSION,
    candidate_index_fingerprint,
    environment_fingerprint,
    implementation_fingerprint,
)
from calm.objective import (
    OBJECTIVE_DIRECTION_CONTRACT_VERSION,
    best_observed,
    is_global_optimum as objective_is_global_optimum,
    objective_optimum,
    to_utility,
    validate_objective_direction,
)
from calm.strategies.random_baseline import RandomBaseline
from calm.strategies.gpr_ucb import GPRUCB
from calm.strategies.rfr_ucb import RFRUCB
from calm.strategies.gbdt_ucb import GBDTUCB
from calm.strategies.nn_ucb import NNUCB
from calm.strategies.adaptive_ucb import AdaptiveUCB
from calm.strategies.llm_rank import LLMRank
from calm.strategies.llm_dynamic import LLMDynamic
from calm.strategies.hybrid import HybridAcquisition
from calm.strategies.hybrid_static import HybridStatic
from common.llm.key_pool import AsyncUSTCKeyPool, require_ustc_api_keys

EXTERNAL = PROJECT / "dabase"
CURATED = EXTERNAL / "curated"
N_RUNS = int(PRODUCTION_PROTOCOL["n_runs"])
INITIAL_LABELED_FRACTION = float(PRODUCTION_PROTOCOL["initial_fraction"])
BATCH_QUERY_FRACTION = float(PRODUCTION_PROTOCOL["batch_fraction"])
TOTAL_LABELED_BUDGET_FRACTION = float(
    PRODUCTION_PROTOCOL["total_budget_fraction"]
)
SEED = int(PRODUCTION_PROTOCOL["seed"])
FEATURE_REPRESENTATION = str(PRODUCTION_PROTOCOL["feature_representation"])
LAM_LLM = float(PRODUCTION_PROTOCOL["lam_llm"])
EWRC_GAMMA = float(PRODUCTION_PROTOCOL["ewrc_gamma"])
KNOWLEDGE_SCORE_CONTROL = str(PRODUCTION_PROTOCOL["knowledge_score_control"])
UCB_ALPHA = float(PRODUCTION_PROTOCOL["ucb_alpha"])
ENSEMBLE_MIN_TRAIN_SAMPLES = int(PRODUCTION_PROTOCOL["ensemble_min_train_samples"])
ENSEMBLE_WEIGHT_UPDATE_SAMPLES = int(
    PRODUCTION_PROTOCOL["ensemble_weight_update_samples"]
)
ENSEMBLE_WEIGHT_FLOOR = float(PRODUCTION_PROTOCOL["ensemble_weight_floor"])
ENSEMBLE_WEIGHT_TEMPERATURE = float(
    PRODUCTION_PROTOCOL["ensemble_weight_temperature"]
)
ENSEMBLE_TOP_WEIGHT_STRENGTH = float(
    PRODUCTION_PROTOCOL["ensemble_top_weight_strength"]
)
GBDT_BOOTSTRAP_MODELS = int(PRODUCTION_PROTOCOL["gbdt_bootstrap_models"])
SURROGATE_SIGMA_CALIBRATION = str(
    PRODUCTION_PROTOCOL["surrogate_sigma_calibration"]
)
RESCORING_INTERVAL = int(PRODUCTION_PROTOCOL["rescoring_interval"])
DYNAMIC_TOP_M_FRAC = float(PRODUCTION_PROTOCOL["dynamic_top_m_frac"])
DYNAMIC_TOP_M_MIN_MULT = int(PRODUCTION_PROTOCOL["dynamic_top_m_min_mult"])
DYNAMIC_TOP_M_CAP = int(PRODUCTION_PROTOCOL["dynamic_top_m_cap"])
DYNAMIC_RANDOM_FRAC = float(PRODUCTION_PROTOCOL["dynamic_random_frac"])
LLM_RETRY_UNTIL_SUCCESS = bool(PRODUCTION_PROTOCOL["llm_retry_until_success"])
LLM_STRICT_SCORING = bool(PRODUCTION_PROTOCOL["llm_strict_scoring"])
LLM_MAX_CONCURRENCY = int(PRODUCTION_PROTOCOL["llm_max_concurrency"])
LLM_API_KEY_COUNT = int(PRODUCTION_PROTOCOL["llm_api_key_count"])
LLM_PER_KEY_CONCURRENCY = int(PRODUCTION_PROTOCOL["llm_per_key_concurrency"])
LLM_RATE_LIMIT_BATCH_SIZE = int(PRODUCTION_PROTOCOL["llm_rate_limit_batch_size"])
LLM_RATE_LIMIT_SLEEP_SECONDS = float(
    PRODUCTION_PROTOCOL["llm_rate_limit_sleep_seconds"]
)
LLM_RETRY_BASE_SECONDS = float(PRODUCTION_PROTOCOL["llm_retry_base_seconds"])
LLM_RETRY_MAX_SECONDS = float(PRODUCTION_PROTOCOL["llm_retry_max_seconds"])
API_KEY_SLOT = 1
STRATEGY_SEED_OFFSETS = {
    "Random": 101,
    "GPR-UCB": 211,
    "RFR-UCB": 307,
    "GBDT-UCB": 401,
    "NN-UCB": 503,
    # Matched common-random-number pair with Hybrid.
    "Adaptive-UCB": 701,
    "LLM-rank": 601,
    "LLM-dynamic": 701,
    "Hybrid-static": 701,
    "Hybrid": 701,
}


# ═══════════════════════════════════════════════════════════════════════
# Dataset Registry
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DatasetConfig:
    """数据集最小描述 — 元素列和工艺列自动检测, 无需手写."""
    name: str
    # Legacy public field name retained for compatibility; CSV is also valid.
    xlsx_path: Path
    target_col: str
    composition_unit: str
    objective_direction: str
    objective_description: str = ""
    material_system: str = ""          # LLM 领域知识提示 (可选)
    target_label: str = ""             # 绘图 y 轴标签 (可选, 默认=target_col)
    target_unit: str = "Value"         # 绘图单位 (可选)
    normalize_mixed_units: bool = False
    composition_missing_policy: str = COMPLETE_COMPOSITION_POLICY
    # Measured auxiliary properties that must never be model/LLM inputs.
    additional_property_cols: tuple[str, ...] = ()
    expected_target_valid_rows: int | None = None
    require_dedup_manifest: bool = True

    def __post_init__(self) -> None:
        self.objective_direction = validate_objective_direction(
            self.objective_direction
        )
        if self.composition_unit not in {"wt.%", "at.%"}:
            raise ValueError(
                "composition_unit must be exactly 'wt.%' or 'at.%', got "
                f"{self.composition_unit!r}"
            )
        if self.composition_missing_policy not in {
            COMPLETE_COMPOSITION_POLICY,
            REPORTED_COMPOSITION_POLICY,
        }:
            raise ValueError(
                "Unsupported composition_missing_policy: "
                f"{self.composition_missing_policy!r}"
            )

    @property
    def data_path(self) -> Path:
        """Format-neutral alias used by the current loader."""
        return self.xlsx_path


def _detect_columns(
    df: pd.DataFrame,
    target_col: str,
    data_path: Path,
    known_target_cols: list[str],
    *,
    api_key: str | None = None,
) -> dict[str, list[str]]:
    """Detect four column roles with all known properties forced as targets."""
    return _llm_detect_column_roles(
        df,
        target_col,
        xlsx_path=data_path,
        known_target_cols=known_target_cols,
        api_key=api_key,
        model=LLM_SCORING_MODEL,
        base_url=LLM_API_BASE_URL,
    )


FECONI_PROPERTIES = (
    "Coercivity_mT",
    "Kerr_rotation_mrad",
)
MPEA_PROPERTIES = (
    "Grain_size_um",
    "Experimental_density_g_cm3",
    "Calculated_density_g_cm3",
    "Hardness_HV",
    "YS_MPa",
    "UTS_MPa",
    "Elongation_percent",
    "Plastic_elongation_percent",
    "Experimental_Young_modulus_GPa",
    "Calculated_Young_modulus_GPa",
    "O_content_wppm",
    "N_content_wppm",
    "C_content_wppm",
)


DATASETS: dict[str, DatasetConfig] = {
    "2xxx_T6__UTS": DatasetConfig(
        name="2xxx_T6__UTS",
        xlsx_path=CURATED / "al_alloys.csv",
        target_col="UTS",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Al-alloy",
        expected_target_valid_rows=138,
    ),
    "2xxx_T6__EL": DatasetConfig(
        name="2xxx_T6__EL",
        xlsx_path=CURATED / "al_alloys.csv",
        target_col="EL",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="EL", target_unit="%",
        material_system="Al-alloy",
        expected_target_valid_rows=134,
    ),
    "Cu_alloys__EC": DatasetConfig(
        name="Cu_alloys__EC",
        xlsx_path=CURATED / "Cu_alloys__EC.csv",
        target_col="Electrical_conductivity_pctIACS",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="Conductivity", target_unit="%IACS",
        material_system="Cu-alloy",
        expected_target_valid_rows=1486,
    ),
    "mg_alloy__UTS": DatasetConfig(
        name="mg_alloy__UTS",
        xlsx_path=CURATED / "mg_alloy__UTS.csv",
        target_col="UTS_MPa",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Mg-alloy",
        expected_target_valid_rows=289,
    ),
    "mg_alloy__YS": DatasetConfig(
        name="mg_alloy__YS",
        xlsx_path=CURATED / "mg_alloy__YS.csv",
        target_col="YS_MPa",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="YS", target_unit="MPa",
        material_system="Mg-alloy",
        expected_target_valid_rows=298,
    ),
    "steel__UTS": DatasetConfig(
        name="steel__UTS",
        xlsx_path=CURATED / "steel__UTS.csv",
        target_col="Ultimate_Tensile_Strength_MPa",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Steel",
        additional_property_cols=(
            "Microstructure",
            "Grain_size_um",
            "Yield_Strength_MPa",
            "Uniform_Elongation_percent",
            "Total_Elongation_percent",
        ),
        composition_missing_policy=REPORTED_COMPOSITION_POLICY,
        expected_target_valid_rows=272,
    ),
    "steel_fatigue__Fatigue": DatasetConfig(
        name="steel_fatigue__Fatigue",
        xlsx_path=EXTERNAL / "steel_fatigue__Fatigue.csv",
        target_col="Fatigue",
        composition_unit="wt.%",
        objective_direction="maximize",
        objective_description=(
            "Maximize rotating-bending fatigue strength at 10^7 cycles for "
            "NIMS steels. NT is normalizing temperature; THT/THt/THQCr are "
            "through-hardening temperature/time/cooling rate; CT/Ct are "
            "carburization temperature/time; DT/Dt are diffusion "
            "temperature/time; QmT is quenching-media temperature; TT/Tt/TCr "
            "are tempering temperature/time/cooling rate; RedRatio is the "
            "ingot-to-bar reduction ratio; dA/dB/dC are inclusion-area "
            "proportions. A zero THt, Ct, or Tt means that process stage was "
            "not applied, so its paired filled temperature must not be "
            "interpreted independently."
        ),
        target_label="Fatigue strength",
        target_unit="MPa",
        material_system="NIMS carbon, low-alloy, carburizing, and spring steels",
        composition_missing_policy=REPORTED_COMPOSITION_POLICY,
        expected_target_valid_rows=437,
    ),
    "Ti_alloy__UTS": DatasetConfig(
        name="Ti_alloy__UTS",
        xlsx_path=CURATED / "Ti_alloy__UTS.csv",
        target_col="UTS_MPa",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Ti-alloy",
        additional_property_cols=("moe",),
        expected_target_valid_rows=185,
    ),
    "Ti_alloy__YS": DatasetConfig(
        name="Ti_alloy__YS",
        xlsx_path=CURATED / "Ti_alloy__YS.csv",
        target_col="YS_MPa",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="YS", target_unit="MPa",
        material_system="Ti-alloy",
        additional_property_cols=("moe",),
        expected_target_valid_rows=248,
    ),
    # External diversity expansion: primary tasks.
    "FeCoNi__Coercivity": DatasetConfig(
        name="FeCoNi__Coercivity",
        xlsx_path=EXTERNAL / "FeCoNi__Magnetics.csv",
        target_col="Coercivity_mT",
        composition_unit="at.%",
        objective_direction="minimize",
        objective_description=(
            "Discover low-coercivity Fe-Co-Ni soft-magnetic compositions."
        ),
        target_label="Coercivity", target_unit="mT",
        material_system="Fe-Co-Ni combinatorial thin-film alloy",
        additional_property_cols=FECONI_PROPERTIES,
        expected_target_valid_rows=921,
    ),
    "FeCoNi__Kerr": DatasetConfig(
        name="FeCoNi__Kerr",
        xlsx_path=EXTERNAL / "FeCoNi__Magnetics.csv",
        target_col="Kerr_rotation_mrad",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="Kerr rotation", target_unit="mrad",
        material_system="Fe-Co-Ni combinatorial thin-film alloy",
        additional_property_cols=FECONI_PROPERTIES,
        expected_target_valid_rows=921,
    ),
    "MetallicGlass__Dmax": DatasetConfig(
        name="MetallicGlass__Dmax",
        xlsx_path=EXTERNAL / "MetallicGlass__Dmax.csv",
        target_col="Dmax_mm",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="Dmax", target_unit="mm",
        material_system="Metallic glass",
        expected_target_valid_rows=747,
    ),
    "HEA_Invar__TEC": DatasetConfig(
        name="HEA_Invar__TEC",
        xlsx_path=EXTERNAL / "HEA_Invar__TEC.csv",
        target_col="TEC_1e-6_K-1",
        composition_unit="at.%",
        objective_direction="minimize",
        objective_description=(
            "Discover high-entropy Invar compositions with a low thermal "
            "expansion coefficient."
        ),
        target_label="TEC",
        target_unit="10^-6 K^-1",
        material_system="Fe-Ni-Co-Cr-V-Cu high-entropy Invar alloy",
        expected_target_valid_rows=709,
    ),
    "MPEA__YS": DatasetConfig(
        name="MPEA__YS",
        xlsx_path=EXTERNAL / "MPEA__Mechanical.csv",
        target_col="YS_MPa",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="YS", target_unit="MPa",
        material_system="Multi-principal-element alloy",
        additional_property_cols=MPEA_PROPERTIES,
        expected_target_valid_rows=760,
    ),
    # External diversity expansion: supplementary/robustness tasks.
    "MPEA__Hardness": DatasetConfig(
        name="MPEA__Hardness",
        xlsx_path=EXTERNAL / "MPEA__Mechanical.csv",
        target_col="Hardness_HV",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="Hardness", target_unit="HV",
        material_system="Multi-principal-element alloy",
        additional_property_cols=MPEA_PROPERTIES,
        expected_target_valid_rows=463,
    ),
    "MPEA__UTS": DatasetConfig(
        name="MPEA__UTS",
        xlsx_path=EXTERNAL / "MPEA__Mechanical.csv",
        target_col="UTS_MPa",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Multi-principal-element alloy",
        additional_property_cols=MPEA_PROPERTIES,
        expected_target_valid_rows=362,
    ),
    "MPEA__Elongation": DatasetConfig(
        name="MPEA__Elongation",
        xlsx_path=EXTERNAL / "MPEA__Mechanical.csv",
        target_col="Elongation_percent",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="Elongation", target_unit="%",
        material_system="Multi-principal-element alloy",
        additional_property_cols=MPEA_PROPERTIES,
        expected_target_valid_rows=424,
    ),
}


def _registered_property_cols(config: DatasetConfig) -> list[str]:
    """Collect every registered target sharing the same source plus overrides."""
    source = config.data_path.resolve()
    known: list[str] = []
    for other in DATASETS.values():
        if other.data_path.resolve() == source and other.target_col not in known:
            known.append(other.target_col)
    for col in config.additional_property_cols:
        if col not in known:
            known.append(col)
    return known


# ═══════════════════════════════════════════════════════════════════════
# Strategy Construction
# ═══════════════════════════════════════════════════════════════════════

def build_strategies() -> list:
    ensemble_kwargs = {
        "ensemble_min_train_samples": ENSEMBLE_MIN_TRAIN_SAMPLES,
        "ensemble_weight_update_samples": ENSEMBLE_WEIGHT_UPDATE_SAMPLES,
        "ensemble_weight_floor": ENSEMBLE_WEIGHT_FLOOR,
        "ensemble_weight_temperature": ENSEMBLE_WEIGHT_TEMPERATURE,
        "ensemble_top_weight_strength": ENSEMBLE_TOP_WEIGHT_STRENGTH,
        "gbdt_bootstrap_models": GBDT_BOOTSTRAP_MODELS,
    }
    return [
        RandomBaseline(),
        GPRUCB(alpha=UCB_ALPHA),
        RFRUCB(alpha=UCB_ALPHA, n_estimators=200),
        GBDTUCB(
            alpha=UCB_ALPHA,
            n_estimators=300,
            n_bootstrap_models=GBDT_BOOTSTRAP_MODELS,
        ),
        NNUCB(alpha=UCB_ALPHA),
        AdaptiveUCB(alpha=UCB_ALPHA, lam_llm=0.0, **ensemble_kwargs),
        LLMRank(),
        LLMDynamic(),
        HybridStatic(
            alpha=UCB_ALPHA,
            lam_llm=LAM_LLM,
            extreme_rank_power=EWRC_GAMMA,
            **ensemble_kwargs,
        ),
        HybridAcquisition(
            alpha=UCB_ALPHA,
            lam_llm=LAM_LLM,
            extreme_rank_power=EWRC_GAMMA,
            **ensemble_kwargs,
        ),
    ]


def _strategy_seed(run_seed: int, strategy_name: str) -> int:
    """Stable per-strategy seed; Python's built-in hash is process-randomized."""
    offset = STRATEGY_SEED_OFFSETS.get(
        strategy_name,
        sum((i + 1) * ord(ch) for i, ch in enumerate(strategy_name)),
    )
    return int((run_seed + offset) % (2**31 - 1))


def _resolve_resources(n_samples: int) -> dict[str, int]:
    return resolve_resource_allocation(
        n_samples,
        initial_fraction=INITIAL_LABELED_FRACTION,
        batch_fraction=BATCH_QUERY_FRACTION,
        total_budget_fraction=TOTAL_LABELED_BUDGET_FRACTION,
    )


def _runtime_protocol() -> dict:
    return {
        "n_runs": N_RUNS,
        "candidate_pool_fraction": 1.0,
        "initial_fraction": INITIAL_LABELED_FRACTION,
        "batch_fraction": BATCH_QUERY_FRACTION,
        "total_budget_fraction": TOTAL_LABELED_BUDGET_FRACTION,
        "resource_allocation_policy": RESOURCE_ALLOCATION_POLICY_VERSION,
        "objective_direction_contract": OBJECTIVE_DIRECTION_CONTRACT_VERSION,
        "regret_auc_version": REGRET_AUC_VERSION,
        "stopping_policy": STOPPING_POLICY_VERSION,
        "iteration_checkpoint": ITERATION_CHECKPOINT_VERSION,
        "implementation_fingerprint_version": PRODUCTION_PROTOCOL[
            "implementation_fingerprint_version"
        ],
        "environment_fingerprint_version": PRODUCTION_PROTOCOL[
            "environment_fingerprint_version"
        ],
        "candidate_index_audit": CANDIDATE_INDEX_AUDIT_VERSION,
        "seed": SEED,
        "feature_representation": FEATURE_REPRESENTATION,
        "lam_llm": LAM_LLM,
        "lambda_mode": PRODUCTION_PROTOCOL["lambda_mode"],
        "ewrc_gamma": EWRC_GAMMA,
        "knowledge_score_control": KNOWLEDGE_SCORE_CONTROL,
        "knowledge_score_control_version": KNOWLEDGE_SCORE_CONTROL_VERSION,
        "knowledge_tie_break": KNOWLEDGE_TIE_BREAK_VERSION,
        "knowledge_tie_salt": KNOWLEDGE_TIE_SALT,
        "ucb_alpha": UCB_ALPHA,
        "surrogate_policy": SURROGATE_POLICY_VERSION,
        "nn_surrogate": PRODUCTION_PROTOCOL["nn_surrogate"],
        "surrogate_sigma_calibration": SURROGATE_SIGMA_CALIBRATION,
        "ensemble_min_train_samples": ENSEMBLE_MIN_TRAIN_SAMPLES,
        "ensemble_weight_update_samples": ENSEMBLE_WEIGHT_UPDATE_SAMPLES,
        "ensemble_weight_floor": ENSEMBLE_WEIGHT_FLOOR,
        "ensemble_weight_temperature": ENSEMBLE_WEIGHT_TEMPERATURE,
        "ensemble_top_weight_strength": ENSEMBLE_TOP_WEIGHT_STRENGTH,
        "gbdt_bootstrap_models": GBDT_BOOTSTRAP_MODELS,
        "rescoring_interval": RESCORING_INTERVAL,
        "llm_scoring_prompt": LLM_SCORING_PROMPT_VERSION,
        "llm_evidence_version": LLM_EVIDENCE_VERSION,
        "llm_scoring_model": LLM_SCORING_MODEL,
        "llm_api_base_url": LLM_API_BASE_URL,
        "llm_scoring_temperature": LLM_SCORING_TEMPERATURE,
        "llm_retrieval_top_k": LLM_RETRIEVAL_TOP_K,
        "llm_use_hyde": LLM_USE_HYDE,
        "llm_retrieval_mode": LLM_RETRIEVAL_MODE,
        "llm_retrieval_policy": LLM_RETRIEVAL_POLICY_VERSION,
        "dynamic_top_m_frac": DYNAMIC_TOP_M_FRAC,
        "dynamic_top_m_min_mult": DYNAMIC_TOP_M_MIN_MULT,
        "dynamic_top_m_cap": DYNAMIC_TOP_M_CAP,
        "dynamic_random_frac": DYNAMIC_RANDOM_FRAC,
        "dynamic_rank_fusion": PRODUCTION_PROTOCOL["dynamic_rank_fusion"],
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
        "llm_retry_until_success": LLM_RETRY_UNTIL_SUCCESS,
        "llm_strict_scoring": LLM_STRICT_SCORING,
        "llm_max_concurrency": LLM_MAX_CONCURRENCY,
        "llm_api_key_count": LLM_API_KEY_COUNT,
        "llm_per_key_concurrency": LLM_PER_KEY_CONCURRENCY,
        "llm_key_scheduler": LLM_KEY_SCHEDULER_VERSION,
        "dataset_key_allocation": DATASET_KEY_ALLOCATION_VERSION,
        "dataset_queue_order": DATASET_QUEUE_ORDER_VERSION,
        "dataset_queue_max_workers": DATASET_QUEUE_MAX_WORKERS,
        "llm_rate_limit_batch_size": LLM_RATE_LIMIT_BATCH_SIZE,
        "llm_rate_limit_sleep_seconds": LLM_RATE_LIMIT_SLEEP_SECONDS,
        "llm_retry_base_seconds": LLM_RETRY_BASE_SECONDS,
        "llm_retry_max_seconds": LLM_RETRY_MAX_SECONDS,
        "dynamic_rescoring_failure_policy": (
            PRODUCTION_PROTOCOL["dynamic_rescoring_failure_policy"]
        ),
    }


def _controlled_knowledge_scores(
    scores: np.ndarray,
    *,
    dataset_name: str,
    run_seed: int,
) -> tuple[np.ndarray, dict]:
    """Return the run-specific knowledge channel and its compact audit record.

    The shuffled control preserves the exact score multiset while breaking its
    candidate assignment.  Its seed is derived from immutable protocol/run
    identifiers, so checkpoint resume and independent reruns reproduce the
    same permutation without storing a large index vector.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Knowledge scores must be a non-empty finite 1-D array")

    source_score_digest = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    multiset_digest = hashlib.sha256(
        np.sort(values).tobytes(order="C")
    ).hexdigest()
    if KNOWLEDGE_SCORE_CONTROL == "observed":
        return values, {
            "mode": "observed",
            "version": KNOWLEDGE_SCORE_CONTROL_VERSION,
            "run_seed": int(run_seed),
            "source_score_sha256": source_score_digest,
            "score_multiset_sha256": multiset_digest,
            "permutation_seed": None,
            "permutation_sha256": None,
        }
    if KNOWLEDGE_SCORE_CONTROL != "shuffled":
        raise ValueError(
            "knowledge_score_control must be 'observed' or 'shuffled', got "
            f"{KNOWLEDGE_SCORE_CONTROL!r}"
        )

    seed_payload = (
        f"{KNOWLEDGE_SCORE_CONTROL_VERSION}|{KNOWLEDGE_TIE_SALT}|"
        f"{dataset_name}|{int(run_seed)}"
    ).encode("utf-8")
    permutation_seed = int.from_bytes(
        hashlib.sha256(seed_payload).digest()[:4], "big", signed=False
    )
    permutation = np.random.RandomState(permutation_seed).permutation(len(values))
    controlled = values[permutation]
    if not np.array_equal(np.sort(controlled), np.sort(values)):
        raise RuntimeError("Shuffled knowledge control changed the score multiset")
    return controlled, {
        "mode": "shuffled",
        "version": KNOWLEDGE_SCORE_CONTROL_VERSION,
        "run_seed": int(run_seed),
        "source_score_sha256": source_score_digest,
        "score_multiset_sha256": multiset_digest,
        "permutation_seed": int(permutation_seed),
        "permutation_sha256": hashlib.sha256(
            permutation.astype(np.int64).tobytes(order="C")
        ).hexdigest(),
    }


def _controlled_pool_scores(
    scores: dict[str, np.ndarray],
    *,
    dataset_name: str,
    run_seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Expose only the protocol's ``s_phys`` channel to acquisition.

    Static scoring also returns raw component scores for evidence auditing.
    Those fields are not acquisition inputs under ``s_phys_only_v1`` and must
    not be passed directly to the one-dimensional paired-control helper.
    """
    if not isinstance(scores, dict) or "s_phys" not in scores:
        raise ValueError("Static LLM scores must contain the s_phys channel")
    controlled, audit = _controlled_knowledge_scores(
        scores["s_phys"],
        dataset_name=dataset_name,
        run_seed=run_seed,
    )
    return {"s_phys": controlled}, audit


def _knowledge_tie_break_ranks(
    candidates: list[dict],
    *,
    dataset_name: str,
    target_col: str,
    composition_unit: str,
    objective_direction: str = "maximize",
) -> tuple[np.ndarray, dict]:
    """Return a deterministic total order for candidates with equal LLM scores.

    The order is scientific-protocol state rather than an implementation
    accident such as dataframe position or NumPy sort behavior.  Duplicate
    canonical candidate identities fail closed because they would not admit a
    candidate-specific hash order under the declared protocol.
    """
    if composition_unit not in {"wt.%", "at.%"}:
        raise ValueError("composition_unit must be exactly 'wt.%' or 'at.%'")
    objective_direction = validate_objective_direction(objective_direction)
    keys: list[str] = []
    for candidate in candidates:
        payload = {
            "candidate": _canonical_evidence_value(candidate),
            "dataset": str(dataset_name),
            "target_col": str(target_col),
            "composition_unit": str(composition_unit),
            "objective_direction": objective_direction,
            "experiment_version": KNOWLEDGE_TIE_SALT,
            "tie_break_version": KNOWLEDGE_TIE_BREAK_VERSION,
        }
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        keys.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "Canonical candidate identities are not unique under the knowledge "
            "tie-break protocol"
        )
    order = np.argsort(np.asarray(keys, dtype="U64"), kind="mergesort")
    ranks = np.empty(len(keys), dtype=np.float64)
    ranks[order] = np.arange(len(keys), dtype=np.float64)
    if len(keys) > 1:
        ranks /= float(len(keys) - 1)
    digest = hashlib.sha256("".join(keys).encode("ascii")).hexdigest()
    return ranks, {
        "version": KNOWLEDGE_TIE_BREAK_VERSION,
        "candidate_key_sequence_sha256": digest,
        "n_candidates": len(keys),
    }


def _feature_fit_scope() -> str:
    """Describe exactly which information may shape the representation."""
    if FEATURE_REPRESENTATION == "physics_llm":
        return "initial_labels_plus_full_pool_x"
    return "full_pool_x_only_target_independent"


def _output_namespace() -> tuple[Path, str]:
    runtime = _runtime_protocol()
    if runtime == PRODUCTION_PROTOCOL:
        return OUTPUT_DIR, "production"
    raw = json.dumps(runtime, sort_keys=True, separators=(",", ":"))
    protocol_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return OUTPUT_DIR / "_custom" / protocol_id, "custom"


# ═══════════════════════════════════════════════════════════════════════
# AL Loop (shared, also imported by aggregate.py)
# ═══════════════════════════════════════════════════════════════════════

def _serialize_rng_state(rng: np.random.RandomState) -> dict:
    name, state, position, has_gauss, cached_gaussian = rng.get_state()
    return {
        "bit_generator": name,
        "state": state.astype(np.uint32).tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_rng_state(rng: np.random.RandomState, state: dict) -> None:
    rng.set_state((
        str(state["bit_generator"]),
        np.asarray(state["state"], dtype=np.uint32),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    ))


def _is_global_optimum(
    value: float,
    *,
    global_min: float,
    global_max: float,
    objective_direction: str,
) -> bool:
    return objective_is_global_optimum(
        value,
        global_min=global_min,
        global_max=global_max,
        direction=objective_direction,
    )


def _checkpoint_progress(payload: dict) -> tuple[int, int, int, int]:
    """Return a monotonic progress key for compatible CALM checkpoints."""
    records = payload.get("records", {})
    completed = (
        sum(len(runs) for runs in records.values() if isinstance(runs, list))
        if isinstance(records, dict)
        else 0
    )
    active = payload.get("in_progress") or {}
    run_idx = int(active.get("run_idx", -1))
    try:
        strategy_idx = list(STRATEGY_NAMES).index(active.get("strategy"))
    except ValueError:
        strategy_idx = -1
    try:
        iteration = int(active["state"]["records"][-1]["iteration"])
    except (KeyError, IndexError, TypeError, ValueError):
        iteration = -1
    return completed, run_idx, strategy_idx, iteration


def _load_latest_checkpoint(
    checkpoint_path: Path,
    expected_config: dict,
) -> tuple[dict, Path]:
    """Load the most advanced valid checkpoint, including failed-replace temps."""
    candidates = [checkpoint_path]
    candidates.extend(
        path
        for path in checkpoint_path.parent.glob(f"{checkpoint_path.stem}*.tmp")
        if path != checkpoint_path
    )
    valid: list[tuple[tuple[int, int, int, int], int, dict, Path]] = []
    saw_parseable = False
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        saw_parseable = True
        if payload.get("_config") != expected_config:
            continue
        try:
            modified_ns = path.stat().st_mtime_ns
        except OSError:
            modified_ns = 0
        valid.append((_checkpoint_progress(payload), modified_ns, payload, path))

    if not valid:
        reason = "config mismatch" if saw_parseable else "no valid checkpoint JSON"
        raise ValueError(reason)
    _, _, payload, source = max(valid, key=lambda item: (item[0], item[1]))
    return payload, source


def _atomic_write_checkpoint(
    checkpoint_path: Path,
    payload: dict,
    *,
    replace_attempts: int = 8,
) -> bool:
    """Write JSON safely while tolerating transient Windows destination locks.

    When all replace attempts fail, the complete unique ``.tmp`` file remains
    available for automatic recovery and the expensive AL process can continue.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False))
        stream.flush()
        os.fsync(stream.fileno())

    attempts = max(1, int(replace_attempts))
    for attempt in range(attempts):
        try:
            tmp.replace(checkpoint_path)
            for stale in checkpoint_path.parent.glob(
                f"{checkpoint_path.stem}*.tmp"
            ):
                try:
                    stale.unlink()
                except OSError:
                    pass
            return True
        except PermissionError as exc:
            if attempt + 1 >= attempts:
                print(
                    f"[checkpoint] WARNING: Windows kept {checkpoint_path.name} "
                    f"locked after {attempts} attempts; latest state remains in "
                    f"{tmp.name} and the AL run will continue ({exc})",
                    flush=True,
                )
                return False
            delay = min(1.0, 0.05 * (2 ** attempt))
            print(
                f"[checkpoint] {checkpoint_path.name} is temporarily locked; "
                f"retrying in {delay:.2f}s ({attempt + 1}/{attempts})",
                flush=True,
            )
            time.sleep(delay)
    return False


def run_single_al_loop(
    strategy,
    X: pd.DataFrame,
    y: np.ndarray,
    pool_scores: dict | None,
    initial_idx: np.ndarray,
    batch_size: int,
    total_budget: int,
    rng: np.random.RandomState,
    result_label: str,
    objective_direction: str = "maximize",
    resume_state: dict | None = None,
    initial_strategy_state: dict | None = None,
    initial_record_metadata: dict | None = None,
    checkpoint_callback: Callable[[dict], None] | None = None,
) -> list[dict]:
    if hasattr(strategy, "reset_run_state"):
        strategy.reset_run_state()

    objective_direction = validate_objective_direction(objective_direction)
    n_total = len(X)
    y = np.asarray(y, dtype=np.float64)
    utility_y = to_utility(y, objective_direction)
    initial_idx = np.asarray(initial_idx, dtype=int)
    batch_size = int(batch_size)
    total_budget = int(total_budget)
    if len(y) != n_total:
        raise ValueError(f"X/y length mismatch: {n_total} != {len(y)}")
    if initial_idx.ndim != 1 or len(initial_idx) == 0:
        raise ValueError("initial_idx must be a non-empty 1D array")
    if len(np.unique(initial_idx)) != len(initial_idx):
        raise ValueError("initial_idx contains duplicate candidates")
    if initial_idx.min() < 0 or initial_idx.max() >= n_total:
        raise IndexError("initial_idx contains an out-of-range candidate")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not len(initial_idx) < total_budget <= n_total:
        raise ValueError(
            f"Require initial_size < total_budget <= pool_size; got "
            f"{len(initial_idx)} < {total_budget} <= {n_total}"
        )
    max_iter = (
        total_budget - len(initial_idx) + batch_size - 1
    ) // batch_size
    global_min = float(np.nanmin(y))
    global_max = float(np.nanmax(y))
    initial_indices = [int(index) for index in initial_idx]
    initial_fingerprint = candidate_index_fingerprint(initial_indices)

    if resume_state is not None:
        train_idx = [int(index) for index in resume_state["train_idx"]]
        records = [dict(record) for record in resume_state["records"]]
        if not records:
            raise ValueError("Resume state contains no AL records")
        if len(records) > max_iter + 1:
            raise ValueError("Resume state contains records beyond the fixed budget")
        if records[0].get("initial_candidate_indices") != initial_indices:
            raise ValueError("Resume state initial candidate indices do not match")
        if records[0].get("initial_index_fingerprint") != initial_fingerprint:
            raise ValueError("Resume state initial index fingerprint does not match")
        if initial_record_metadata is not None:
            for key, value in initial_record_metadata.items():
                if records[0].get(key) != value:
                    raise ValueError(
                        f"Resume state initial metadata mismatch for {key!r}"
                    )
        reconstructed_train_idx = list(initial_indices)
        queried = set(initial_indices)
        for expected_iteration, record in enumerate(records[1:], start=1):
            if int(record.get("iteration", -1)) != expected_iteration:
                raise ValueError("Resume state iteration sequence is incomplete")
            selected = record.get("selected_candidate_indices")
            expected_batch_size = min(
                batch_size,
                total_budget - len(reconstructed_train_idx),
            )
            if (
                not isinstance(selected, list)
                or len(selected) != expected_batch_size
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or index >= n_total
                    or index in queried
                    for index in selected
                )
                or len(set(selected)) != expected_batch_size
            ):
                raise ValueError(
                    "Resume state has invalid or repeated selected candidate indices"
                )
            if record.get(
                "selected_index_fingerprint"
            ) != candidate_index_fingerprint(selected):
                raise ValueError(
                    "Resume state selected index fingerprint does not match"
                )
            queried.update(selected)
            reconstructed_train_idx.extend(selected)
        if reconstructed_train_idx != train_idx:
            raise ValueError(
                "Resume state train_idx does not match its audited query history"
            )
        _restore_rng_state(rng, resume_state["rng_state"])
        if hasattr(strategy, "import_run_state"):
            strategy.import_run_state(resume_state.get("strategy_state", {}))
    else:
        if initial_strategy_state is not None and hasattr(
            strategy, "import_run_state"
        ):
            strategy.import_run_state(initial_strategy_state)
        train_idx = [int(index) for index in initial_idx]
        global_best = best_observed(y[train_idx], objective_direction)
        found = _is_global_optimum(
            global_best,
            global_min=global_min,
            global_max=global_max,
            objective_direction=objective_direction,
        )
        initial_record = {
            "iteration": 0,
            result_label: global_best,
            "train_size": len(train_idx),
            "initial_candidate_indices": initial_indices,
            "initial_index_fingerprint": initial_fingerprint,
            "global_optimum_found": found,
            "first_optimum_iteration": 0 if found else None,
            "samples_to_optimum": len(train_idx) if found else None,
            "stop_reason": "in_progress",
        }
        if initial_record_metadata is not None:
            overlap = set(initial_record) & set(initial_record_metadata)
            if overlap:
                raise ValueError(
                    "initial_record_metadata may not replace AL audit fields: "
                    f"{sorted(overlap)}"
                )
            initial_record.update(initial_record_metadata)
        records = [initial_record]

    if len(train_idx) > total_budget:
        raise ValueError(
            f"Resume state exceeds total budget: {len(train_idx)} > {total_budget}"
        )
    first_optimum_iteration = next(
        (
            int(record["first_optimum_iteration"])
            for record in records
            if record.get("first_optimum_iteration") is not None
        ),
        None,
    )
    samples_to_optimum = next(
        (
            int(record["samples_to_optimum"])
            for record in records
            if record.get("samples_to_optimum") is not None
        ),
        None,
    )

    pool_mask = np.ones(n_total, dtype=bool)
    pool_mask[np.asarray(train_idx, dtype=int)] = False
    global_best = best_observed(y[train_idx], objective_direction)

    def _checkpoint() -> None:
        if checkpoint_callback is None:
            return
        strategy_state = (
            strategy.export_run_state()
            if hasattr(strategy, "export_run_state")
            else {}
        )
        checkpoint_callback({
            "train_idx": [int(index) for index in train_idx],
            "records": records,
            "rng_state": _serialize_rng_state(rng),
            "strategy_state": strategy_state,
        })

    records[-1]["stop_reason"] = "in_progress"
    _checkpoint()
    start_iteration = int(records[-1]["iteration"]) + 1

    for it in range(start_iteration, max_iter + 1):

        pool_indices = np.where(pool_mask)[0]
        remaining_budget = total_budget - len(train_idx)
        if len(pool_indices) == 0 or remaining_budget <= 0:
            break
        current_batch_size = min(batch_size, len(pool_indices), remaining_budget)

        train_X = X.iloc[train_idx]
        train_y = utility_y[train_idx]
        pool_X = X.iloc[pool_indices]

        if pool_scores is not None:
            pool_scores_subset = {
                k: v[pool_indices] for k, v in pool_scores.items()
            }
        else:
            pool_scores_subset = None

        if strategy.needs_retrain:
            strategy.fit(train_X, train_y)

        if (
            getattr(strategy, "needs_al_feedback_context", False)
            and hasattr(strategy, "set_al_context")
        ):
            strategy.set_al_context(
                iteration=it,
                pool_indices=pool_indices,
                train_idx=np.array(train_idx),
                y_all=utility_y,
                all_scores=pool_scores,
            )

        visible_pool_scores = (
            pool_scores_subset
            if getattr(strategy, "uses_llm_knowledge", False)
            else None
        )
        selected_local = strategy.select_next(
            pool_X,
            visible_pool_scores,
            current_batch_size,
            rng,
        )
        selected_local = np.asarray(selected_local)
        if selected_local.ndim != 1:
            raise RuntimeError(
                f"{strategy.name} returned a non-1D candidate index array"
            )
        if len(selected_local) != current_batch_size:
            raise RuntimeError(
                f"{strategy.name} selected {len(selected_local)} candidates; "
                f"expected {current_batch_size}"
            )
        if not np.issubdtype(selected_local.dtype, np.integer):
            raise RuntimeError(
                f"{strategy.name} returned non-integer candidate indices"
            )
        selected_local = selected_local.astype(int, copy=False)
        if len(np.unique(selected_local)) != len(selected_local):
            raise RuntimeError(
                f"{strategy.name} returned duplicate candidate indices"
            )
        if selected_local.min() < 0 or selected_local.max() >= len(pool_indices):
            raise RuntimeError(
                f"{strategy.name} returned an out-of-range candidate index"
            )
        selected_global = pool_indices[selected_local]

        train_idx.extend(selected_global.tolist())
        pool_mask[selected_global] = False

        global_best = best_observed(
            np.asarray(
                [
                    global_best,
                    best_observed(y[selected_global], objective_direction),
                ],
                dtype=np.float64,
            ),
            objective_direction,
        )

        found = _is_global_optimum(
            global_best,
            global_min=global_min,
            global_max=global_max,
            objective_direction=objective_direction,
        )
        if found and first_optimum_iteration is None:
            first_optimum_iteration = int(it)
            samples_to_optimum = int(len(train_idx))
        diagnostics = (
            strategy.get_iteration_diagnostics()
            if hasattr(strategy, "get_iteration_diagnostics")
            else {}
        )
        record = {
            "iteration": it,
            result_label: global_best,
            "train_size": len(train_idx),
            "selected_candidate_indices": [
                int(index) for index in selected_global.tolist()
            ],
            "selected_index_fingerprint": candidate_index_fingerprint(
                selected_global.tolist()
            ),
            "global_optimum_found": found,
            "first_optimum_iteration": first_optimum_iteration,
            "samples_to_optimum": samples_to_optimum,
            "stop_reason": "in_progress",
        }
        if diagnostics:
            record["acquisition_diagnostics"] = diagnostics
        records.append(record)
        _checkpoint()

    if len(train_idx) != total_budget:
        raise RuntimeError(
            f"{strategy.name} stopped at train_size={len(train_idx)} instead of "
            f"the fixed total_budget={total_budget}"
        )
    records[-1]["stop_reason"] = "total_budget_reached"
    _checkpoint()
    return records


# ═══════════════════════════════════════════════════════════════════════
# LLM Scoring
# ═══════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path) -> dict:
    """Return a portable content fingerprint for one cache dependency."""
    if not path.exists():
        return {"name": path.name, "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "name": path.name,
        "exists": True,
        "size": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


@lru_cache(maxsize=1)
def _llm_runtime_fingerprint() -> dict:
    """Fingerprint every active model/retrieval input that can change scores."""
    base = {
        "api_base_url": LLM_API_BASE_URL,
        "scoring_model": LLM_SCORING_MODEL,
        "temperature": LLM_SCORING_TEMPERATURE,
        "top_k": LLM_RETRIEVAL_TOP_K,
        "use_hyde": LLM_USE_HYDE,
        "retrieval_mode": LLM_RETRIEVAL_MODE,
        "retrieval_policy": LLM_RETRIEVAL_POLICY_VERSION,
        "prompt_implementation": _sha256_file(
            PROJECT / "common" / "llm" / "prompts.py"
        ),
        "retrieval_implementation": [
            _sha256_file(PROJECT / "common" / "llm" / "reviewer.py"),
        ],
    }
    if LLM_RETRIEVAL_MODE == "disabled":
        return {
            **base,
            "bm25_required": False,
            "rag_artifacts": [],
            "index_implementation": [],
        }

    from common.llm.reviewer import (
        RAG_MAX_PER_SOURCE,
        RAG_SECTION_FILTER,
        RAG_SNIPPET_MAX_CHARS,
    )
    from common.rag.identity import (
        BM25_DOCUMENT_VERSION,
        CHUNK_ID_VERSION,
        INDEX_ID_SCHEMA_VERSION,
    )
    from common.rag.process_normalization import PROCESS_NORMALIZATION_VERSION
    from common.rag.retrieve import (
        BM25_PATH,
        EMBED_DIM,
        META_PATH,
        MODEL as EMBEDDING_MODEL,
        VEC_PATH,
    )
    from common.rag.tokenization import BM25_TOKENIZATION_VERSION

    return {
        **base,
        "section_filter": list(RAG_SECTION_FILTER),
        "snippet_max_chars": int(RAG_SNIPPET_MAX_CHARS),
        "max_per_source": int(RAG_MAX_PER_SOURCE),
        "bm25_weight": 0.4,
        "bm25_tokenization": BM25_TOKENIZATION_VERSION,
        "bm25_document": BM25_DOCUMENT_VERSION,
        "process_normalization": PROCESS_NORMALIZATION_VERSION,
        "chunk_id_version": CHUNK_ID_VERSION,
        "index_id_schema_version": INDEX_ID_SCHEMA_VERSION,
        "bm25_required": True,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": int(EMBED_DIM),
        "rag_artifacts": [
            _sha256_file(Path(VEC_PATH)),
            _sha256_file(Path(META_PATH)),
            _sha256_file(Path(BM25_PATH)),
        ],
        "retrieval_implementation": [
            *base["retrieval_implementation"],
            _sha256_file(PROJECT / "common" / "rag" / "retrieve.py"),
            _sha256_file(PROJECT / "common" / "rag" / "tokenization.py"),
            _sha256_file(
                PROJECT / "common" / "rag" / "process_normalization.py"
            ),
        ],
        "index_implementation": [
            _sha256_file(PROJECT / "common" / "rag" / "identity.py"),
            _sha256_file(PROJECT / "common" / "rag" / "build_index.py"),
            _sha256_file(PROJECT / "common" / "rag" / "embed_chunks.py"),
        ],
    }


def _llm_evidence_path(cache_path: Path) -> Path:
    return cache_path.with_name("_llm_evidence.jsonl")


def _canonical_evidence_value(value):
    """Normalize candidate values to the exact JSON representation on disk."""
    if isinstance(value, np.generic):
        return _canonical_evidence_value(value.item())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _canonical_evidence_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_evidence_value(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


_STATIC_EVIDENCE_SCORE_FIELDS = (
    "s_phys",
    "s_target_raw",
    "s_phys_raw",
    "s_general_plausibility_raw",
)


def _static_evidence_generation_id(records: list[dict]) -> str:
    """Hash one complete static evidence generation in candidate order."""
    ordered = sorted(records, key=lambda record: int(record["candidate_index"]))
    canonical = _canonical_evidence_value(ordered)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_sha256(value) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _static_score_vectors(
    value,
    expected_length: int,
) -> dict[str, list[float]] | None:
    """Return finite cache score vectors required by the evidence contract."""
    if not isinstance(value, dict):
        return None
    vectors: dict[str, list[float]] = {}
    for field in _STATIC_EVIDENCE_SCORE_FIELDS:
        raw = value.get(field)
        if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != expected_length:
            return None
        try:
            numeric = [float(item) for item in raw]
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(item) for item in numeric):
            return None
        vectors[field] = numeric
    return vectors


def _retrieval_queries_complete(value) -> bool:
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(value.get(key), str) and bool(value[key].strip())
        for key in ("dense", "sparse")
    )


def _references_complete(value) -> bool:
    if not isinstance(value, list) or not value:
        return False
    chunk_ids: set[str] = set()
    for reference in value:
        if not isinstance(reference, dict):
            return False
        chunk_id = str(reference.get("chunk_id", "")).strip()
        text_sha256 = str(reference.get("text_sha256", "")).strip().lower()
        snippet = reference.get("snippet")
        if (
            not chunk_id
            or chunk_id in chunk_ids
            or len(text_sha256) != 64
            or any(char not in "0123456789abcdef" for char in text_sha256)
            or not isinstance(snippet, str)
            or not snippet.strip()
        ):
            return False
        chunk_ids.add(chunk_id)
    return True


def _static_evidence_complete(
    path: Path,
    expected_length: int,
    expected_candidates: list[dict] | None = None,
    *,
    expected_generation_id: str | None = None,
    expected_scores: dict | None = None,
    expected_composition_unit: str | None = None,
    expected_objective_direction: str | None = None,
    expected_retrieval_mode: str = "full",
) -> bool:
    """Require one score-bound evidence generation for every cache row."""
    generation_id = str(expected_generation_id or "").strip().lower()
    score_vectors = _static_score_vectors(expected_scores, int(expected_length))
    if (
        not path.exists()
        or expected_candidates is None
        or len(expected_candidates) != int(expected_length)
        or not _valid_sha256(generation_id)
        or score_vectors is None
    ):
        return False
    canonical_candidates = [
        _canonical_evidence_value(candidate) for candidate in expected_candidates
    ]
    indices: set[int] = set()
    generation_records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("evidence_version") != LLM_EVIDENCE_VERSION:
                    return False
                candidate_index = int(record["candidate_index"])
                if candidate_index in indices or not 0 <= candidate_index < expected_length:
                    return False
                if record.get("phase") != "static":
                    return False
                if (
                    expected_composition_unit is not None
                    and record.get("composition_unit")
                    != expected_composition_unit
                ):
                    return False
                if (
                    expected_objective_direction is not None
                    and record.get("objective_direction")
                    != expected_objective_direction
                ):
                    return False
                if record.get("retrieval_mode") != expected_retrieval_mode:
                    return False
                if str(record.get("evidence_generation_id", "")).lower() != generation_id:
                    return False
                if expected_retrieval_mode == "disabled":
                    evidence_payload_ok = (
                        record.get("retrieval_queries") == {}
                        and record.get("references") == []
                    )
                else:
                    evidence_payload_ok = (
                        _retrieval_queries_complete(
                            record.get("retrieval_queries")
                        )
                        and _references_complete(record.get("references"))
                    )
                if (
                    _canonical_evidence_value(record.get("candidate_input"))
                    != canonical_candidates[candidate_index]
                    or not evidence_payload_ok
                ):
                    return False
                for field, vector in score_vectors.items():
                    try:
                        evidence_score = float(record[field])
                    except (KeyError, TypeError, ValueError, OverflowError):
                        return False
                    if (
                        not math.isfinite(evidence_score)
                        or not math.isclose(
                            evidence_score,
                            vector[candidate_index],
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        return False
                generation_record = dict(record)
                generation_record.pop("evidence_generation_id", None)
                generation_records.append(generation_record)
                indices.add(candidate_index)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return False
    expected_indices = set(range(int(expected_length)))
    if indices != expected_indices:
        return False
    try:
        return _static_evidence_generation_id(generation_records) == generation_id
    except (KeyError, TypeError, ValueError):
        return False


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_scalar,
            ))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def _json_scalar(value):
    """Serialize pandas/numpy scalar values without changing numeric meaning."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _dataframe_input_records(
    frame: pd.DataFrame,
    columns: list[str],
) -> list[dict]:
    """Serialize source-missing inputs as JSON null, never non-standard NaN."""
    selected = frame.loc[:, columns].astype(object)
    selected = selected.where(pd.notna(selected), None)
    return selected.to_dict("records")


def _static_evidence_records(
    results: list[dict],
    candidates: list[dict],
    composition_unit: str,
    objective_direction: str = "maximize",
) -> list[dict]:
    objective_direction = validate_objective_direction(objective_direction)
    records = []
    for index, (result, candidate) in enumerate(zip(results, candidates)):
        record = {
            "evidence_version": LLM_EVIDENCE_VERSION,
            "phase": "static",
            "composition_unit": composition_unit,
            "objective_direction": objective_direction,
            "retrieval_mode": str(result.get("_retrieval_mode", "full")),
            "candidate_index": int(index),
            "candidate_input": candidate,
            "s_phys": float(result.get("s_phys", 0.0)),
            "s_target_raw": float(result.get("s_target_raw", 0.0)),
            "s_phys_raw": float(
                result.get("s_phys_raw", result.get("s_target_raw", 0.0))
            ),
            "s_general_plausibility_raw": float(
                result.get("s_general_plausibility_raw", 0.0)
            ),
            "principles": result.get("principles", []),
            "off_target_notes": result.get("off_target_notes", []),
            "reason": str(result.get("reason", "")),
            "retrieval_queries": result.get("_retrieval_queries", {}),
            "references": result.get("_refs", []),
            "key_alias": str(result.get("_key_alias", "unknown")),
        }
        records.append(record)
    return records


def precompute_llm_scores(
    df: pd.DataFrame,
    comp_df: pd.DataFrame,
    target_col: str,
    element_cols: list[str],
    process_cols: list[str],
    material_system: str,
    composition_unit: str,
    objective_direction: str,
    cache_path: Path,
    objective_description: str = "",
    cache_metadata: dict | None = None,
    forbidden_target_cols: list[str] | None = None,
    key_pool: AsyncUSTCKeyPool | None = None,
    evidence_path: Path | None = None,
) -> dict[str, np.ndarray]:
    evidence_path = evidence_path or _llm_evidence_path(cache_path)
    blocked_targets = list(dict.fromkeys([target_col, *(forbidden_target_cols or [])]))
    candidate_cols: list[str] = []
    for col in [*element_cols, *process_cols]:
        if col in blocked_targets or col not in df.columns or col in candidate_cols:
            continue
        candidate_cols.append(col)
    assert_no_target_leakage(
        candidate_cols,
        blocked_targets,
        context="static LLM candidate payload",
    )
    candidates = _dataframe_input_records(df, candidate_cols)

    if cache_path.exists():
        print(f"[llm] Loading cached scores from {cache_path.name}")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata_ok = True
        if cache_metadata is not None:
            metadata_ok = cache.get("_cache_metadata") == cache_metadata
        score_vectors = _static_score_vectors(cache, len(df))
        evidence_ok = _static_evidence_complete(
            evidence_path,
            len(df),
            candidates,
            expected_generation_id=cache.get("_evidence_generation_id"),
            expected_scores=cache,
            expected_composition_unit=composition_unit,
            expected_objective_direction=objective_direction,
            expected_retrieval_mode=LLM_RETRIEVAL_MODE,
        )
        if (
            score_vectors is not None
            and metadata_ok
            and evidence_ok
        ):
            scores = {"s_phys": np.array(cache["s_phys"], dtype=np.float64)}
            if "s_target_raw" in cache and len(cache["s_target_raw"]) == len(df):
                scores["s_target_raw"] = np.array(cache["s_target_raw"], dtype=np.float64)
            if (
                "s_general_plausibility_raw" in cache
                and len(cache["s_general_plausibility_raw"]) == len(df)
            ):
                scores["s_general_plausibility_raw"] = np.array(
                    cache["s_general_plausibility_raw"], dtype=np.float64
                )
            if "s_phys_raw" in cache and len(cache["s_phys_raw"]) == len(df):
                scores["s_phys_raw"] = np.array(cache["s_phys_raw"], dtype=np.float64)
            return scores
        if cache_metadata is not None and not metadata_ok:
            print("[llm] Cache metadata mismatch; recomputing")
        elif not evidence_ok:
            print(
                "[llm] Cache evidence generation is missing, incomplete, or "
                "score-mismatched; recomputing"
            )
        else:
            print(
                f"[llm] Cache score vectors are missing/invalid for "
                f"data length {len(df)}; recomputing"
            )

    from common.llm.reviewer import LLMReviewer

    stats = {}
    for col in element_cols:
        if col not in comp_df.columns:
            continue
        c = col.lower()
        stats[f"{c}_min"] = float(comp_df[col].min())
        stats[f"{c}_max"] = float(comp_df[col].max())
        stats[f"{c}_med"] = float(comp_df[col].median())

    reviewer = LLMReviewer(
        model=LLM_SCORING_MODEL,
        temperature=LLM_SCORING_TEMPERATURE,
        top_k=LLM_RETRIEVAL_TOP_K,
        use_hyde=LLM_USE_HYDE,
        retrieval_mode=LLM_RETRIEVAL_MODE,
        material_system=material_system,
        composition_unit=composition_unit,
        element_cols=element_cols,
        process_cols=process_cols,
        target_col=target_col,
        objective_direction=objective_direction,
        objective_description=objective_description,
        max_concurrency=LLM_MAX_CONCURRENCY,
        rate_limit_batch_size=LLM_RATE_LIMIT_BATCH_SIZE,
        rate_limit_sleep_seconds=LLM_RATE_LIMIT_SLEEP_SECONDS,
        retry_until_success=LLM_RETRY_UNTIL_SUCCESS,
        retry_base_seconds=LLM_RETRY_BASE_SECONDS,
        retry_max_seconds=LLM_RETRY_MAX_SECONDS,
        strict_scoring=LLM_STRICT_SCORING,
        min_api_keys=LLM_API_KEY_COUNT,
        per_key_concurrency=LLM_PER_KEY_CONCURRENCY,
        key_pool=key_pool,
    )
    reviewer.set_dataset(df, target_col=target_col,
                         element_cols=element_cols, process_cols=process_cols)

    print(
        f"[llm] Candidate payload excludes target={target_col}; "
        f"visible input columns={len(candidate_cols)}"
    )

    async def _score_all():
        # Keep one reviewer session for the full pool. LLMReviewer itself
        # enforces the registered request batches and waits; an outer chunk
        # loop would restart that batch-to-batch schedule every 50 candidates.
        return await reviewer.review_batch(candidates)

    print(f"[llm] Scoring {len(candidates)} candidates (material={material_system})...")
    results = asyncio.run(_score_all())
    key_alias_counts = Counter(
        str(result.get("_key_alias", "unknown")) for result in results
    )
    print(f"[llm] Key usage: {dict(sorted(key_alias_counts.items()))}")

    s_phys = np.array([r.get("s_phys", 0.0) for r in results], dtype=np.float64)
    s_target_raw = np.array(
        [r.get("s_target_raw", round(float(s) * 100.0)) for r, s in zip(results, s_phys)],
        dtype=np.float64,
    )
    s_general_plausibility_raw = np.array(
        [
            r.get("s_general_plausibility_raw", t)
            for r, t in zip(results, s_target_raw)
        ],
        dtype=np.float64,
    )
    s_phys_raw = np.array(
        [r.get("s_phys_raw", t) for r, t in zip(results, s_target_raw)],
        dtype=np.float64,
    )

    cache_data = {
        "s_phys": s_phys.tolist(),
        "s_target_raw": s_target_raw.tolist(),
        "s_general_plausibility_raw": s_general_plausibility_raw.tolist(),
        "s_phys_raw": s_phys_raw.tolist(),
        "_key_alias_counts": dict(sorted(key_alias_counts.items())),
        "_cache_metadata": cache_metadata or {},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_records = _static_evidence_records(
        results,
        candidates,
        composition_unit,
        objective_direction,
    )
    evidence_generation_id = _static_evidence_generation_id(evidence_records)
    for record in evidence_records:
        record["evidence_generation_id"] = evidence_generation_id
    cache_data["_evidence_generation_id"] = evidence_generation_id
    _write_jsonl_atomic(evidence_path, evidence_records)
    if not _static_evidence_complete(
        evidence_path,
        len(candidates),
        candidates,
        expected_generation_id=evidence_generation_id,
        expected_scores=cache_data,
        expected_composition_unit=composition_unit,
        expected_objective_direction=objective_direction,
        expected_retrieval_mode=LLM_RETRIEVAL_MODE,
    ):
        raise RuntimeError(
            "Static LLM evidence is incomplete: require one content-hashed "
            "generation with exact candidates, scores, retrieval mode, and "
            "mode-consistent evidence"
        )
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache_path)
    print(
        f"[llm] Saved scores to {cache_path.name} and traceable evidence "
        f"to {evidence_path.name}"
    )

    return {
        "s_phys": s_phys,
        "s_target_raw": s_target_raw,
        "s_general_plausibility_raw": s_general_plausibility_raw,
        "s_phys_raw": s_phys_raw,
    }


# ═══════════════════════════════════════════════════════════════════════
# Single-Dataset Pipeline
# ═══════════════════════════════════════════════════════════════════════

def _reuse_compatible_llm_cache(
    dataset_name: str,
    destination: Path,
    expected_metadata: dict,
    expected_length: int,
    expected_candidates: list[dict],
) -> bool:
    """Reuse exact static scores when only acquisition-side controls changed."""
    if destination.exists():
        return False
    sources: list[tuple[str, Path]] = []
    if expected_metadata.get("run_mode") == "custom":
        sources.append(
            (
                f"{EXPERIMENT_VERSION}/production",
                OUTPUT_DIR / dataset_name / "_llm_scores.json",
            )
        )
    sources.extend(
        (
            version,
            OUTPUT_DIR.parent / version / dataset_name / "_llm_scores.json",
        )
        for version in COMPATIBLE_LLM_CACHE_EXPERIMENTS
    )
    for source_label, source in sources:
        source_evidence = _llm_evidence_path(source)
        if not source.exists():
            continue
        try:
            cached = json.loads(source.read_text(encoding="utf-8"))
            migrated_metadata = dict(cached.get("_cache_metadata", {}))
            migrated_metadata["experiment_version"] = expected_metadata["experiment_version"]
            migrated_metadata["run_mode"] = expected_metadata["run_mode"]
            score_vectors = _static_score_vectors(cached, expected_length)
            if (
                score_vectors is None
                or migrated_metadata != expected_metadata
                or not _static_evidence_complete(
                    source_evidence,
                    expected_length,
                    expected_candidates,
                    expected_generation_id=cached.get(
                        "_evidence_generation_id"
                    ),
                    expected_scores=cached,
                    expected_composition_unit=expected_metadata.get(
                        "composition_unit"
                    ),
                    expected_objective_direction=expected_metadata.get(
                        "objective_direction"
                    ),
                    expected_retrieval_mode=expected_metadata.get(
                        "llm_retrieval_mode",
                        "full",
                    ),
                )
            ):
                continue
            cached["_cache_metadata"] = expected_metadata
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_evidence, _llm_evidence_path(destination))
            tmp = destination.with_suffix(".tmp")
            tmp.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
            tmp.replace(destination)
            print(
                "[reuse] Migrated exact static LLM scores from "
                f"{source_label}"
            )
            return True
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return False


def _reuse_compatible_feature_run(
    dataset_name: str,
    run_idx: int,
    destination: Path,
) -> bool:
    """Copy prior initial-fit artifacts; the loader revalidates their metadata."""
    if destination.exists():
        return False
    for version in COMPATIBLE_FEATURE_CACHE_EXPERIMENTS:
        source = (
            OUTPUT_DIR.parent / version / dataset_name
            / "_feature_fits" / f"run_{run_idx:03d}"
        )
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
            print(f"[reuse] Copied compatible feature fit for run {run_idx + 1} from {version}")
            return True
    return False


def run_dataset(config: DatasetConfig) -> None:
    if not config.data_path.is_file():
        raise FileNotFoundError(
            f"Dataset {config.name} requires {config.data_path}. "
            "Place the prepared CSV and its columns/dedup sidecars at that path "
            "before running this task. No scoring request has been made."
        )
    output_namespace, run_mode = _output_namespace()
    out_dir = output_namespace / config.name

    implementation_identity = implementation_fingerprint(PROJECT)
    environment_identity = environment_fingerprint()
    out_dir.mkdir(parents=True, exist_ok=True)
    reproducibility_config = {
        "implementation_fingerprint_version": PRODUCTION_PROTOCOL[
            "implementation_fingerprint_version"
        ],
        "environment_fingerprint_version": PRODUCTION_PROTOCOL[
            "environment_fingerprint_version"
        ],
        "implementation_fingerprint": implementation_identity,
        "environment_fingerprint": environment_identity,
        "candidate_index_audit": CANDIDATE_INDEX_AUDIT_VERSION,
    }
    print(
        "[repro] implementation="
        f"{implementation_identity['sha256'][:12]}, environment="
        f"{environment_identity['sha256'][:12]}"
    )

    available_keys = require_ustc_api_keys(API_KEY_SLOT)
    api_keys = [available_keys[API_KEY_SLOT - 1]]
    shared_key_pool = AsyncUSTCKeyPool(
        api_keys,
        base_url=LLM_API_BASE_URL,
        key_aliases=[f"key_slot_{API_KEY_SLOT}"],
        per_key_concurrency=LLM_PER_KEY_CONCURRENCY,
    )

    # 一次读取, 共享给列检测、数据加载和 LLM 评审
    df_raw = read_dataset_table(config.data_path)

    # Four-role classification. Registered sibling targets and explicit
    # auxiliary properties are forced into the blocked response set.
    forced_property_cols = _registered_property_cols(config)
    missing_forced = [c for c in forced_property_cols if c not in df_raw.columns]
    if missing_forced:
        raise KeyError(
            f"Registered response/property columns missing from {config.data_path.name}: "
            f"{missing_forced}"
        )
    column_roles = _detect_columns(
        df_raw,
        config.target_col,
        config.data_path,
        forced_property_cols,
        api_key=api_keys[0],
    )
    elem_cols = column_roles["element_cols"]
    proc_cols = column_roles["process_cols"]
    property_cols = column_roles["target_cols"]
    assert_no_target_leakage(
        [*elem_cols, *proc_cols],
        property_cols,
        context=f"{config.name} runner input roles",
    )
    if config.expected_target_valid_rows is None:
        raise RuntimeError(
            f"{config.name} must register expected_target_valid_rows for the "
            "dataset loading contract"
        )
    data_guard = validate_model_ready_dataset(
        data_path=config.data_path,
        df=df_raw,
        target_col=config.target_col,
        element_cols=elem_cols,
        process_cols=proc_cols,
        expected_target_n=config.expected_target_valid_rows,
        expected_composition_unit=config.composition_unit,
        expected_composition_missing_policy=config.composition_missing_policy,
        require_dedup_manifest=config.require_dedup_manifest,
        require_roles_sidecar=True,
    )
    data_guard_metadata = {
        key: data_guard[key]
        for key in (
            "guard_version",
            "status",
            "processed_sha256",
            "n_unique_rows",
            "target_col",
            "target_valid_rows",
            "composition_unit",
            "composition_missing_policy",
            "composition_missing_cell_count",
            "composition_rows_with_missing",
            "composition_sum_range",
            "composition_sum_min",
            "composition_sum_max",
            "candidate_key_columns",
            "numeric_key_rounding_decimals",
        )
    }
    df_raw, unit_meta = normalize_mixed_composition_units(
        df_raw,
        elem_cols,
        enabled=config.normalize_mixed_units,
        label=config.name,
    )

    print(f"\n{'='*60}")
    print(f"Dataset: {config.name}")
    print(f"  Target: {config.target_col}")
    print(f"  Material: {config.material_system}")
    print(
        f"  Elements: {len(elem_cols)}, Process: {len(proc_cols)}, "
        f"Blocked properties: {property_cols}"
    )
    print(f"  Output: {out_dir}")
    print(
        f"  Data guard: {data_guard['status']} "
        f"(N={data_guard['target_valid_rows']}, "
        f"unique rows={data_guard['n_unique_rows']})"
    )
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    # ── 1) Load data + feature engineering ──
    result_label = f"best_{config.target_col.lower()}"
    target_valid_mask = pd.to_numeric(
        df_raw[config.target_col], errors="coerce"
    ).notna()
    df_llm = df_raw.loc[target_valid_mask].reset_index(drop=True)
    y = pd.to_numeric(df_llm[config.target_col], errors="coerce").to_numpy(dtype=np.float32)
    if len(df_llm) != len(df_raw):
        print(
            f"[setup] LLM scoring uses {len(df_llm)} target-valid rows "
            f"(dropped {len(df_raw) - len(df_llm)} NaN {config.target_col} rows)"
        )

    resources = _resolve_resources(len(y))
    n_init = resources["n_init"]
    batch_size = resources["batch_size"]
    total_budget = resources["total_budget"]
    budget_max_iter = resources["max_iter"]
    elem_available = [c for c in elem_cols if c in df_llm.columns]
    safe_input_cols: list[str] = []
    for col in [*elem_cols, *proc_cols]:
        if col in df_llm.columns and col not in safe_input_cols:
            safe_input_cols.append(col)
    assert_no_target_leakage(
        safe_input_cols,
        property_cols,
        context=f"{config.name} LLM/model visible columns",
    )
    review_df = df_llm[[*safe_input_cols, config.target_col]].copy()
    comp_df_for_llm = review_df[elem_available].copy()
    comp_df_for_llm[config.target_col] = y
    dynamic_input_cols = safe_input_cols.copy()

    global_max = float(y.max())
    global_min = float(y.min())
    global_optimum = objective_optimum(y, config.objective_direction)
    print(f"[setup] Objective = {config.objective_direction}, global optimum = "
          f"{global_optimum:.6g}, raw range = [{global_min:.6g}, {global_max:.6g}], "
          f"n = {len(y)}, "
          f"n_init = {n_init} ({n_init / len(y):.1%}), "
          f"batch = {batch_size} ({batch_size / len(y):.1%}), "
          f"total_budget = {total_budget} ({total_budget / len(y):.1%}), "
          f"feature_representation = {FEATURE_REPRESENTATION}, "
          f"feature_fit_scope = {_feature_fit_scope()}, "
          "strategy_features = matched_shared_pre_acquisition_features, "
          f"acquisition_rounds = {budget_max_iter}, "
          f"final_batch = {resources['final_batch_size']}")
    llm_reliability_config = {
        "llm_retry_policy": LLM_RETRY_POLICY_VERSION,
        "llm_retry_until_success": LLM_RETRY_UNTIL_SUCCESS,
        "llm_strict_scoring": LLM_STRICT_SCORING,
        "llm_max_concurrency": LLM_MAX_CONCURRENCY,
        "llm_api_key_count": shared_key_pool.key_count,
        "llm_per_key_concurrency": LLM_PER_KEY_CONCURRENCY,
        "llm_key_scheduler": LLM_KEY_SCHEDULER_VERSION,
        "llm_key_slot_alias": f"key_slot_{API_KEY_SLOT}",
        "dataset_key_allocation": DATASET_KEY_ALLOCATION_VERSION,
        "dataset_queue_order": DATASET_QUEUE_ORDER_VERSION,
        "dataset_queue_max_workers": DATASET_QUEUE_MAX_WORKERS,
        "llm_rate_limit_batch_size": LLM_RATE_LIMIT_BATCH_SIZE,
        "llm_rate_limit_sleep_seconds": LLM_RATE_LIMIT_SLEEP_SECONDS,
        "llm_retry_base_seconds": LLM_RETRY_BASE_SECONDS,
        "llm_retry_max_seconds": LLM_RETRY_MAX_SECONDS,
        "dynamic_rescoring_failure_policy": (
            PRODUCTION_PROTOCOL["dynamic_rescoring_failure_policy"]
        ),
    }
    print(
        f"[setup] Strict LLM scoring: retry transient failures until success, "
        f"keys={shared_key_pool.key_count}, "
        f"assigned_slot={API_KEY_SLOT}, "
        f"per-key concurrency={LLM_PER_KEY_CONCURRENCY}, "
        f"total concurrency={LLM_MAX_CONCURRENCY}, no static fallback"
    )

    # ── 2) LLM review scores ──
    llm_cache = out_dir / "_llm_scores.json"
    llm_cache_metadata = {
        "llm_scoring_prompt": LLM_SCORING_PROMPT_VERSION,
        "llm_evidence_version": LLM_EVIDENCE_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "run_mode": run_mode,
        "material_system": config.material_system,
        "composition_unit": config.composition_unit,
        "objective_direction": config.objective_direction,
        "objective_description": config.objective_description,
        "composition_missing_policy": config.composition_missing_policy,
        "llm_retrieval_mode": LLM_RETRIEVAL_MODE,
        "column_role_schema_version": COLUMN_ROLE_SCHEMA_VERSION,
        "target_col": config.target_col,
        "target_cols": property_cols,
        "model_ready_data_guard": data_guard_metadata,
        "visible_input_columns": safe_input_cols,
        "visible_input_fingerprint": dataframe_fingerprint(
            review_df, safe_input_cols
        ),
        "source_order_fingerprint": dataframe_fingerprint(
            review_df, [*safe_input_cols, config.target_col]
        ),
        "llm_runtime_fingerprint": _llm_runtime_fingerprint(),
        "knowledge_tie_break": KNOWLEDGE_TIE_BREAK_VERSION,
        "knowledge_tie_salt": KNOWLEDGE_TIE_SALT,
        **llm_reliability_config,
    }
    if unit_meta.get("applied"):
        llm_cache_metadata["composition_unit_normalization"] = unit_meta.get("cache_key")
    expected_llm_candidates = _dataframe_input_records(review_df, safe_input_cols)
    knowledge_tie_break_ranks, knowledge_tie_break_audit = (
        _knowledge_tie_break_ranks(
            expected_llm_candidates,
            dataset_name=config.name,
            target_col=config.target_col,
            composition_unit=config.composition_unit,
            objective_direction=config.objective_direction,
        )
    )
    _reuse_compatible_llm_cache(
        config.name,
        llm_cache,
        llm_cache_metadata,
        len(review_df),
        expected_llm_candidates,
    )
    llm_scores = precompute_llm_scores(
        df=review_df,
        comp_df=comp_df_for_llm,
        target_col=config.target_col,
        element_cols=elem_cols,
        process_cols=proc_cols,
        material_system=config.material_system,
        composition_unit=config.composition_unit,
        objective_direction=config.objective_direction,
        cache_path=llm_cache,
        objective_description=config.objective_description,
        cache_metadata=llm_cache_metadata,
        forbidden_target_cols=property_cols,
        key_pool=shared_key_pool,
    )

    # ── 3) Build strategies ──
    strategies = build_strategies()
    strategy_names = [s.name for s in strategies]
    if len(strategy_names) != len(set(strategy_names)):
        raise RuntimeError(f"Duplicate strategy names: {strategy_names}")
    if tuple(strategy_names) != STRATEGY_NAMES:
        raise RuntimeError(
            f"Strategy protocol drift: {strategy_names} != {list(STRATEGY_NAMES)}"
        )
    print(f"[setup] {len(strategies)} strategies: {strategy_names}")

    from common.llm.reviewer import LLMReviewer
    from calm.dynamic_scorer import DynamicLLMScorer

    reviewer_dyn = LLMReviewer(
        model=LLM_SCORING_MODEL,
        temperature=LLM_SCORING_TEMPERATURE,
        top_k=LLM_RETRIEVAL_TOP_K,
        use_hyde=LLM_USE_HYDE,
        retrieval_mode=LLM_RETRIEVAL_MODE,
        material_system=config.material_system,
        composition_unit=config.composition_unit,
        element_cols=elem_cols,
        process_cols=proc_cols,
        target_col=config.target_col,
        objective_direction=config.objective_direction,
        objective_description=config.objective_description,
        max_concurrency=LLM_MAX_CONCURRENCY,
        rate_limit_batch_size=LLM_RATE_LIMIT_BATCH_SIZE,
        rate_limit_sleep_seconds=LLM_RATE_LIMIT_SLEEP_SECONDS,
        retry_until_success=LLM_RETRY_UNTIL_SUCCESS,
        retry_base_seconds=LLM_RETRY_BASE_SECONDS,
        retry_max_seconds=LLM_RETRY_MAX_SECONDS,
        strict_scoring=LLM_STRICT_SCORING,
        min_api_keys=LLM_API_KEY_COUNT,
        per_key_concurrency=LLM_PER_KEY_CONCURRENCY,
        key_pool=shared_key_pool,
    )
    reviewer_dyn.set_dataset(
        review_df,
        target_col=config.target_col,
        element_cols=elem_cols,
        process_cols=proc_cols,
    )
    dynamic_scorers = {
        strategy_name: DynamicLLMScorer(
            reviewer=reviewer_dyn,
            comp_df=review_df,
            element_cols=elem_cols,
            process_cols=proc_cols,
            target_col=config.target_col,
            objective_direction=config.objective_direction,
            forbidden_target_cols=property_cols,
            top_k=LLM_RETRIEVAL_TOP_K,
            anchor_best_k=ANCHOR_BEST_K,
            anchor_worst_k=ANCHOR_WORST_K,
            anchor_nearby_k=ANCHOR_NEARBY_K,
            candidate_memory_prefilter_k=CANDIDATE_MEMORY_PREFILTER_K,
            audit_path=out_dir / DYNAMIC_EVIDENCE_FILENAMES[strategy_name],
        )
        for strategy_name in DYNAMIC_STRATEGY_NAMES
    }
    for strategy in strategies:
        if getattr(strategy, "supports_dynamic_rescoring", False):
            if strategy.name not in dynamic_scorers:
                raise RuntimeError(
                    f"No isolated dynamic scorer registered for {strategy.name}"
                )
            strategy.configure_dynamic_rescoring(
                dynamic_scorers[strategy.name],
                RESCORING_INTERVAL,
                top_m_frac=DYNAMIC_TOP_M_FRAC,
                top_m_min_mult=DYNAMIC_TOP_M_MIN_MULT,
                top_m_cap=DYNAMIC_TOP_M_CAP,
                random_frac=DYNAMIC_RANDOM_FRAC,
            )
    if RESCORING_INTERVAL > 0:
        print(
            f"[setup] Dynamic LLM rescoring enabled for "
            f"{list(DYNAMIC_STRATEGY_NAMES)} "
            f"(interval={RESCORING_INTERVAL})"
        )
    else:
        print("[setup] Dynamic LLM rescoring is disabled")

    # ── 5) AL benchmark ──
    master_rng = np.random.RandomState(SEED)
    paired_run_seeds = [
        int(master_rng.randint(0, 2**31 - 1)) for _ in range(N_RUNS)
    ]
    knowledge_score_control_audit: list[dict] = []
    checkpoint_path = out_dir / "_checkpoint.json"
    ckpt_config = {
        "experiment_version": EXPERIMENT_VERSION,
        "run_mode": run_mode,
        "n_runs": N_RUNS,
        "candidate_pool_fraction": 1.0,
        "initial_fraction": INITIAL_LABELED_FRACTION,
        "batch_fraction": BATCH_QUERY_FRACTION,
        "total_budget_fraction": TOTAL_LABELED_BUDGET_FRACTION,
        "resource_allocation_policy": RESOURCE_ALLOCATION_POLICY_VERSION,
        "objective_direction_contract": OBJECTIVE_DIRECTION_CONTRACT_VERSION,
        "objective_direction": config.objective_direction,
        "objective_description": config.objective_description,
        "regret_auc_version": REGRET_AUC_VERSION,
        "n_samples": len(y),
        "n_init": n_init,
        "n_init_mode": "ceil_fraction",
        "feature_fit_scope": _feature_fit_scope(),
        "feature_fit_version": INITIAL_FEATURE_FIT_VERSION,
        "feature_representation": FEATURE_REPRESENTATION,
        "process_feature_encoding": PROCESS_FEATURE_ENCODING_VERSION,
        "strategy_feature_scope": "matched_shared_pre_acquisition_features",
        "strategy_set": STRATEGY_SET_VERSION,
        "strategy_names": strategy_names,
        "batch_size": batch_size,
        "batch_size_mode": "round_half_up_fraction",
        "total_budget": total_budget,
        "total_budget_mode": "ceil_fraction_including_initial",
        "max_iter": budget_max_iter,
        "final_batch_size": resources["final_batch_size"],
        "stopping_policy": STOPPING_POLICY_VERSION,
        "iteration_checkpoint": ITERATION_CHECKPOINT_VERSION,
        **reproducibility_config,
        "seed": SEED, "dataset": config.name,
        "target_col": config.target_col,
        "target_cols": property_cols,
        "model_ready_data_guard": data_guard_metadata,
        "composition_unit": config.composition_unit,
        "composition_missing_policy": config.composition_missing_policy,
        "column_role_schema_version": COLUMN_ROLE_SCHEMA_VERSION,
        "column_roles": column_roles,
        "visible_input_columns": safe_input_cols,
        "source_order_fingerprint": llm_cache_metadata["source_order_fingerprint"],
        "ucb_alpha": UCB_ALPHA,
        "lam_llm": LAM_LLM,
        "surrogate_policy": SURROGATE_POLICY_VERSION,
        "nn_surrogate": PRODUCTION_PROTOCOL["nn_surrogate"],
        "surrogate_sigma_calibration": SURROGATE_SIGMA_CALIBRATION,
        "ensemble_min_train_samples": ENSEMBLE_MIN_TRAIN_SAMPLES,
        "ensemble_weight_update_samples": ENSEMBLE_WEIGHT_UPDATE_SAMPLES,
        "ensemble_weight_floor": ENSEMBLE_WEIGHT_FLOOR,
        "ensemble_weight_temperature": ENSEMBLE_WEIGHT_TEMPERATURE,
        "ensemble_top_weight_strength": ENSEMBLE_TOP_WEIGHT_STRENGTH,
        "gbdt_bootstrap_models": GBDT_BOOTSTRAP_MODELS,
        "llm_score_mode": "objective_utility_static_s_phys_plus_tie_rank_v6",
        "knowledge_score_control": KNOWLEDGE_SCORE_CONTROL,
        "knowledge_score_control_version": KNOWLEDGE_SCORE_CONTROL_VERSION,
        "knowledge_tie_break": KNOWLEDGE_TIE_BREAK_VERSION,
        "knowledge_tie_salt": KNOWLEDGE_TIE_SALT,
        "knowledge_tie_break_audit": knowledge_tie_break_audit,
        "llm_scoring_prompt": LLM_SCORING_PROMPT_VERSION,
        "llm_evidence_version": LLM_EVIDENCE_VERSION,
        "llm_scoring_model": LLM_SCORING_MODEL,
        "llm_api_base_url": LLM_API_BASE_URL,
        "llm_scoring_temperature": LLM_SCORING_TEMPERATURE,
        "llm_retrieval_top_k": LLM_RETRIEVAL_TOP_K,
        "llm_use_hyde": LLM_USE_HYDE,
        "llm_retrieval_mode": LLM_RETRIEVAL_MODE,
        "llm_retrieval_policy": LLM_RETRIEVAL_POLICY_VERSION,
        "llm_runtime_fingerprint": llm_cache_metadata["llm_runtime_fingerprint"],
        "lambda_mode": PRODUCTION_PROTOCOL["lambda_mode"],
        "ewrc_gamma": EWRC_GAMMA,
        "composition_unit_normalization": (
            unit_meta.get("cache_key") if unit_meta.get("applied") else "disabled"
        ),
        "dynamic_rescoring": (
            "disabled_for_static_control"
            if RESCORING_INTERVAL == 0
            else (
                "llmdynamic_and_hybrid_independent_per_round_shortlist_"
                "static_rank_slots_v3"
            )
        ),
        "dynamic_rescoring_input_mode": "element_cols_plus_process_cols",
        "dynamic_rescoring_input_columns": dynamic_input_cols,
        "dynamic_rescoring_cache": DYNAMIC_RESCORING_CACHE_VERSION,
        "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
        "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
        "candidate_memory_prefilter_k": CANDIDATE_MEMORY_PREFILTER_K,
        "dynamic_prequery_ledger": DYNAMIC_PREQUERY_LEDGER_VERSION,
        "anchor_best_k": ANCHOR_BEST_K,
        "anchor_worst_k": ANCHOR_WORST_K,
        "anchor_nearby_k": ANCHOR_NEARBY_K,
        "rescoring_interval": RESCORING_INTERVAL,
        "dynamic_top_m_frac": DYNAMIC_TOP_M_FRAC,
        "dynamic_top_m_min_mult": DYNAMIC_TOP_M_MIN_MULT,
        "dynamic_top_m_cap": DYNAMIC_TOP_M_CAP,
        "dynamic_random_frac": DYNAMIC_RANDOM_FRAC,
        "dynamic_rank_fusion": PRODUCTION_PROTOCOL["dynamic_rank_fusion"],
        "llm_dynamic_shortlist_policy": LLM_DYNAMIC_SHORTLIST_POLICY_VERSION,
        "dynamic_strategy_names": list(DYNAMIC_STRATEGY_NAMES),
        "dynamic_evidence_files": dict(DYNAMIC_EVIDENCE_FILENAMES),
        "strategy_seed_mode": "stable_offsets_v1",
        "matched_pair_seed": [
            "Adaptive-UCB",
            "LLM-dynamic",
            "Hybrid-static",
            "Hybrid",
        ],
        **llm_reliability_config,
    }

    all_records: dict[str, list[list[dict]]]
    in_progress: dict | None = None
    resume_offset: dict[str, int] = {}
    checkpoint_candidates_exist = checkpoint_path.exists() or any(
        checkpoint_path.parent.glob(f"{checkpoint_path.stem}*.tmp")
    )
    if checkpoint_candidates_exist:
        try:
            ckpt, checkpoint_source = _load_latest_checkpoint(
                checkpoint_path,
                ckpt_config,
            )
            all_records = ckpt["records"]
            in_progress = ckpt.get("in_progress")
            if list(all_records) != strategy_names:
                raise ValueError(
                    f"strategy keys/order mismatch: {list(all_records)} != {strategy_names}"
                )
            if any(not isinstance(runs, list) or len(runs) > N_RUNS
                   for runs in all_records.values()):
                raise ValueError("invalid completed-run counts")
            for sname, runs in all_records.items():
                resume_offset[sname] = len(runs)
            done_runs = min(resume_offset.values()) if resume_offset else 0
            print(f"[resume] Loaded checkpoint: {done_runs} runs fully done, "
                  + ", ".join(f"{k}={v}" for k, v in resume_offset.items()))
            if checkpoint_source != checkpoint_path:
                print(
                    f"[resume] Recovered newer state from {checkpoint_source.name} "
                    f"after an interrupted Windows file replacement"
                )
            if in_progress is not None:
                print(
                    f"[resume] Iteration checkpoint: run={in_progress['run_idx'] + 1}, "
                    f"strategy={in_progress['strategy']}, "
                    f"iteration={in_progress['state']['records'][-1]['iteration']}"
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise RuntimeError(
                f"Cannot resume incompatible checkpoint {checkpoint_path}: {e}. "
                "The checkpoint has not been replaced; use matching settings/source "
                "or a separate output location for a new run."
            ) from e
    else:
        all_records = {s.name: [] for s in strategies}

    def _save_checkpoint(active: dict | None = None):
        ckpt_data = {
            "_config": ckpt_config,
            "records": all_records,
            "in_progress": active,
        }
        _atomic_write_checkpoint(checkpoint_path, ckpt_data)

    n_al_loops = len(strategies) * N_RUNS
    total_done = sum(len(v) for v in all_records.values())
    print(f"[benchmark] {n_al_loops} AL loops ({len(strategies)} strategies × {N_RUNS} runs), "
          f"{total_done} already done")

    for run_idx in range(N_RUNS):
        run_seed = paired_run_seeds[run_idx]
        rng = np.random.RandomState(run_seed)
        initial_idx = rng.choice(len(y), size=n_init, replace=False)
        run_pool_scores, run_control_audit = _controlled_pool_scores(
            llm_scores,
            dataset_name=config.name,
            run_seed=run_seed,
        )
        run_pool_scores["_tie_break_rank"] = knowledge_tie_break_ranks.copy()
        knowledge_score_control_audit.append(
            {"run_index": int(run_idx), **run_control_audit}
        )

        pending_strategies = [
            s for s in strategies
            if run_idx >= resume_offset.get(s.name, 0)
        ]
        if not pending_strategies:
            continue

        y_run = y
        run_feature_dir = out_dir / "_feature_fits" / f"run_{run_idx:03d}"
        _reuse_compatible_feature_run(config.name, run_idx, run_feature_dir)
        X_run, y_run, _ = load_external_dataset_initial_fit(
            xlsx_path=config.data_path,
            target_col=config.target_col,
            initial_idx=initial_idx,
            element_cols=elem_cols,
            process_cols=proc_cols,
            target_cols=property_cols,
            output_dir=run_feature_dir,
            force_rerun_selection=False,
            normalize_mixed_units=config.normalize_mixed_units,
            material_system=config.material_system,
            composition_unit=config.composition_unit,
            composition_missing_policy=config.composition_missing_policy,
            feature_representation=FEATURE_REPRESENTATION,
        )
        if len(y_run) != len(y) or not np.allclose(y_run, y, equal_nan=True):
            raise RuntimeError(
                f"Initial-fit feature engineering returned a target vector mismatch "
                f"for run {run_idx}"
            )
        assert_no_target_leakage(
            list(X_run.columns),
            property_cols,
            context=f"{config.name} run {run_idx} feature matrix",
        )

        for strategy in strategies:
            sname = strategy.name
            if run_idx < resume_offset.get(sname, 0):
                continue

            strategy_pool_scores = run_pool_scores
            initial_strategy_state = None
            initial_record_metadata = None

            if sname in (
                "LLM-dynamic",
                "Hybrid",
                "Hybrid-static",
                "Adaptive-UCB",
                "NN-UCB",
            ):
                print(f"    [{sname}] run {run_idx+1}/{N_RUNS}...", end=" ", flush=True)

            resume_loop_state = None
            if (
                in_progress is not None
                and int(in_progress.get("run_idx", -1)) == run_idx
                and in_progress.get("strategy") == sname
            ):
                resume_loop_state = in_progress["state"]

            def _save_iteration(state: dict, *, _run=run_idx, _strategy=sname) -> None:
                nonlocal in_progress
                in_progress = {
                    "run_idx": int(_run),
                    "strategy": _strategy,
                    "state": state,
                }
                _save_checkpoint(in_progress)

            records = run_single_al_loop(
                strategy=strategy,
                X=X_run,
                y=y_run,
                pool_scores=strategy_pool_scores,
                initial_idx=initial_idx.copy(),
                batch_size=batch_size, total_budget=total_budget,
                rng=np.random.RandomState(_strategy_seed(run_seed, sname)),
                result_label=result_label,
                objective_direction=config.objective_direction,
                resume_state=resume_loop_state,
                initial_strategy_state=initial_strategy_state,
                initial_record_metadata=initial_record_metadata,
                checkpoint_callback=_save_iteration,
            )
            all_records[sname].append(records)
            in_progress = None
            _save_checkpoint(None)

            if sname in (
                "LLM-dynamic",
                "Hybrid",
                "Hybrid-static",
                "Adaptive-UCB",
                "NN-UCB",
            ):
                print("done", flush=True)

        if (run_idx + 1) % 5 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{run_idx + 1}/{N_RUNS}] runs done, {elapsed:.0f}s elapsed", flush=True)

    # ── 6) Save results ──
    conv_output = {
        "config": {
            "experiment_version": EXPERIMENT_VERSION,
            "run_mode": run_mode,
            "dataset": config.name,
            "target_col": config.target_col,
            "material_system": config.material_system,
            "n_runs": N_RUNS,
            "candidate_pool_fraction": 1.0,
            "initial_fraction": INITIAL_LABELED_FRACTION,
            "batch_fraction": BATCH_QUERY_FRACTION,
            "total_budget_fraction": TOTAL_LABELED_BUDGET_FRACTION,
            "resource_allocation_policy": RESOURCE_ALLOCATION_POLICY_VERSION,
            "objective_direction_contract": OBJECTIVE_DIRECTION_CONTRACT_VERSION,
            "objective_direction": config.objective_direction,
            "objective_description": config.objective_description,
            "regret_auc_version": REGRET_AUC_VERSION,
            "n_init": n_init,
            "n_init_mode": "ceil_fraction",
            "feature_fit_scope": _feature_fit_scope(),
            "feature_fit_version": INITIAL_FEATURE_FIT_VERSION,
            "feature_representation": FEATURE_REPRESENTATION,
            "process_feature_encoding": PROCESS_FEATURE_ENCODING_VERSION,
            "strategy_feature_scope": "matched_shared_pre_acquisition_features",
            "strategy_set": STRATEGY_SET_VERSION,
            "strategy_names": strategy_names,
            "batch_size": batch_size,
            "batch_size_mode": "round_half_up_fraction",
            "total_budget": total_budget,
            "total_budget_mode": "ceil_fraction_including_initial",
            "max_iter": budget_max_iter,
            "final_batch_size": resources["final_batch_size"],
            "stopping_policy": STOPPING_POLICY_VERSION,
            "iteration_checkpoint": ITERATION_CHECKPOINT_VERSION,
            **reproducibility_config,
            "seed": SEED,
            "target_cols": property_cols,
            "model_ready_data_guard": data_guard_metadata,
            "composition_unit": config.composition_unit,
            "composition_missing_policy": config.composition_missing_policy,
            "column_role_schema_version": COLUMN_ROLE_SCHEMA_VERSION,
            "column_roles": column_roles,
            "visible_input_columns": safe_input_cols,
            "source_order_fingerprint": llm_cache_metadata["source_order_fingerprint"],
            "ucb_alpha": UCB_ALPHA,
            "lam_llm": LAM_LLM,
            "surrogate_policy": SURROGATE_POLICY_VERSION,
            "nn_surrogate": PRODUCTION_PROTOCOL["nn_surrogate"],
            "surrogate_sigma_calibration": SURROGATE_SIGMA_CALIBRATION,
            "ensemble_min_train_samples": ENSEMBLE_MIN_TRAIN_SAMPLES,
            "ensemble_weight_update_samples": ENSEMBLE_WEIGHT_UPDATE_SAMPLES,
            "ensemble_weight_floor": ENSEMBLE_WEIGHT_FLOOR,
            "ensemble_weight_temperature": ENSEMBLE_WEIGHT_TEMPERATURE,
            "ensemble_top_weight_strength": ENSEMBLE_TOP_WEIGHT_STRENGTH,
            "gbdt_bootstrap_models": GBDT_BOOTSTRAP_MODELS,
            "llm_score_mode": "objective_utility_static_s_phys_plus_tie_rank_v6",
            "knowledge_score_control": KNOWLEDGE_SCORE_CONTROL,
            "knowledge_score_control_version": KNOWLEDGE_SCORE_CONTROL_VERSION,
            "knowledge_score_control_audit": knowledge_score_control_audit,
            "knowledge_tie_break": KNOWLEDGE_TIE_BREAK_VERSION,
            "knowledge_tie_salt": KNOWLEDGE_TIE_SALT,
            "knowledge_tie_break_audit": knowledge_tie_break_audit,
            "paired_run_seeds_sha256": hashlib.sha256(
                np.asarray(paired_run_seeds, dtype=np.int64).tobytes(order="C")
            ).hexdigest(),
            "llm_scoring_prompt": LLM_SCORING_PROMPT_VERSION,
            "llm_evidence_version": LLM_EVIDENCE_VERSION,
            "llm_scoring_model": LLM_SCORING_MODEL,
            "llm_api_base_url": LLM_API_BASE_URL,
            "llm_scoring_temperature": LLM_SCORING_TEMPERATURE,
            "llm_retrieval_top_k": LLM_RETRIEVAL_TOP_K,
            "llm_use_hyde": LLM_USE_HYDE,
            "llm_retrieval_mode": LLM_RETRIEVAL_MODE,
            "llm_retrieval_policy": LLM_RETRIEVAL_POLICY_VERSION,
            "llm_runtime_fingerprint": llm_cache_metadata["llm_runtime_fingerprint"],
            "lambda_mode": PRODUCTION_PROTOCOL["lambda_mode"],
            "ewrc_gamma": EWRC_GAMMA,
            "composition_unit_normalization": (
                unit_meta.get("cache_key") if unit_meta.get("applied") else "disabled"
            ),
            "dynamic_rescoring": (
                "disabled_for_static_control"
                if RESCORING_INTERVAL == 0
                else (
                    "llmdynamic_and_hybrid_independent_per_round_shortlist_"
                    "static_rank_slots_v3"
                )
            ),
            "dynamic_rescoring_input_mode": "element_cols_plus_process_cols",
            "dynamic_rescoring_input_columns": dynamic_input_cols,
            "dynamic_rescoring_cache": DYNAMIC_RESCORING_CACHE_VERSION,
            "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
            "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
            "candidate_memory_prefilter_k": CANDIDATE_MEMORY_PREFILTER_K,
            "dynamic_prequery_ledger": DYNAMIC_PREQUERY_LEDGER_VERSION,
            "anchor_best_k": ANCHOR_BEST_K,
            "anchor_worst_k": ANCHOR_WORST_K,
            "anchor_nearby_k": ANCHOR_NEARBY_K,
            "rescoring_interval": RESCORING_INTERVAL,
            "dynamic_top_m_frac": DYNAMIC_TOP_M_FRAC,
            "dynamic_top_m_min_mult": DYNAMIC_TOP_M_MIN_MULT,
            "dynamic_top_m_cap": DYNAMIC_TOP_M_CAP,
            "dynamic_random_frac": DYNAMIC_RANDOM_FRAC,
            "dynamic_rank_fusion": PRODUCTION_PROTOCOL["dynamic_rank_fusion"],
            "llm_dynamic_shortlist_policy": (
                LLM_DYNAMIC_SHORTLIST_POLICY_VERSION
            ),
            "dynamic_strategy_names": list(DYNAMIC_STRATEGY_NAMES),
            "dynamic_evidence_files": dict(DYNAMIC_EVIDENCE_FILENAMES),
            "strategy_seed_mode": "stable_offsets_v1",
            "matched_pair_seed": [
                "Adaptive-UCB",
                "LLM-dynamic",
                "Hybrid-static",
                "Hybrid",
            ],
            **llm_reliability_config,
            "global_max": float(global_max),
            "global_min": float(global_min),
            "n_samples": len(y),
        },
        "strategies": {},
    }
    for name, runs in all_records.items():
        conv_output["strategies"][name] = runs

    conv_path = out_dir / "convergence.json"
    conv_path.write_text(json.dumps(conv_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {conv_path}")

    stats = compute_stats(
        all_records,
        global_max,
        target_label=result_label,
        global_min=global_min,
        n_samples=len(y),
        n_init=n_init,
        total_budget=total_budget,
        objective_direction=config.objective_direction,
        primary_strategy=PRIMARY_INFERENCE_STRATEGY,
        primary_comparators=PRIMARY_INFERENCE_COMPARATORS,
        primary_inference_version=PRIMARY_INFERENCE_VERSION,
        exploratory_inference_version=EXPLORATORY_INFERENCE_VERSION,
    )
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {stats_path}")

    plot_convergence(all_records, global_max, conv_output["config"],
                     out_dir / "convergence_plot.png",
                     target_label=config.target_label, target_unit=config.target_unit)
    print(f"[save] {out_dir / 'convergence_plot.png'}")

    removed_checkpoint = False
    for stale in [
        checkpoint_path,
        *checkpoint_path.parent.glob(f"{checkpoint_path.stem}*.tmp"),
    ]:
        if not stale.exists():
            continue
        try:
            stale.unlink()
            removed_checkpoint = True
        except PermissionError as exc:
            print(
                f"[cleanup] WARNING: could not remove locked checkpoint "
                f"{stale.name}; result files are already complete ({exc})"
            )
    if removed_checkpoint:
        print("[cleanup] Removed checkpoint files")

    elapsed = time.perf_counter() - t0
    print(f"\n[dataset] {config.name} completed in {elapsed:.0f}s")

    # Print ranking
    ranking = sorted(stats["strategies"].items(),
                     key=lambda kv: kv[1].get("regret_auc_mean", float("inf")))
    for i, (sname, s) in enumerate(ranking):
        print(f"  {i+1}. {sname}: regret AUC={s['regret_auc_mean']:.1f}")

    print("[done]")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CALM: calibrated knowledge-data active learning for metallic materials"
    )
    p.add_argument("--dataset", type=str, default="2xxx_T6__UTS",
                   help=f"Dataset name. Available: {list(DATASETS.keys())}")
    p.add_argument("--n-runs", type=int, default=N_RUNS)
    p.add_argument(
        "--initial-fraction", type=float, default=INITIAL_LABELED_FRACTION
    )
    p.add_argument(
        "--batch-fraction", type=float, default=BATCH_QUERY_FRACTION
    )
    p.add_argument(
        "--total-budget-fraction",
        type=float,
        default=TOTAL_LABELED_BUDGET_FRACTION,
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--api-key-slot",
        type=int,
        choices=range(1, DATASET_QUEUE_MAX_WORKERS + 1),
        default=1,
        help=(
            "One-based slot in CALM_API_KEY_1..4. "
            "Each dataset runner uses one key."
        ),
    )
    p.add_argument(
        "--feature-representation",
        choices=("raw", "physics", "physics_llm"),
        default=FEATURE_REPRESENTATION,
        help=(
            "raw=composition+process; physics=raw plus corrected deterministic "
            "atomic descriptors; physics_llm=physics plus optional audited "
            "LLM algebraic candidates"
        ),
    )
    p.add_argument("--rescoring-interval", type=int, default=RESCORING_INTERVAL)
    p.add_argument(
        "--ewrc-gamma",
        type=float,
        default=EWRC_GAMMA,
        help="Extreme-rank exponent; 0 reproduces global Spearman, 2 is EWRC.",
    )
    p.add_argument(
        "--knowledge-score-control",
        choices=("observed", "shuffled"),
        default=KNOWLEDGE_SCORE_CONTROL,
        help=(
            "observed=use candidate-matched LLM scores; shuffled=apply a "
            "deterministic paired-run permutation. Shuffled requires "
            "--rescoring-interval 0."
        ),
    )
    return p.parse_args()


def main():
    global N_RUNS, INITIAL_LABELED_FRACTION, BATCH_QUERY_FRACTION
    global TOTAL_LABELED_BUDGET_FRACTION, SEED, RESCORING_INTERVAL, EWRC_GAMMA
    global FEATURE_REPRESENTATION, KNOWLEDGE_SCORE_CONTROL, API_KEY_SLOT

    args = parse_args()

    positive_args = {"n_runs": args.n_runs}
    invalid = {name: value for name, value in positive_args.items() if value <= 0}
    fraction_invalid = not (
        0.0 < args.initial_fraction < args.total_budget_fraction <= 1.0
        and 0.0 < args.batch_fraction <= 1.0
    )
    ewrc_invalid = not math.isfinite(args.ewrc_gamma) or args.ewrc_gamma < 0.0
    if (
        invalid
        or fraction_invalid
        or args.rescoring_interval < 0
        or ewrc_invalid
        or not 1 <= args.api_key_slot <= DATASET_QUEUE_MAX_WORKERS
    ):
        raise ValueError(
            "Require positive run count, non-negative rescoring interval and "
            "EWRC gamma, a key slot in "
            f"1..{DATASET_QUEUE_MAX_WORKERS}, and "
            "0 < initial_fraction < total_budget_fraction <= 1 with "
            f"0 < batch_fraction <= 1; got {vars(args)}"
        )
    if args.knowledge_score_control == "shuffled" and args.rescoring_interval != 0:
        raise ValueError(
            "The shuffled knowledge-score control must use "
            "--rescoring-interval 0 so dynamic LLM feedback cannot restore the "
            "candidate-score association."
        )
    N_RUNS = args.n_runs
    INITIAL_LABELED_FRACTION = args.initial_fraction
    BATCH_QUERY_FRACTION = args.batch_fraction
    TOTAL_LABELED_BUDGET_FRACTION = args.total_budget_fraction
    SEED = args.seed
    FEATURE_REPRESENTATION = args.feature_representation
    RESCORING_INTERVAL = args.rescoring_interval
    EWRC_GAMMA = args.ewrc_gamma
    KNOWLEDGE_SCORE_CONTROL = args.knowledge_score_control
    API_KEY_SLOT = int(args.api_key_slot)

    if args.dataset not in DATASETS:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(DATASETS.keys())}")
        sys.exit(1)

    config = DATASETS[args.dataset]
    run_dataset(config)


if __name__ == "__main__":
    main()
