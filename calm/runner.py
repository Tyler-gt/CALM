"""CALM and its acquisition baselines on paired, fixed-budget alloy searches."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
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
    load_column_roles,
    assert_no_target_leakage,
    dataframe_fingerprint,
    load_external_dataset_initial_fit,
    read_dataset_table,
    validate_model_ready_dataset,
)
from calm.experiment import (
    ANCHOR_BEST_K,
    ANCHOR_NEARBY_K,
    ANCHOR_WORST_K,
    CANDIDATE_MEMORY_POLICY_VERSION,
    CANDIDATE_MEMORY_PREFILTER_K,
    DYNAMIC_FEEDBACK_POLICY_VERSION,
    DYNAMIC_EVIDENCE_FILENAMES,
    DYNAMIC_PREQUERY_LEDGER_VERSION,
    DYNAMIC_RESCORING_CACHE_VERSION,
    DYNAMIC_STRATEGY_NAMES,
    DATASET_KEY_ALLOCATION_VERSION,
    EXPERIMENT_VERSION,
    ITERATION_CHECKPOINT_VERSION,
    KNOWLEDGE_TIE_BREAK_VERSION,
    KNOWLEDGE_TIE_SALT,
    LLM_EVIDENCE_VERSION,
    LLM_KEY_SCHEDULER_VERSION,
    LLM_DYNAMIC_SHORTLIST_POLICY_VERSION,
    LLM_RETRIEVAL_MODE,
    LLM_RETRY_POLICY_VERSION,
    LLM_API_BASE_URL,
    LLM_SCORING_MODEL,
    LLM_SCORING_PROMPT_VERSION,
    LLM_SCORING_TEMPERATURE,
    OUTPUT_DIR,
    DEFAULT_PROTOCOL,
    REGRET_AUC_VERSION,
    RESOURCE_ALLOCATION_POLICY_VERSION,
    STRATEGY_NAMES,
    STRATEGY_SET_VERSION,
    STOPPING_POLICY_VERSION,
    SURROGATE_POLICY_VERSION,
    resolve_resource_allocation,
)
from calm.benchmark import PAPER_RUNS, paired_seeds
from calm.file_lock import exclusive_file_lock
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
from calm.strategies.calm import CALMAcquisition
from calm.strategies.calm_static import CALMStatic
from common.llm.key_pool import AsyncAPIKeyPool, require_api_keys

EXTERNAL = PROJECT / "dabase"
CURATED = EXTERNAL / "curated"
N_RUNS = int(DEFAULT_PROTOCOL["n_runs"])
INITIAL_LABELED_FRACTION = float(DEFAULT_PROTOCOL["initial_fraction"])
BATCH_QUERY_FRACTION = float(DEFAULT_PROTOCOL["batch_fraction"])
TOTAL_LABELED_BUDGET_FRACTION = float(
    DEFAULT_PROTOCOL["total_budget_fraction"]
)
SEED: int | None = None
EXPERIMENT_MODE = "benchmark"
FIXED_LAMBDA: float | None = None
FEATURE_REPRESENTATION = str(DEFAULT_PROTOCOL["feature_representation"])
LAM_LLM = float(DEFAULT_PROTOCOL["lam_llm"])
EWRC_GAMMA = float(DEFAULT_PROTOCOL["ewrc_gamma"])
UCB_ALPHA = float(DEFAULT_PROTOCOL["ucb_alpha"])
ENSEMBLE_MIN_TRAIN_SAMPLES = int(DEFAULT_PROTOCOL["ensemble_min_train_samples"])
ENSEMBLE_WEIGHT_UPDATE_SAMPLES = int(
    DEFAULT_PROTOCOL["ensemble_weight_update_samples"]
)
ENSEMBLE_WEIGHT_FLOOR = float(DEFAULT_PROTOCOL["ensemble_weight_floor"])
ENSEMBLE_WEIGHT_TEMPERATURE = float(
    DEFAULT_PROTOCOL["ensemble_weight_temperature"]
)
ENSEMBLE_TOP_WEIGHT_STRENGTH = float(
    DEFAULT_PROTOCOL["ensemble_top_weight_strength"]
)
GBDT_BOOTSTRAP_MODELS = int(DEFAULT_PROTOCOL["gbdt_bootstrap_models"])
SURROGATE_SIGMA_CALIBRATION = str(
    DEFAULT_PROTOCOL["surrogate_sigma_calibration"]
)
RESCORING_INTERVAL = int(DEFAULT_PROTOCOL["rescoring_interval"])
DYNAMIC_TOP_M_FRAC = float(DEFAULT_PROTOCOL["dynamic_top_m_frac"])
DYNAMIC_TOP_M_MIN_MULT = int(DEFAULT_PROTOCOL["dynamic_top_m_min_mult"])
DYNAMIC_TOP_M_CAP = int(DEFAULT_PROTOCOL["dynamic_top_m_cap"])
DYNAMIC_RANDOM_FRAC = float(DEFAULT_PROTOCOL["dynamic_random_frac"])
LLM_MAX_CONCURRENCY = int(DEFAULT_PROTOCOL["llm_max_concurrency"])
LLM_API_KEY_COUNT = int(DEFAULT_PROTOCOL["llm_api_key_count"])
LLM_PER_KEY_CONCURRENCY = int(DEFAULT_PROTOCOL["llm_per_key_concurrency"])
LLM_RATE_LIMIT_BATCH_SIZE = int(DEFAULT_PROTOCOL["llm_rate_limit_batch_size"])
LLM_RATE_LIMIT_SLEEP_SECONDS = float(
    DEFAULT_PROTOCOL["llm_rate_limit_sleep_seconds"]
)
LLM_RETRY_BASE_SECONDS = float(DEFAULT_PROTOCOL["llm_retry_base_seconds"])
LLM_RETRY_MAX_SECONDS = float(DEFAULT_PROTOCOL["llm_retry_max_seconds"])
API_KEY_SLOT = 1
OUTPUT_ROOT: Path | None = None
STRATEGY_SEED_OFFSETS = {
    "Random": 101,
    "GPR-UCB": 211,
    "RFR-UCB": 307,
    "GBDT-UCB": 401,
    "NN-UCB": 503,
    # Matched common-random-number pair with CALM.
    "Adaptive-UCB": 701,
    "LLM-rank": 601,
    "LLM-dynamic": 701,
    "CALM-static": 701,
    "CALM": 701,
}


# ═══════════════════════════════════════════════════════════════════════
# Dataset Registry
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DatasetConfig:
    """Dataset specification with element and process roles from audited sidecars."""
    name: str
    data_path: Path
    target_col: str
    composition_unit: str
    objective_direction: str
    objective_description: str = ""
    material_system: str = ""          # Optional material-domain context for the LLM
    target_label: str = ""             # Optional plot y-axis label; defaults to target_col
    target_unit: str = "Value"         # Optional unit for plots
    composition_missing_policy: str = COMPLETE_COMPOSITION_POLICY
    # Measured auxiliary properties that must never be model/LLM inputs.
    additional_property_cols: tuple[str, ...] = ()
    expected_target_valid_rows: int | None = None

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

def _detect_columns(
    df: pd.DataFrame,
    target_col: str,
    data_path: Path,
    known_target_cols: list[str],
) -> dict[str, list[str]]:
    """Read audited column roles and exclude all registered properties."""
    return load_column_roles(
        df,
        target_col,
        data_path=data_path,
        known_target_cols=known_target_cols,
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
    'Cu_alloys__EC': DatasetConfig(
        name="Cu_alloys__EC",
        data_path=CURATED / "Cu_alloys__EC.csv",
        target_col="Electrical_conductivity_pctIACS",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="Conductivity", target_unit="%IACS",
        material_system="Cu-alloy",
        expected_target_valid_rows=1486,
    ),
    'HEA_Invar__TEC': DatasetConfig(
        name="HEA_Invar__TEC",
        data_path=EXTERNAL / "HEA_Invar__TEC.csv",
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
    'MPEA__UTS': DatasetConfig(
        name="MPEA__UTS",
        data_path=EXTERNAL / "MPEA__Mechanical.csv",
        target_col="UTS_MPa",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Multi-principal-element alloy",
        additional_property_cols=MPEA_PROPERTIES,
        expected_target_valid_rows=362,
    ),
    'mg_alloy__UTS': DatasetConfig(
        name="mg_alloy__UTS",
        data_path=CURATED / "mg_alloy__UTS.csv",
        target_col="UTS_MPa",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Mg-alloy",
        expected_target_valid_rows=289,
    ),
    'Ti_alloy__YS': DatasetConfig(
        name="Ti_alloy__YS",
        data_path=CURATED / "Ti_alloy__YS.csv",
        target_col="YS_MPa",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="YS", target_unit="MPa",
        material_system="Ti-alloy",
        additional_property_cols=("moe",),
        expected_target_valid_rows=248,
    ),
    'Ti_alloy__UTS': DatasetConfig(
        name="Ti_alloy__UTS",
        data_path=CURATED / "Ti_alloy__UTS.csv",
        target_col="UTS_MPa",
        composition_unit="at.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Ti-alloy",
        additional_property_cols=("moe",),
        expected_target_valid_rows=185,
    ),
    'mg_alloy__YS': DatasetConfig(
        name="mg_alloy__YS",
        data_path=CURATED / "mg_alloy__YS.csv",
        target_col="YS_MPa",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="YS", target_unit="MPa",
        material_system="Mg-alloy",
        expected_target_valid_rows=298,
    ),
    '2xxx_T6__EL': DatasetConfig(
        name="2xxx_T6__EL",
        data_path=CURATED / "al_alloys.csv",
        target_col="EL",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="EL", target_unit="%",
        material_system="Al-alloy",
        expected_target_valid_rows=134,
    ),
    '2xxx_T6__UTS': DatasetConfig(
        name="2xxx_T6__UTS",
        data_path=CURATED / "al_alloys.csv",
        target_col="UTS",
        composition_unit="wt.%",
        objective_direction="maximize",
        target_label="UTS", target_unit="MPa",
        material_system="Al-alloy",
        expected_target_valid_rows=138,
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
    calm = CALMAcquisition(
        alpha=UCB_ALPHA, lam_llm=LAM_LLM,
        extreme_rank_power=EWRC_GAMMA, fixed_lambda=FIXED_LAMBDA,
        **ensemble_kwargs,
    )
    if EXPERIMENT_MODE not in {"benchmark", "supplementary"}:
        return [calm]
    strategies = [
        RandomBaseline(), GPRUCB(alpha=UCB_ALPHA),
        RFRUCB(alpha=UCB_ALPHA, n_estimators=200),
        GBDTUCB(alpha=UCB_ALPHA, n_estimators=300,
                n_bootstrap_models=GBDT_BOOTSTRAP_MODELS),
        NNUCB(alpha=UCB_ALPHA),
        AdaptiveUCB(alpha=UCB_ALPHA, lam_llm=0.0, **ensemble_kwargs),
        LLMRank(), LLMDynamic(),
    ]
    if EXPERIMENT_MODE == "supplementary":
        strategies.append(CALMStatic(
            alpha=UCB_ALPHA, lam_llm=LAM_LLM,
            extreme_rank_power=EWRC_GAMMA, **ensemble_kwargs,
        ))
    return [*strategies, calm]


def _active_strategy_names() -> tuple[str, ...]:
    if EXPERIMENT_MODE == "benchmark":
        return STRATEGY_NAMES
    if EXPERIMENT_MODE == "supplementary":
        return (*STRATEGY_NAMES[:-1], "CALM-static", "CALM")
    return ("CALM",)


def _active_dynamic_names() -> tuple[str, ...]:
    return tuple(name for name in DYNAMIC_STRATEGY_NAMES
                 if name in _active_strategy_names())


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
        **DEFAULT_PROTOCOL,
        "n_runs": N_RUNS,
        "initial_fraction": INITIAL_LABELED_FRACTION,
        "batch_fraction": BATCH_QUERY_FRACTION,
        "total_budget_fraction": TOTAL_LABELED_BUDGET_FRACTION,
        "seed": SEED,
        "initialization": "paper_selected_runs" if SEED is None else "new_seed_runs",
        "experiment": EXPERIMENT_MODE,
        "strategy_names": list(_active_strategy_names()),
        "fixed_lambda": FIXED_LAMBDA,
        "lambda_mode": "fixed" if FIXED_LAMBDA is not None else DEFAULT_PROTOCOL["lambda_mode"],
        "ewrc_gamma": EWRC_GAMMA,
        "rescoring_interval": RESCORING_INTERVAL,
        "llm_scoring_model": LLM_SCORING_MODEL,
        "llm_api_base_url": LLM_API_BASE_URL,
        "dynamic_strategy_names": list(_active_dynamic_names()),
    }






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
    return "full_pool_x_only_target_independent"


def _output_namespace() -> tuple[Path, str]:
    runtime = _runtime_protocol()
    output_root = OUTPUT_ROOT if OUTPUT_ROOT is not None else OUTPUT_DIR
    if runtime == DEFAULT_PROTOCOL:
        return output_root, "paper"
    raw = json.dumps(runtime, sort_keys=True, separators=(",", ":"))
    protocol_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return output_root / "_custom" / protocol_id, "custom"


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
        strategy_idx = list(payload.get("config", {}).get("strategy_names", _active_strategy_names())).index(active.get("strategy"))
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
    """Bind cached scores to the service, model and active scoring code."""
    return {
        "api_base_url": LLM_API_BASE_URL,
        "scoring_model": LLM_SCORING_MODEL,
        "temperature": LLM_SCORING_TEMPERATURE,
        "retrieval_mode": "disabled",
        "prompt_implementation": _sha256_file(PROJECT / "common/llm/prompts.py"),
        "scoring_implementation": _sha256_file(PROJECT / "common/llm/reviewer.py"),
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
                evidence_payload_ok = (
                    expected_retrieval_mode == "disabled"
                    and record.get("retrieval_queries") == {}
                    and record.get("references") == []
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


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_scalar)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def _publish_result_json(path: Path, payload: dict) -> None:
    """Finish an interrupted publication without replacing an existing result."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        canonical = lambda value: json.dumps(
            value, sort_keys=True, default=_json_scalar
        )
        if canonical(existing) != canonical(payload):
            raise RuntimeError(
                f"Refusing to replace existing result {path}; use --output-root "
                "for an independent rerun."
            )
        return
    _write_json_atomic(path, payload)


def _prepare_output_directory(
    out_dir: Path, run_identity: dict, resources: dict,
) -> bool:
    """Bind output ownership and recognize complete runs before remote work.

    The caller holds _run.lock. Outputs without a matching identity are kept
    intact: even apparently complete legacy results cannot prove this request.
    """
    identity_path = out_dir / "_run_identity.json"
    refusal = (
        f"Existing output in {out_dir} cannot be reused for this request; "
        "all files were preserved. Use matching settings/source or "
        "--output-root with a separate directory for an independent rerun."
    )
    if identity_path.exists():
        try:
            existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(refusal + " Invalid run identity.") from exc
        if existing_identity != run_identity:
            raise RuntimeError(refusal + " Run identity mismatch.")
    else:
        if any(path.name != "_run.lock" for path in out_dir.iterdir()):
            raise RuntimeError(refusal + " Missing run identity.")
        _write_json_atomic(identity_path, run_identity)

    conv_path = out_dir / "convergence.json"
    stats_path = out_dir / "stats.json"
    if not conv_path.exists() and not stats_path.exists():
        return False
    if not conv_path.exists() or not stats_path.exists():
        checkpoint_path = out_dir / "_checkpoint.json"
        if not (checkpoint_path.exists() or any(
            out_dir.glob(f"{checkpoint_path.stem}*.tmp")
        )):
            raise RuntimeError(refusal + " Incomplete result publication without checkpoint.")
        return False
    try:
        convergence = json.loads(conv_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        names = list(_active_strategy_names())
        if (list(convergence["strategies"]) != names
                or set(stats["strategies"]) != set(names)
                or convergence["config"].get("run_identity_sha256") != hashlib.sha256(
                    json.dumps(run_identity, sort_keys=True).encode("utf-8")
                ).hexdigest()
                or convergence["config"]["n_runs"] != N_RUNS
                or convergence["config"]["total_budget"] != resources["total_budget"]):
            raise ValueError("result configuration mismatch")
        for runs in convergence["strategies"].values():
            if not isinstance(runs, list) or len(runs) != N_RUNS:
                raise ValueError("incomplete run count")
            for records in runs:
                if (not records
                        or records[-1].get("stop_reason") != "total_budget_reached"
                        or records[-1].get("train_size") != resources["total_budget"]):
                    raise ValueError("incomplete fixed budget")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError,
            TypeError, ValueError, IndexError) as exc:
        raise RuntimeError(refusal + f" Invalid completed results: {exc}") from exc
    return True


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
    key_pool: AsyncAPIKeyPool | None = None,
    evidence_path: Path | None = None,
) -> dict[str, np.ndarray]:
    # Main comparisons and ablations share this identity. Keep the lock across
    # the cache recheck, remote generation, and both publication operations.
    with exclusive_file_lock(cache_path.with_name(cache_path.name + ".lock")):
        return _precompute_llm_scores_locked(
            df=df, comp_df=comp_df, target_col=target_col,
            element_cols=element_cols, process_cols=process_cols,
            material_system=material_system, composition_unit=composition_unit,
            objective_direction=objective_direction, cache_path=cache_path,
            objective_description=objective_description,
            cache_metadata=cache_metadata, forbidden_target_cols=forbidden_target_cols,
            key_pool=key_pool, evidence_path=evidence_path,
        )


def _precompute_llm_scores_locked(
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
    key_pool: AsyncAPIKeyPool | None = None,
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
        retry_base_seconds=LLM_RETRY_BASE_SECONDS,
        retry_max_seconds=LLM_RETRY_MAX_SECONDS,
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
    _write_json_atomic(cache_path, cache_data)
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





def _static_cache_path(dataset: str, metadata: dict) -> Path:
    """Share a task's blind knowledge baseline across paired comparisons."""
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    output_root = OUTPUT_ROOT if OUTPUT_ROOT is not None else OUTPUT_DIR
    return output_root / "_static" / dataset / identity / "_llm_scores.json"


def _static_knowledge_identity(scores: dict, cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    generation = cache.get("_evidence_generation_id")
    values = np.asarray(scores["s_phys"], dtype=np.float64)
    if (not _valid_sha256(generation)
            or not np.array_equal(values, np.asarray(cache.get("s_phys"), dtype=np.float64))):
        raise RuntimeError("Static knowledge generation changed while preparing the run")
    return {
        "evidence_generation_id": generation,
        "score_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def run_dataset(config: DatasetConfig) -> None:
    """Serialize all writers to one task output, including completed reruns."""
    output_namespace, _ = _output_namespace()
    with exclusive_file_lock(output_namespace / config.name / "_run.lock"):
        _run_dataset_locked(config)


def _run_dataset_locked(config: DatasetConfig) -> None:
    paired_run_seeds = paired_seeds(config.name, N_RUNS, seed=SEED)
    run_sources = (PAPER_RUNS[config.name]["source_run_numbers"][:N_RUNS]
                   if SEED is None else None)
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
        "implementation_fingerprint_version": DEFAULT_PROTOCOL[
            "implementation_fingerprint_version"
        ],
        "environment_fingerprint_version": DEFAULT_PROTOCOL[
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

    # Read once and share with column detection, data loading, and LLM review.
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
        "llm_retry_until_success": True,
        "llm_strict_scoring": True,
        "llm_max_concurrency": LLM_MAX_CONCURRENCY,
        "llm_api_key_count": LLM_API_KEY_COUNT,
        "llm_per_key_concurrency": LLM_PER_KEY_CONCURRENCY,
        "llm_key_scheduler": LLM_KEY_SCHEDULER_VERSION,
        "llm_key_slot_alias": f"key_slot_{API_KEY_SLOT}",
        "dataset_key_allocation": DATASET_KEY_ALLOCATION_VERSION,
        "llm_rate_limit_batch_size": LLM_RATE_LIMIT_BATCH_SIZE,
        "llm_rate_limit_sleep_seconds": LLM_RATE_LIMIT_SLEEP_SECONDS,
        "llm_retry_base_seconds": LLM_RETRY_BASE_SECONDS,
        "llm_retry_max_seconds": LLM_RETRY_MAX_SECONDS,
        "dynamic_rescoring_failure_policy": (
            DEFAULT_PROTOCOL["dynamic_rescoring_failure_policy"]
        ),
    }
    print(
        f"[setup] Strict LLM scoring: retry transient failures until success, "
        f"keys={LLM_API_KEY_COUNT}, "
        f"assigned_slot={API_KEY_SLOT}, "
        f"per-key concurrency={LLM_PER_KEY_CONCURRENCY}, "
        f"total concurrency={LLM_MAX_CONCURRENCY}, no static fallback"
    )

    # ── 2) LLM review scores ──
    llm_cache_metadata = {
        "llm_scoring_prompt": LLM_SCORING_PROMPT_VERSION,
        "llm_evidence_version": LLM_EVIDENCE_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
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
    }
    llm_cache = _static_cache_path(config.name, llm_cache_metadata)
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
    run_identity = json.loads(json.dumps({
        "version": "calm_output_identity_v1",
        "dataset": {key: value for key, value in asdict(config).items()
                    if key != "data_path"},
        "dataset_files": [
            _sha256_file(config.data_path),
            _sha256_file(config.data_path.with_suffix(".columns.json")),
            _sha256_file(config.data_path.with_suffix(".dedup.json")),
        ],
        "protocol": _runtime_protocol(),
        "paired_run_seeds": paired_run_seeds,
        "implementation_sha256": implementation_identity["sha256"],
        "environment_sha256": environment_identity["sha256"],
        "llm_cache_metadata": llm_cache_metadata,
        "api_key_slot": API_KEY_SLOT,
    }, ensure_ascii=False, sort_keys=True, default=_json_scalar))
    if _prepare_output_directory(out_dir, run_identity, resources):
        # A crash after the two JSON publications may leave only the optional
        # plot unfinished. Complete it locally without reopening the AL run.
        if not (out_dir / "convergence_plot.png").exists():
            completed = json.loads((out_dir / "convergence.json").read_text(encoding="utf-8"))
            plot_convergence(
                completed["strategies"], global_max, completed["config"],
                out_dir / "convergence_plot.png",
                target_label=config.target_label, target_unit=config.target_unit,
            )
        print(f"[done] {config.name} is already complete; preserved {out_dir}")
        return

    # Registered local inputs and existing outputs are checked before loading
    # credentials, constructing reviewers, or making a scoring request.
    available_keys = require_api_keys(API_KEY_SLOT)
    api_keys = [available_keys[API_KEY_SLOT - 1]]
    shared_key_pool = AsyncAPIKeyPool(
        api_keys,
        base_url=LLM_API_BASE_URL,
        key_aliases=[f"key_slot_{API_KEY_SLOT}"],
        per_key_concurrency=LLM_PER_KEY_CONCURRENCY,
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

    static_knowledge_identity = _static_knowledge_identity(llm_scores, llm_cache)

    # ── 3) Build strategies ──
    strategies = build_strategies()
    strategy_names = [s.name for s in strategies]
    if len(strategy_names) != len(set(strategy_names)):
        raise RuntimeError(f"Duplicate strategy names: {strategy_names}")
    if tuple(strategy_names) != _active_strategy_names():
        raise RuntimeError(
            f"Strategy protocol drift: {strategy_names} != {list(_active_strategy_names())}"
        )
    print(f"[setup] {len(strategies)} strategies: {strategy_names}")

    from common.llm.reviewer import LLMReviewer
    from calm.dynamic_scorer import DynamicLLMScorer

    reviewer_dyn = LLMReviewer(
        model=LLM_SCORING_MODEL,
        temperature=LLM_SCORING_TEMPERATURE,
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
        retry_base_seconds=LLM_RETRY_BASE_SECONDS,
        retry_max_seconds=LLM_RETRY_MAX_SECONDS,
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
            anchor_best_k=ANCHOR_BEST_K,
            anchor_worst_k=ANCHOR_WORST_K,
            anchor_nearby_k=ANCHOR_NEARBY_K,
            candidate_memory_prefilter_k=CANDIDATE_MEMORY_PREFILTER_K,
            audit_path=out_dir / DYNAMIC_EVIDENCE_FILENAMES[strategy_name],
            audit_identity={"run_identity_sha256": hashlib.sha256(
                json.dumps(run_identity, sort_keys=True).encode("utf-8")
            ).hexdigest()},
        )
        for strategy_name in _active_dynamic_names()
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
            f"{list(_active_dynamic_names())} "
            f"(interval={RESCORING_INTERVAL})"
        )
    else:
        print("[setup] Dynamic LLM rescoring is disabled")

    # ── 5) AL benchmark ──
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
        "paired_run_seeds": paired_run_seeds,
        "paper_run_sources": run_sources,
        "experiment": EXPERIMENT_MODE,
        "fixed_lambda": FIXED_LAMBDA,
        "target_col": config.target_col,
        "target_cols": property_cols,
        "model_ready_data_guard": data_guard_metadata,
        "composition_unit": config.composition_unit,
        "composition_missing_policy": config.composition_missing_policy,
        "column_role_schema_version": COLUMN_ROLE_SCHEMA_VERSION,
        "column_roles": column_roles,
        "visible_input_columns": safe_input_cols,
        "source_order_fingerprint": llm_cache_metadata["source_order_fingerprint"],
        "static_knowledge_identity": static_knowledge_identity,
        "ucb_alpha": UCB_ALPHA,
        "lam_llm": LAM_LLM,
        "surrogate_policy": SURROGATE_POLICY_VERSION,
        "nn_surrogate": DEFAULT_PROTOCOL["nn_surrogate"],
        "surrogate_sigma_calibration": SURROGATE_SIGMA_CALIBRATION,
        "ensemble_min_train_samples": ENSEMBLE_MIN_TRAIN_SAMPLES,
        "ensemble_weight_update_samples": ENSEMBLE_WEIGHT_UPDATE_SAMPLES,
        "ensemble_weight_floor": ENSEMBLE_WEIGHT_FLOOR,
        "ensemble_weight_temperature": ENSEMBLE_WEIGHT_TEMPERATURE,
        "ensemble_top_weight_strength": ENSEMBLE_TOP_WEIGHT_STRENGTH,
        "gbdt_bootstrap_models": GBDT_BOOTSTRAP_MODELS,
        "llm_score_mode": "objective_utility_static_s_phys_plus_tie_rank_v6",
        "knowledge_tie_break": KNOWLEDGE_TIE_BREAK_VERSION,
        "knowledge_tie_salt": KNOWLEDGE_TIE_SALT,
        "knowledge_tie_break_audit": knowledge_tie_break_audit,
        "llm_scoring_prompt": LLM_SCORING_PROMPT_VERSION,
        "llm_evidence_version": LLM_EVIDENCE_VERSION,
        "llm_scoring_model": LLM_SCORING_MODEL,
        "llm_api_base_url": LLM_API_BASE_URL,
        "llm_scoring_temperature": LLM_SCORING_TEMPERATURE,
        "llm_retrieval_mode": LLM_RETRIEVAL_MODE,
        "llm_runtime_fingerprint": llm_cache_metadata["llm_runtime_fingerprint"],
        "lambda_mode": "fixed" if FIXED_LAMBDA is not None else DEFAULT_PROTOCOL["lambda_mode"],
        "ewrc_gamma": EWRC_GAMMA,
        "dynamic_rescoring": (
            "disabled_for_static_control"
            if RESCORING_INTERVAL == 0
            else (
                "llmdynamic_and_calm_independent_per_round_shortlist_"
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
        "dynamic_rank_fusion": DEFAULT_PROTOCOL["dynamic_rank_fusion"],
        "llm_dynamic_shortlist_policy": LLM_DYNAMIC_SHORTLIST_POLICY_VERSION,
        "dynamic_strategy_names": list(_active_dynamic_names()),
        "dynamic_evidence_files": dict(DYNAMIC_EVIDENCE_FILENAMES),
        "strategy_seed_mode": "stable_offsets_v1",
        "matched_pair_seed": [
            "Adaptive-UCB",
            "LLM-dynamic",
            "CALM-static",
            "CALM",
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
        run_pool_scores = {"s_phys": np.asarray(llm_scores["s_phys"]).copy(),
                           "_tie_break_rank": knowledge_tie_break_ranks.copy()}

        pending_strategies = [
            s for s in strategies
            if run_idx >= resume_offset.get(s.name, 0)
        ]
        if not pending_strategies:
            continue

        y_run = y
        run_feature_dir = out_dir / "_feature_fits" / f"run_{run_idx:03d}"
        X_run, y_run, _ = load_external_dataset_initial_fit(
            data_path=config.data_path,
            target_col=config.target_col,
            initial_idx=initial_idx,
            element_cols=elem_cols,
            process_cols=proc_cols,
            target_cols=property_cols,
            output_dir=run_feature_dir,
            force_rerun_selection=False,
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
                "CALM",
                "CALM-static",
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

            if sname in dynamic_scorers:
                dynamic_scorers[sname].set_run_context(run_idx, sname)
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
                "CALM",
                "CALM-static",
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
            "run_identity_sha256": hashlib.sha256(
                json.dumps(run_identity, sort_keys=True).encode("utf-8")
            ).hexdigest(),
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
            "paired_run_seeds": paired_run_seeds,
            "paper_run_sources": run_sources,
            "experiment": EXPERIMENT_MODE,
            "fixed_lambda": FIXED_LAMBDA,
            "target_cols": property_cols,
            "model_ready_data_guard": data_guard_metadata,
            "composition_unit": config.composition_unit,
            "composition_missing_policy": config.composition_missing_policy,
            "column_role_schema_version": COLUMN_ROLE_SCHEMA_VERSION,
            "column_roles": column_roles,
            "visible_input_columns": safe_input_cols,
            "source_order_fingerprint": llm_cache_metadata["source_order_fingerprint"],
            "static_knowledge_identity": static_knowledge_identity,
            "ucb_alpha": UCB_ALPHA,
            "lam_llm": LAM_LLM,
            "surrogate_policy": SURROGATE_POLICY_VERSION,
            "nn_surrogate": DEFAULT_PROTOCOL["nn_surrogate"],
            "surrogate_sigma_calibration": SURROGATE_SIGMA_CALIBRATION,
            "ensemble_min_train_samples": ENSEMBLE_MIN_TRAIN_SAMPLES,
            "ensemble_weight_update_samples": ENSEMBLE_WEIGHT_UPDATE_SAMPLES,
            "ensemble_weight_floor": ENSEMBLE_WEIGHT_FLOOR,
            "ensemble_weight_temperature": ENSEMBLE_WEIGHT_TEMPERATURE,
            "ensemble_top_weight_strength": ENSEMBLE_TOP_WEIGHT_STRENGTH,
            "gbdt_bootstrap_models": GBDT_BOOTSTRAP_MODELS,
            "llm_score_mode": "objective_utility_static_s_phys_plus_tie_rank_v6",
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
            "llm_retrieval_mode": LLM_RETRIEVAL_MODE,
            "llm_runtime_fingerprint": llm_cache_metadata["llm_runtime_fingerprint"],
            "lambda_mode": "fixed" if FIXED_LAMBDA is not None else DEFAULT_PROTOCOL["lambda_mode"],
            "ewrc_gamma": EWRC_GAMMA,
            "dynamic_rescoring": (
                "disabled_for_static_control"
                if RESCORING_INTERVAL == 0
                else (
                    "llmdynamic_and_calm_independent_per_round_shortlist_"
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
            "dynamic_rank_fusion": DEFAULT_PROTOCOL["dynamic_rank_fusion"],
            "llm_dynamic_shortlist_policy": (
                LLM_DYNAMIC_SHORTLIST_POLICY_VERSION
            ),
            "dynamic_strategy_names": list(_active_dynamic_names()),
            "dynamic_evidence_files": dict(DYNAMIC_EVIDENCE_FILENAMES),
            "strategy_seed_mode": "stable_offsets_v1",
            "matched_pair_seed": [
                "Adaptive-UCB",
                "LLM-dynamic",
                "CALM-static",
                "CALM",
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
    _publish_result_json(conv_path, conv_output)
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
        reference_strategy="CALM",
    )
    stats_path = out_dir / "stats.json"
    _publish_result_json(stats_path, stats)
    print(f"[save] {stats_path}")

    if not (out_dir / "convergence_plot.png").exists():
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
        description="CALM: the nine-task, single-objective paper benchmark"
    )
    p.add_argument("--dataset", choices=list(DATASETS), default="2xxx_T6__UTS")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Separate output and static-cache root for an independent rerun.")
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--initial-fraction", type=float, default=INITIAL_LABELED_FRACTION)
    p.add_argument("--batch-fraction", type=float, default=BATCH_QUERY_FRACTION)
    p.add_argument("--total-budget-fraction", type=float, default=TOTAL_LABELED_BUDGET_FRACTION)
    p.add_argument("--seed", type=int, default=None,
                   help="Generate new paired runs; omit to use the paper's selected run seeds.")
    p.add_argument("--api-key-slot", type=int, choices=range(1, 5), default=1,
                   help="One-based slot in CALM_API_KEY_1..4.")
    p.add_argument("--experiment", choices=(
        "benchmark", "supplementary", "full", "no-fsr", "no-ewrc",
        "no-fsr-no-ewrc", "spearman"), default="benchmark",
        help="Nine-strategy main comparison, ten-strategy supplement, or one paper ablation.")
    return p.parse_args()


def main():
    global N_RUNS, INITIAL_LABELED_FRACTION, BATCH_QUERY_FRACTION
    global TOTAL_LABELED_BUDGET_FRACTION, SEED, RESCORING_INTERVAL, EWRC_GAMMA
    global API_KEY_SLOT, EXPERIMENT_MODE, FIXED_LAMBDA, OUTPUT_ROOT
    args = parse_args()
    if (args.n_runs < 1
            or not 0 < args.initial_fraction < args.total_budget_fraction <= 1
            or not 0 < args.batch_fraction <= 1
            or (args.seed is not None and not 0 <= args.seed < 2**32)):
        raise ValueError("Require positive runs, valid budget fractions and a seed in [0, 2**32).")
    N_RUNS = args.n_runs
    INITIAL_LABELED_FRACTION = args.initial_fraction
    BATCH_QUERY_FRACTION = args.batch_fraction
    TOTAL_LABELED_BUDGET_FRACTION = args.total_budget_fraction
    SEED = args.seed
    API_KEY_SLOT = args.api_key_slot
    OUTPUT_ROOT = args.output_root.resolve() if args.output_root is not None else None
    EXPERIMENT_MODE = args.experiment
    RESCORING_INTERVAL = 0 if args.experiment in {"no-fsr", "no-fsr-no-ewrc"} else 1
    EWRC_GAMMA = 0.0 if args.experiment == "spearman" else 2.0
    FIXED_LAMBDA = 0.3 if args.experiment in {"no-ewrc", "no-fsr-no-ewrc"} else None
    run_dataset(DATASETS[args.dataset])


if __name__ == "__main__":
    main()
