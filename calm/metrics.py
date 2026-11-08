"""CALM convergence, regret-AUC, and paired significance statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from calm.experiment import (
    EXPLORATORY_INFERENCE_VERSION,
    PRIMARY_INFERENCE_COMPARATORS,
    PRIMARY_INFERENCE_STRATEGY,
    PRIMARY_INFERENCE_VERSION,
)
from calm.objective import (
    objective_regret,
    validate_objective_direction,
)


def _paired_bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> list[float]:
    """Deterministic percentile CI for the mean paired difference."""
    differences = np.asarray(differences, dtype=float)
    if len(differences) == 1:
        value = float(differences[0])
        return [value, value]
    rng = np.random.RandomState(42)
    samples = rng.choice(
        differences,
        size=(int(n_bootstrap), len(differences)),
        replace=True,
    )
    means = samples.mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return [
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    ]


def _holm_adjust(
    entries: dict[str, dict],
    alpha: float = 0.05,
    *,
    family_size: int | None = None,
) -> None:
    """Attach family-wise Holm adjusted p-values in place.

    ``family_size`` defaults to every configured hypothesis, including an
    entry that could not be tested.  This avoids making the correction less
    conservative merely because one comparison has insufficient data.
    """
    valid = [
        (name, float(entry["p_value"]))
        for name, entry in entries.items()
        if "p_value" in entry and np.isfinite(entry["p_value"])
    ]
    valid.sort(key=lambda item: item[1])
    configured_family_size = (
        len(entries) if family_size is None else int(family_size)
    )
    if configured_family_size < len(valid):
        raise ValueError(
            "Holm family_size cannot be smaller than the number of valid tests"
        )
    running_max = 0.0
    for rank, (name, p_value) in enumerate(valid):
        adjusted = min(1.0, (configured_family_size - rank) * p_value)
        running_max = max(running_max, adjusted)
        entries[name]["p_value_holm"] = float(running_max)
        entries[name]["significant_holm_0.05"] = bool(running_max <= alpha)


def _paired_wilcoxon_entry(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    mean_label_a: str,
    mean_label_b: str,
) -> dict:
    """Compute a paired test and interpretable paired effect sizes."""
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(values_a) & np.isfinite(values_b)
    a = values_a[mask]
    b = values_b[mask]
    if len(a) < 3:
        return {
            "error": "insufficient matched pairs",
            "n_pairs": int(len(a)),
        }

    differences = a - b
    if np.allclose(differences, 0.0, rtol=1e-12, atol=1e-12):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(
            a,
            b,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
    return {
        "W_statistic": float(statistic),
        "p_value": float(p_value),
        "significant_0.05_uncorrected": bool(p_value < 0.05),
        "n_pairs": int(len(a)),
        mean_label_a: float(np.mean(a)),
        mean_label_b: float(np.mean(b)),
        "mean_paired_difference_a_minus_b": float(np.mean(differences)),
        "median_paired_difference_a_minus_b": float(np.median(differences)),
        "mean_paired_difference_95pct_bootstrap_ci": (
            _paired_bootstrap_mean_ci(differences)
        ),
        "paired_win_rate_a_lower": float(np.mean(a < b)),
        "paired_tie_rate": float(np.mean(np.isclose(a, b))),
    }


def compute_stats(
    all_records: dict[str, list[list[dict]]],
    global_max: float,
    target_label: str = "best_uts",
    global_max_label: str = "global_max",
    *,
    global_min: float,
    n_samples: int,
    n_init: int,
    total_budget: int,
    objective_direction: str = "maximize",
    primary_strategy: str | None = None,
    primary_comparators: tuple[str, ...] | list[str] | None = None,
    primary_inference_version: str = "caller_defined_paired_wilcoxon_holm_v1",
    exploratory_inference_version: str = (
        "all_strategy_pairs_paired_wilcoxon_holm_descriptive_v1"
    ),
) -> dict:
    """Compute query-fraction-normalized regret AUC and paired inference.

    All strategies share the same initial set for each run, so Wilcoxon signed-
    rank tests are the primary inference.  The caller may explicitly specify one
    focal strategy and a comparator family; Holm correction is then applied
    only within that confirmatory family.  All strategy pairs are retained in
    a separately corrected exploratory table.  Mann-Whitney U is retained only
    as a backward-compatible unpaired supplementary statistic.
    """
    objective_direction = validate_objective_direction(objective_direction)
    strategy_names = list(all_records.keys())
    if not strategy_names:
        raise ValueError("No strategy records were provided")
    n_runs = len(all_records[strategy_names[0]])
    if n_runs == 0 or any(len(all_records[name]) != n_runs for name in strategy_names):
        raise ValueError("Every strategy must contain the same non-zero number of runs")
    n_samples = int(n_samples)
    n_init = int(n_init)
    total_budget = int(total_budget)
    if not 0 < n_init < total_budget <= n_samples:
        raise ValueError("Invalid initial/total budget for normalized regret AUC")
    target_range = float(global_max) - float(global_min)
    if not np.isfinite(target_range) or target_range <= 0:
        raise ValueError("global_max - global_min must be positive and finite")

    max_steps = max(len(run) for name in strategy_names for run in all_records[name])

    curves = np.full(
        (len(strategy_names), n_runs, max_steps),
        np.nan,
        dtype=np.float64,
    )
    train_sizes = np.full_like(curves, np.nan)
    for strategy_index, name in enumerate(strategy_names):
        for run_index, run in enumerate(all_records[name]):
            iterations = [int(record["iteration"]) for record in run]
            if iterations != list(range(len(run))) or len(run) != max_steps:
                raise ValueError(
                    f"{name}/run{run_index} does not span the complete fixed budget"
                )
            for record in run:
                step = int(record["iteration"])
                curves[strategy_index, run_index, step] = float(record[target_label])
                train_sizes[strategy_index, run_index, step] = int(record["train_size"])
            if int(train_sizes[strategy_index, run_index, 0]) != n_init:
                raise ValueError(f"{name}/run{run_index} starts outside the initial budget")
            if int(train_sizes[strategy_index, run_index, -1]) != total_budget:
                raise ValueError(f"{name}/run{run_index} does not end at total_budget")
            if np.any(np.diff(train_sizes[strategy_index, run_index]) <= 0):
                raise ValueError(f"{name}/run{run_index} has non-increasing train sizes")

    reference_sizes = train_sizes[0, 0]
    if not np.all(train_sizes == reference_sizes):
        raise ValueError("Strategies/runs do not share one fixed query schedule")

    summary: dict[str, dict] = {}
    run_ddof = 1 if n_runs > 1 else 0
    for strategy_index, name in enumerate(strategy_names):
        curve = curves[strategy_index]
        mean_best = np.nanmean(curve, axis=0).tolist()
        std_best = np.nanstd(curve, axis=0, ddof=run_ddof).tolist()
        summary[name] = {
            "mean_best_value": mean_best,
            "std_best_value": std_best,
            # Backward-compatible aliases retained for older plotting/report code.
            "mean_best_uts": mean_best,
            "std_best_uts": std_best,
            "mean_convergence_rate": (
                (
                    np.nanmean(curve, axis=0) - float(global_min)
                    if objective_direction == "maximize"
                    else float(global_max) - np.nanmean(curve, axis=0)
                )
                / target_range
            ).tolist(),
        }

    regret = objective_regret(
        curves,
        global_min=global_min,
        global_max=global_max,
        direction=objective_direction,
    )
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    budget_span = float(total_budget - n_init)
    regret_auc = np.empty((len(strategy_names), n_runs), dtype=float)
    normalized_regret_auc = np.empty_like(regret_auc)
    for strategy_index in range(len(strategy_names)):
        for run_index in range(n_runs):
            area = integrate(
                regret[strategy_index, run_index],
                x=train_sizes[strategy_index, run_index],
            )
            regret_auc[strategy_index, run_index] = float(area / budget_span)
            normalized_regret_auc[strategy_index, run_index] = float(
                area / (target_range * budget_span)
            )

    for strategy_index, name in enumerate(strategy_names):
        auc_values = regret_auc[strategy_index]
        normalized_values = normalized_regret_auc[strategy_index]
        summary[name]["regret_auc_mean"] = float(np.nanmean(auc_values))
        summary[name]["regret_auc_std"] = float(
            np.nanstd(auc_values, ddof=run_ddof)
        )
        summary[name]["norm_regret_auc_mean"] = float(
            np.nanmean(normalized_values)
        )
        summary[name]["norm_regret_auc_std"] = float(
            np.nanstd(normalized_values, ddof=run_ddof)
        )

        hit_iterations: list[float] = []
        samples_to_optimum: list[float] = []
        hit_by_budget: list[bool] = []
        for run in all_records[name]:
            hit_record = next(
                (
                    record
                    for record in run
                    if bool(record.get("global_optimum_found", False))
                    or np.isclose(
                        float(record[target_label]),
                        (
                            global_max
                            if objective_direction == "maximize"
                            else global_min
                        ),
                        rtol=1e-9,
                        atol=1e-6,
                    )
                ),
                None,
            )
            hit_iteration = (
                int(
                    hit_record.get(
                        "first_optimum_iteration", hit_record["iteration"]
                    )
                )
                if hit_record is not None
                else None
            )
            hit_by_budget.append(hit_iteration is not None)
            hit_iterations.append(
                float(hit_iteration) if hit_iteration is not None else np.nan
            )
            samples_to_optimum.append(
                float(
                    hit_record.get(
                        "samples_to_optimum", hit_record["train_size"]
                    )
                )
                if hit_record is not None
                else np.nan
            )

        hit_array = np.asarray(hit_iterations, dtype=float)
        sample_array = np.asarray(samples_to_optimum, dtype=float)
        summary[name].update({
            "optimum_hit_rate_by_budget": float(np.mean(hit_by_budget)),
            "first_optimum_iterations": [
                None if np.isnan(value) else int(value) for value in hit_array
            ],
            "first_optimum_iteration_mean": (
                float(np.nanmean(hit_array))
                if np.isfinite(hit_array).any()
                else None
            ),
            "first_optimum_iteration_median": (
                float(np.nanmedian(hit_array))
                if np.isfinite(hit_array).any()
                else None
            ),
            "first_optimum_iteration_std": (
                float(np.nanstd(hit_array, ddof=1))
                if np.isfinite(hit_array).sum() >= 2
                else None
            ),
            "samples_to_optimum_mean": (
                float(np.nanmean(sample_array))
                if np.isfinite(sample_array).any()
                else None
            ),
        })

    strategy_auc = {
        name: normalized_regret_auc[index]
        for index, name in enumerate(strategy_names)
    }

    exploratory_paired_auc_tests: dict[str, dict] = {}
    mw_tests: dict[str, dict] = {}
    for index_a, name_a in enumerate(strategy_names):
        for index_b in range(index_a + 1, len(strategy_names)):
            name_b = strategy_names[index_b]
            comparison = f"{name_a}_vs_{name_b}"
            auc_a = normalized_regret_auc[index_a]
            auc_b = normalized_regret_auc[index_b]
            exploratory_paired_auc_tests[comparison] = _paired_wilcoxon_entry(
                auc_a,
                auc_b,
                mean_label_a="norm_regret_auc_a",
                mean_label_b="norm_regret_auc_b",
            )

            mask = np.isfinite(auc_a) & np.isfinite(auc_b)
            if mask.sum() < 3:
                mw_tests[comparison] = {"error": "insufficient data"}
            else:
                statistic, p_value = mannwhitneyu(
                    auc_a[mask], auc_b[mask], alternative="two-sided"
                )
                mw_tests[comparison] = {
                    "U_statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant_0.05": bool(p_value < 0.05),
                    "norm_regret_auc_a": float(np.mean(auc_a[mask])),
                    "norm_regret_auc_b": float(np.mean(auc_b[mask])),
                }
    if primary_strategy is None:
        if primary_comparators is not None:
            raise ValueError(
                "primary_comparators requires an explicit primary_strategy"
            )
        # Generic callers with no configured focal strategy retain the
        # legacy behavior: every available pair forms the primary family.
        primary_paired_auc_tests = {
            name: dict(entry)
            for name, entry in exploratory_paired_auc_tests.items()
        }
        primary_family_strategy = None
        primary_family_comparators = []
        primary_family_description = "all_available_strategy_pairs"
        _holm_adjust(primary_paired_auc_tests)
    else:
        if primary_strategy not in strategy_auc:
            raise ValueError(
                f"Primary strategy {primary_strategy!r} is not present in records"
            )
        if primary_comparators is None:
            raise ValueError(
                "An explicit primary_strategy requires primary_comparators"
            )
        primary_family_comparators = [
            str(name) for name in primary_comparators
        ]
        if len(primary_family_comparators) != len(
            set(primary_family_comparators)
        ):
            raise ValueError("primary_comparators contains duplicate strategies")
        if primary_strategy in primary_family_comparators:
            raise ValueError(
                "primary_strategy cannot also be a primary comparator"
            )
        missing = [
            name
            for name in primary_family_comparators
            if name not in strategy_auc
        ]
        if missing:
            raise ValueError(
                f"Primary comparator strategies are missing from records: {missing}"
            )

        primary_paired_auc_tests = {}
        for comparator in primary_family_comparators:
            comparison = f"{primary_strategy}_vs_{comparator}"
            primary_paired_auc_tests[comparison] = _paired_wilcoxon_entry(
                strategy_auc[primary_strategy],
                strategy_auc[comparator],
                mean_label_a="norm_regret_auc_primary",
                mean_label_b="norm_regret_auc_comparator",
            )
        _holm_adjust(
            primary_paired_auc_tests,
            family_size=len(primary_family_comparators),
        )
        single_confirmatory_comparison = (
            len(primary_family_comparators) == 1
        )
        for entry in primary_paired_auc_tests.values():
            mean_difference = entry.get(
                "mean_paired_difference_a_minus_b"
            )
            direction_supported = bool(
                mean_difference is not None
                and np.isfinite(mean_difference)
                and float(mean_difference) < 0.0
            )
            entry["primary_direction_supported"] = direction_supported
            claim_supported = bool(
                direction_supported
                and entry.get("significant_holm_0.05", False)
            )
            entry["primary_claim_supported_0.05"] = claim_supported
            # Kept as a compatibility alias for readers of older result JSON.
            entry["primary_claim_supported_holm_0.05"] = claim_supported
        primary_family_strategy = primary_strategy
        primary_family_description = (
            "focal_strategy_vs_configured_primary_comparators"
        )
    primary_family_size = len(primary_paired_auc_tests)
    single_confirmatory_comparison = primary_family_size == 1

    return {
        "n_strategies": len(strategy_names),
        "n_runs": n_runs,
        "max_steps": max_steps,
        "max_iter": max_steps - 1,
        "n_samples": n_samples,
        "n_init": n_init,
        "total_budget": total_budget,
        "query_fraction_by_iteration": (reference_sizes / n_samples).tolist(),
        "regret_auc_definition": (
            f"{objective_direction}_raw_target_simple_regret_"
            "trapezoid_over_train_size_divided_by_budget_span; "
            "normalized_again_by_global_target_range"
        ),
        "objective_direction": objective_direction,
        "global_optimum": float(
            global_max if objective_direction == "maximize" else global_min
        ),
        global_max_label: float(global_max),
        "global_min": float(global_min),
        "strategies": summary,
        "primary_inference": (
            "paired_wilcoxon_two_sided_one_preregistered_comparison"
            if single_confirmatory_comparison
            else
            "paired_wilcoxon_two_sided_holm_fwer_0.05_on_configured_primary_family"
        ),
        "primary_inference_version": str(primary_inference_version),
        "primary_test_family": {
            "description": primary_family_description,
            "confirmatory": True,
            "focal_strategy": primary_family_strategy,
            "comparators": primary_family_comparators,
            "family_size": len(primary_paired_auc_tests),
            "metric": "normalized_regret_auc",
            "lower_is_better": True,
            "multiplicity_correction": (
                "none; one preregistered comparison"
                if single_confirmatory_comparison
                else "Holm FWER 0.05"
            ),
        },
        "paired_wilcoxon_input": "normalized_regret_auc",
        "primary_paired_wilcoxon_regret_auc": primary_paired_auc_tests,
        # Backward-compatible alias.  Its scope is now the explicitly
        # configured primary family, not every possible strategy pair.
        "paired_wilcoxon_regret_auc": primary_paired_auc_tests,
        "exploratory_inference": (
            "all_pairwise_paired_wilcoxon_two_sided_unadjusted_descriptive_only"
        ),
        "exploratory_inference_version": str(exploratory_inference_version),
        "exploratory_test_family": {
            "description": "all_available_strategy_pairs",
            "confirmatory": False,
            "family_size": len(exploratory_paired_auc_tests),
            "metric": "normalized_regret_auc",
            "lower_is_better": True,
            "multiplicity_correction": "none; descriptive only",
        },
        "exploratory_all_pairwise_wilcoxon_regret_auc": (
            exploratory_paired_auc_tests
        ),
        "mann_whitney_u": mw_tests,
    }


def compute_stats_from_convergence(convergence: dict) -> dict:
    """Recompute the current statistical analysis from saved AL trajectories."""
    config = convergence.get("config")
    all_records = convergence.get("strategies")
    if not isinstance(config, dict) or not isinstance(all_records, dict):
        raise ValueError("convergence payload must contain config and strategies")
    if not all_records:
        raise ValueError("convergence payload contains no strategies")

    first_runs = next(iter(all_records.values()))
    if not first_runs or not first_runs[0]:
        raise ValueError("convergence strategies must contain at least one run")
    best_keys = [
        str(key)
        for key in first_runs[0][0]
        if str(key).startswith("best_")
    ]
    if len(best_keys) != 1:
        raise ValueError(
            f"Expected exactly one best_* record field, got {best_keys}"
        )

    required_config = (
        "global_max",
        "global_min",
        "objective_direction",
        "n_samples",
        "n_init",
        "total_budget",
    )
    missing = [key for key in required_config if key not in config]
    if missing:
        raise ValueError(f"convergence config is missing fields: {missing}")

    return compute_stats(
        all_records,
        float(config["global_max"]),
        target_label=best_keys[0],
        global_min=float(config["global_min"]),
        n_samples=int(config["n_samples"]),
        n_init=int(config["n_init"]),
        total_budget=int(config["total_budget"]),
        objective_direction=str(config["objective_direction"]),
        primary_strategy=PRIMARY_INFERENCE_STRATEGY,
        primary_comparators=PRIMARY_INFERENCE_COMPARATORS,
        primary_inference_version=PRIMARY_INFERENCE_VERSION,
        exploratory_inference_version=EXPLORATORY_INFERENCE_VERSION,
    )


def recompute_stats_file(
    convergence_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Atomically rewrite stats.json from an existing convergence.json."""
    convergence_path = Path(convergence_path)
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    stats = compute_stats_from_convergence(convergence)
    destination = (
        Path(output_path)
        if output_path is not None
        else convergence_path.with_name("stats.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute CALM stats from completed convergence files"
    )
    parser.add_argument(
        "convergence",
        nargs="+",
        type=Path,
        help="One or more completed convergence.json files",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    for convergence_path in args.convergence:
        destination = recompute_stats_file(convergence_path)
        print(f"[stats] {convergence_path} -> {destination}")


if __name__ == "__main__":
    _main()
