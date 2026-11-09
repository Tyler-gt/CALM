"""Descriptive fixed-budget search metrics used in the CALM manuscript."""

from __future__ import annotations

import numpy as np

from calm.objective import objective_regret, validate_objective_direction


def _mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else None


def _sample_std(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else None


def _json_values(values: np.ndarray) -> list:
    return [float(value) if np.isfinite(value) else None for value in values]


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
    reference_strategy: str | None = "CALM",
) -> dict:
    """Summarize complete trajectories and paired differences by run position.

    AUC integrates direction-aware raw regret over measured sample count and
    divides by the post-initialization budget span. nAUC additionally divides
    by the target range; it is undefined (JSON null) when that range is zero.
    Standard deviations use the sample definition and require two observations.
    """
    direction = validate_objective_direction(objective_direction)
    names = list(all_records)
    if not names:
        raise ValueError("No strategy records were provided")
    n_runs = len(all_records[names[0]])
    if n_runs == 0 or any(len(all_records[name]) != n_runs for name in names):
        raise ValueError("Every strategy must contain the same non-zero number of runs")
    n_samples, n_init, total_budget = int(n_samples), int(n_init), int(total_budget)
    if not 0 < n_init < total_budget <= n_samples:
        raise ValueError("Invalid initial/total budget for regret AUC")
    target_range = float(global_max) - float(global_min)
    if not np.isfinite(global_max) or not np.isfinite(global_min) or target_range < 0:
        raise ValueError("Target extrema must be finite and ordered")

    n_steps = len(all_records[names[0]][0])
    if n_steps < 2:
        raise ValueError("Each trajectory must include initialization and acquisition")
    curves = np.empty((len(names), n_runs, n_steps), dtype=float)
    sizes = np.empty_like(curves)
    for strategy_index, name in enumerate(names):
        for run_index, run in enumerate(all_records[name]):
            if [int(row["iteration"]) for row in run] != list(range(n_steps)):
                raise ValueError(f"{name}/run{run_index} does not span the complete budget")
            curves[strategy_index, run_index] = [float(row[target_label]) for row in run]
            sizes[strategy_index, run_index] = [int(row["train_size"]) for row in run]
            schedule = sizes[strategy_index, run_index]
            if schedule[0] != n_init or schedule[-1] != total_budget or np.any(np.diff(schedule) <= 0):
                raise ValueError(f"{name}/run{run_index} has an invalid measurement schedule")
            # Pair by initialization and run position, never by search outcome.
            reference_initial = all_records[names[0]][run_index][0]
            for field in ("initial_index_fingerprint", "initial_candidate_indices", "run_seed"):
                if field in reference_initial or field in run[0]:
                    if reference_initial.get(field) != run[0].get(field):
                        raise ValueError(f"{name}/run{run_index} has a different paired {field}")
    if not np.all(np.isfinite(curves)):
        raise ValueError("Best-observed values must be finite")
    schedule = sizes[0, 0]
    if not np.all(sizes == schedule):
        raise ValueError("Strategies and runs must share the query schedule")

    regret = objective_regret(curves, global_min=global_min,
                              global_max=global_max, direction=direction)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = integrate(regret, x=schedule, axis=-1) / float(total_budget - n_init)
    normalized = auc / target_range if target_range > 0 else np.full_like(auc, np.nan)
    optimum = float(global_max if direction == "maximize" else global_min)
    summaries = {}
    for index, name in enumerate(names):
        first_hit_samples = np.full(n_runs, np.nan)
        first_hit_iterations = [None] * n_runs
        for run_index, curve in enumerate(curves[index]):
            hits = np.flatnonzero(np.isclose(curve, optimum, rtol=1e-9, atol=1e-6))
            if len(hits):
                first = int(hits[0])
                first_hit_iterations[run_index] = first
                first_hit_samples[run_index] = schedule[first]
        first_hit_percent = 100.0 * first_hit_samples / n_samples
        summaries[name] = {
            "mean_best_value": np.mean(curves[index], axis=0).tolist(),
            "std_best_value": (
                np.std(curves[index], axis=0, ddof=1).tolist()
                if n_runs > 1 else [None] * n_steps
            ),
            "regret_auc_per_run": auc[index].tolist(),
            "regret_auc_mean": _mean(auc[index]),
            "regret_auc_std": _sample_std(auc[index]),
            "norm_regret_auc_per_run": _json_values(normalized[index]),
            "norm_regret_auc_mean": _mean(normalized[index]),
            "norm_regret_auc_std": _sample_std(normalized[index]),
            "optimum_hit_rate_by_budget": float(np.mean(np.isfinite(first_hit_samples))),
            "successful_runs": int(np.isfinite(first_hit_samples).sum()),
            "first_optimum_iterations": first_hit_iterations,
            "samples_to_optimum_per_run": _json_values(first_hit_samples),
            "samples_to_optimum_mean": _mean(first_hit_samples),
            "first_hit_measured_fraction_percent_per_run": _json_values(first_hit_percent),
            "first_hit_measured_fraction_percent_mean": _mean(first_hit_percent),
            "first_hit_measured_fraction_percent_std": _sample_std(first_hit_percent),
        }

    comparisons = {}
    if reference_strategy is not None:
        if reference_strategy not in names:
            raise ValueError(f"Reference strategy {reference_strategy!r} is absent")
        reference_index = names.index(reference_strategy)
        for index, name in enumerate(names):
            if name == reference_strategy:
                continue
            differences = normalized[reference_index] - normalized[index]
            raw_differences = auc[reference_index] - auc[index]
            comparisons[f"{reference_strategy}_vs_{name}"] = {
                "reference_strategy": reference_strategy,
                "comparator": name,
                "difference_direction": "reference_minus_comparator",
                "n_pairs": n_runs,
                "n_defined_normalized_pairs": int(np.isfinite(differences).sum()),
                "regret_auc_difference_per_run": raw_differences.tolist(),
                "regret_auc_difference_mean": _mean(raw_differences),
                "regret_auc_difference_std": _sample_std(raw_differences),
                "norm_regret_auc_difference_per_run": _json_values(differences),
                "norm_regret_auc_difference_mean": _mean(differences),
                "norm_regret_auc_difference_std": _sample_std(differences),
            }
    return {
        "n_strategies": len(names), "n_runs": n_runs,
        "max_steps": n_steps, "max_iter": n_steps - 1,
        "n_samples": n_samples, "n_init": n_init, "total_budget": total_budget,
        "query_fraction_by_iteration": (schedule / n_samples).tolist(),
        "regret_auc_definition": (
            "direction-aware raw regret integrated by trapezoid over measured "
            "sample count, divided by total_budget - n_init; nAUC divides "
            "again by global_max - global_min"
        ),
        "normalized_regret_auc_defined": bool(target_range > 0),
        "objective_direction": direction,
        "global_optimum": optimum, global_max_label: float(global_max),
        "global_min": float(global_min),
        "strategies": summaries,
        "paired_comparisons": comparisons,
        "statistics": "descriptive paired differences; sample standard deviation",
    }
