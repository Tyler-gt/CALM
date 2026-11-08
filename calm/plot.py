"""CALM publication plots for the proportional-budget AL benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calm.objective import (
    objective_regret,
    validate_objective_direction,
)


COLORS = {
    "Random": "#7f7f7f",
    "GPR-UCB": "#1f77b4",
    "RFR-UCB": "#ff7f0e",
    "GBDT-UCB": "#2ca02c",
    "NN-UCB": "#17becf",
    "Adaptive-UCB": "#8c564b",
    "LLM-rank": "#9467bd",
    "LLM-dynamic": "#bcbd22",
    "CALM-static": "#e377c2",
    "CALM": "#d62728",
}
LINE_STYLES = {
    "Random": "--",
    "GPR-UCB": "-",
    "RFR-UCB": "-",
    "GBDT-UCB": "-",
    "NN-UCB": "-",
    "Adaptive-UCB": "--",
    "LLM-rank": "-.",
    "LLM-dynamic": ":",
    "CALM-static": "--",
    "CALM": "-",
}
MARKERS = {
    "Random": "s",
    "GPR-UCB": "o",
    "RFR-UCB": "D",
    "GBDT-UCB": "^",
    "NN-UCB": "p",
    "Adaptive-UCB": "X",
    "LLM-rank": "v",
    "LLM-dynamic": "h",
    "CALM-static": "P",
    "CALM": "*",
}


def plot_convergence(
    all_records: dict[str, list[list[dict]]],
    global_max: float,
    config: dict,
    out_path: Path,
    target_label: str = "UTS",
    target_unit: str = "MPa",
) -> None:
    """Plot best-so-far, normalized regret, and Hit@40% for all strategies."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5))

    strategy_names = list(all_records)
    sample_run = all_records[strategy_names[0]][0]
    record_key = next(key for key in sample_run[0] if key.startswith("best_"))
    n_samples = int(config["n_samples"])
    n_init = int(config["n_init"])
    total_budget = int(config["total_budget"])
    init_percent = 100.0 * n_init / n_samples
    budget_percent = 100.0 * total_budget / n_samples
    global_min = float(config["global_min"])
    objective_direction = validate_objective_direction(
        config.get("objective_direction", "maximize")
    )
    global_optimum = (
        float(global_max)
        if objective_direction == "maximize"
        else global_min
    )
    target_range = float(global_max) - global_min
    if target_range <= 0:
        raise ValueError("Global target range must be positive for plotting")

    queried_percent = (
        np.asarray([record["train_size"] for record in sample_run], dtype=float)
        / n_samples
        * 100.0
    )
    marker_every = max(1, len(queried_percent) // 5)
    hit_rates: list[float] = []

    for name in strategy_names:
        runs = all_records[name]
        values = np.asarray(
            [[float(record[record_key]) for record in run] for run in runs],
            dtype=float,
        )
        mean = np.mean(values, axis=0)
        ddof = 1 if len(runs) > 1 else 0
        std = np.std(values, axis=0, ddof=ddof)
        color = COLORS.get(name, "#333333")
        line_style = LINE_STYLES.get(name, "-")
        marker = MARKERS.get(name, "")

        ax1.plot(
            queried_percent,
            mean,
            color=color,
            ls=line_style,
            marker=marker,
            markevery=marker_every,
            markersize=5,
            label=name,
            linewidth=1.8,
        )
        ax1.fill_between(
            queried_percent,
            mean - std,
            mean + std,
            color=color,
            alpha=0.12,
        )

        normalized_regret = objective_regret(
            values,
            global_min=global_min,
            global_max=global_max,
            direction=objective_direction,
        ) / target_range
        regret_mean = np.mean(normalized_regret, axis=0)
        regret_std = np.std(normalized_regret, axis=0, ddof=ddof)
        ax2.plot(
            queried_percent,
            regret_mean,
            color=color,
            ls=line_style,
            marker=marker,
            markevery=marker_every,
            markersize=5,
            label=name,
            linewidth=1.8,
        )
        ax2.fill_between(
            queried_percent,
            np.maximum(regret_mean - regret_std, 0.0),
            regret_mean + regret_std,
            color=color,
            alpha=0.12,
        )
        hit_rates.append(float(np.mean([
            bool(run[-1].get("global_optimum_found", False)) for run in runs
        ])))

    for axis, ylabel in (
        (
            ax1,
            f"Best-so-far {target_label} ({target_unit}; "
            f"{'higher' if objective_direction == 'maximize' else 'lower'} is better)",
        ),
        (ax2, "Normalized simple regret"),
    ):
        axis.set_xlabel("Labeled candidate-pool fraction (%)")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)

    dataset_name = config.get("dataset", config.get("dataset_name", "CALM"))
    ax1.axhline(global_optimum, color="#333333", ls=":", lw=1.2)
    ax1.set_title(
        f"{dataset_name}: {init_percent:.1f}%-{budget_percent:.1f}% AL benchmark "
        f"({config.get('n_runs', 10)} runs)"
    )
    ax2.set_title("Regret trajectory (lower is better)")

    positions = np.arange(len(strategy_names))
    bars = ax3.bar(
        positions,
        hit_rates,
        color=[COLORS.get(name, "#333333") for name in strategy_names],
        alpha=0.85,
    )
    ax3.set_xticks(positions, strategy_names, rotation=35, ha="right", fontsize=8)
    ax3.set_ylim(0.0, 1.05)
    ax3.set_xlabel("Strategy")
    ax3.set_ylabel("Global-optimum hit rate")
    ax3.set_title(f"Hit@{budget_percent:.1f}% budget")
    ax3.grid(True, axis="y", alpha=0.3)
    for bar, rate in zip(bars, hit_rates):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 0.025, 1.02),
            f"{rate:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")
