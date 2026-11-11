"""Actual per-run seeds for the nine manuscript tasks."""
from __future__ import annotations

import numpy as np

PAPER_RUN_SEEDS: dict[str, list[int]] = {
    "Cu_alloys__EC": [
        1608637542, 1273642419, 1935803228, 787846414, 996406378,
        1201263687, 423734972, 415968276, 670094950, 1914837113,
    ],
    "HEA_Invar__TEC": [
        1608637542, 1273642419, 1935803228, 787846414, 996406378,
        1201263687, 423734972, 415968276, 670094950, 1914837113,
    ],
    "MPEA__UTS": [
        1608637542, 1273642419, 1935803228, 787846414, 996406378,
        1201263687, 423734972, 415968276, 670094950, 669991378,
    ],
    "mg_alloy__UTS": [
        1782176804, 1028655772, 465679224, 1114009941, 678236246,
        1400879312, 1283215481, 1709079418, 1324328782, 78442560,
    ],
    "Ti_alloy__YS": [
        1608637542, 1273642419, 1935803228, 787846414, 996406378,
        1201263687, 423734972, 415968276, 670094950, 1914837113,
    ],
    "Ti_alloy__UTS": [
        1608637542, 1273642419, 1935803228, 787846414, 996406378,
        1201263687, 423734972, 415968276, 670094950, 1914837113,
    ],
    "mg_alloy__YS": [
        1608637542, 1273642419, 669991378, 787846414, 996406378,
        1201263687, 423734972, 415968276, 670094950, 1914837113,
    ],
    "2xxx_T6__EL": [
        1608637542, 1273642419, 1935803228, 787846414, 996406378,
        1201263687, 423734972, 415968276, 670094950, 1914837113,
    ],
    "2xxx_T6__UTS": [
        1608637542, 1273642419, 1935803228, 787846414, 996406378,
        1201263687, 423734972, 415968276, 669991378, 1914837113,
    ],
}


def paired_seeds(dataset: str, n_runs: int, *, seed: int | None = None) -> list[int]:
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    if seed is not None:
        rng = np.random.RandomState(seed)
        return [int(rng.randint(0, 2**31 - 1)) for _ in range(n_runs)]
    run_seeds = PAPER_RUN_SEEDS[dataset]
    if n_runs > len(run_seeds):
        raise ValueError("The paper provides ten paired runs; use --seed for a new experiment.")
    return list(run_seeds[:n_runs])
