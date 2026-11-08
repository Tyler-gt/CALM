"""小样本安全的确定性特征筛选。

正式路径仅使用可观测协变量删除常量和高度冗余列，不使用目标值。
可通过 ``select_features()`` 以编程方式调用，也可通过 ``main()`` CLI 使用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

# ── 默认常量 (CLI / 快速调用使用) ──────────────────────────────────
DEFAULT_PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = DEFAULT_PROJECT / "dabase" / "2xxx_mn_features_for_strength_tc.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT / "dabase"
DEFAULT_TARGETS = ["Strength", "TC"]
DEFAULT_DROP_COLS = ["Alloy"]
# Feature removal must be dataset-specific and explicitly requested by callers.
# In particular, Al is a meaningful alloying element outside Al-base datasets
# and must not be silently removed by a shared feature-selection utility.
FEATURE_SELECTION_POLICY_VERSION = "deterministic_unsupervised_pool_redundancy_v3"
DEFAULT_FORCE_DROP_COLS: list[str] = []
DEFAULT_FORCE_KEEP_COLS = [
    "Al", "Cu", "Mg", "Mn", "Si", "Fe", "Cr", "Zn", "Ti", "Zr", "Ni", "Bi", "Pb",
]

# 超参（调优后基本不需改）
VARIANCE_THRESHOLD = 1e-10
CORR_THRESHOLD = 0.95
KFOLD_SPLITS = 5
PERM_N_REPEATS = 20
TOP_K_PER_TARGET = 25
IMPORTANCE_FREQ_THRESHOLD = 0.60
RANDOM_STATE = 42

# The publication path uses target-independent screening.  When
# ``features_initial_fit.csv`` is accompanied by ``features_full_all_rows.csv``,
# the latter is used only for X-side diagnostics (never for response values).
UNSUPERVISED_SELECTION_METHOD = "deterministic_unsupervised_redundancy_v1"
FULL_POOL_REFERENCE_NAME = "features_full_all_rows.csv"
MIN_CORRELATION_OBSERVATIONS = 12
SMALL_SAMPLE_CORR_THRESHOLD = 0.999999
MEDIUM_SAMPLE_CORR_THRESHOLD = 0.98


# ── 内部工具函数 ──────────────────────────────────────────────────

def _prepare_xy(df: pd.DataFrame, targets: Iterable[str],
                drop_cols: Iterable[str], force_drop_cols: Iterable[str]):
    targets = list(targets)
    drop_cols = list(drop_cols)
    force_drop_cols = list(force_drop_cols)
    feature_cols = [c for c in df.columns
                    if c not in set(targets + drop_cols + force_drop_cols)]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[targets].apply(pd.to_numeric, errors="coerce")
    valid_mask = y.notna().all(axis=1)
    x = x.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    return x, y


def _variance_filter(x: pd.DataFrame):
    variances = x.var(axis=0, skipna=True)
    keep = variances[variances > VARIANCE_THRESHOLD].index.tolist()
    return x[keep].copy(), variances


def _feature_group(column: str, force_keep_cols: Iterable[str]) -> str:
    """Return a stable, human-readable feature family for audit reports."""
    if column in set(force_keep_cols):
        return "raw_input_anchor"
    if str(column).startswith("MagpieData "):
        return "magpie_descriptor"
    if str(column).startswith("LLM_"):
        return "llm_designed"
    if str(column) in {"comp_entropy", "radius_mismatch"}:
        return "physics_summary"
    return "engineered_other"


def _load_unsupervised_screening_matrix(
    data_path: Path,
    x_fit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Load an optional all-candidate X matrix without consulting any labels.

    The AL-safe loader writes ``features_initial_fit.csv`` next to
    ``features_full_all_rows.csv``.  The full file masks unqueried responses,
    but this function goes one step further and reads only columns already
    identified as features in ``x_fit``.  Candidate-pool covariates are
    observable in pool-based active learning and make constant/redundancy
    screening much more reliable than estimating correlations from a tiny
    initial labelled subset.
    """
    reference_path = data_path.with_name(FULL_POOL_REFERENCE_NAME)
    use_reference = (
        data_path.name == "features_initial_fit.csv"
        and reference_path.is_file()
    )
    if use_reference:
        reference_columns = pd.read_csv(reference_path, nrows=0).columns
        missing = [c for c in x_fit.columns if c not in reference_columns]
        if not missing:
            # ``usecols`` makes the no-target boundary explicit: response
            # columns are not even loaded from disk.
            reference_df = pd.read_csv(
                reference_path,
                usecols=list(x_fit.columns),
            )
            x_reference = reference_df.loc[:, x_fit.columns].apply(
                pd.to_numeric,
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
            return x_reference.reset_index(drop=True), {
                "scope": "candidate_pool_covariates",
                "source": str(reference_path),
                "n_rows": int(len(x_reference)),
                "used_target_values": False,
                "reference_detected": True,
                "reference_missing_feature_columns": [],
            }
    else:
        missing = []

    return x_fit.replace([np.inf, -np.inf], np.nan).copy(), {
        "scope": "selection_input_covariates",
        "source": str(data_path),
        "n_rows": int(len(x_fit)),
        "used_target_values": False,
        "reference_detected": bool(use_reference),
        "reference_missing_feature_columns": missing,
    }


def _is_numerically_constant(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return bool(float(np.max(values) - np.min(values)) <= 1e-12 + 1e-10 * scale)


def _unsupervised_constant_filter(
    x_fit: pd.DataFrame,
    x_screen: pd.DataFrame,
    force_keep_cols: list[str],
    *,
    screening_scope: str,
) -> tuple[pd.DataFrame, dict]:
    """Remove unusable/full-pool constants without looking at ``y``.

    A force-kept raw input that is constant only in the labelled initial set is
    retained because it may vary in the candidate pool.  If an all-candidate
    reference matrix is available and proves that the same column is constant,
    it is safely removed.
    """
    force_keep = set(force_keep_cols)
    kept: list[str] = []
    all_missing: list[str] = []
    constants_dropped: list[str] = []
    force_kept_initial_constants: list[str] = []
    stats: dict[str, dict] = {}
    has_pool_reference = screening_scope == "candidate_pool_covariates"

    for column in x_fit.columns:
        values = pd.to_numeric(x_screen[column], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        is_all_missing = finite.empty
        is_constant = _is_numerically_constant(finite)
        variance = float(finite.var(ddof=0)) if len(finite) else None
        stats[column] = {
            "n_observed": int(len(finite)),
            "missing_fraction": float(1.0 - len(finite) / max(1, len(values))),
            "n_unique": int(finite.nunique(dropna=True)),
            "variance": variance,
            "is_constant": bool(is_constant),
        }

        if is_all_missing:
            all_missing.append(column)
            continue
        if is_constant:
            if column in force_keep and not has_pool_reference:
                kept.append(column)
                force_kept_initial_constants.append(column)
            else:
                constants_dropped.append(column)
            continue
        kept.append(column)

    return x_fit.loc[:, kept].copy(), {
        "all_missing_dropped": all_missing,
        "constant_features_dropped": constants_dropped,
        "force_kept_initial_constants_without_pool_reference": (
            force_kept_initial_constants
        ),
        "feature_diagnostics": stats,
    }


def _adaptive_correlation_policy(n_rows: int) -> tuple[float, int]:
    """Use conservative redundancy thresholds when X itself is very small."""
    if n_rows < MIN_CORRELATION_OBSERVATIONS:
        return SMALL_SAMPLE_CORR_THRESHOLD, max(3, n_rows)
    if n_rows < 30:
        return MEDIUM_SAMPLE_CORR_THRESHOLD, MIN_CORRELATION_OBSERVATIONS
    return CORR_THRESHOLD, MIN_CORRELATION_OBSERVATIONS


def _representative_key(
    column: str,
    groups: dict[str, str],
    diagnostics: dict[str, dict],
) -> tuple:
    """Rank representatives without target values or input-column order."""
    group_priority = {
        "raw_input_anchor": 0,
        "physics_summary": 1,
        "magpie_descriptor": 2,
        "llm_designed": 3,
        "engineered_other": 4,
    }
    info = diagnostics[column]
    return (
        group_priority.get(groups[column], 99),
        float(info["missing_fraction"]),
        -int(info["n_unique"]),
        str(column).casefold(),
        str(column),
    )


def _unsupervised_redundancy_filter(
    x_fit: pd.DataFrame,
    x_screen: pd.DataFrame,
    force_keep_cols: list[str],
    diagnostics: dict[str, dict],
) -> tuple[pd.DataFrame, list[dict], float, int, dict[str, str]]:
    """Greedily retain deterministic representatives of correlated columns."""
    columns = list(x_fit.columns)
    force_keep = {c for c in force_keep_cols if c in columns}
    groups = {c: _feature_group(c, force_keep) for c in columns}
    threshold, min_observations = _adaptive_correlation_policy(len(x_screen))

    corr = x_screen.loc[:, columns].corr(
        method="pearson",
        min_periods=min_observations,
    ).abs()
    ordered = sorted(
        columns,
        key=lambda c: _representative_key(c, groups, diagnostics),
    )
    kept: list[str] = []
    dropped: list[dict] = []

    for column in ordered:
        if column in force_keep:
            # Composition closure and one-hot encodings can make two raw
            # anchors collinear.  They remain explicit inputs by policy.
            kept.append(column)
            continue

        conflicts: list[tuple[float, str]] = []
        for representative in kept:
            value = corr.loc[column, representative]
            if pd.notna(value) and float(value) >= threshold:
                conflicts.append((float(value), representative))
        if not conflicts:
            kept.append(column)
            continue

        correlation, representative = sorted(
            conflicts,
            key=lambda item: (
                -item[0],
                _representative_key(item[1], groups, diagnostics),
            ),
        )[0]
        overlap = int(
            (
                x_screen[column].notna()
                & x_screen[representative].notna()
            ).sum()
        )
        dropped.append(
            {
                "dropped": column,
                "kept_representative": representative,
                "absolute_pearson_correlation": correlation,
                "n_pairwise_observations": overlap,
                "reason": (
                    "perfect_linear_redundancy"
                    if correlation >= 1.0 - 1e-12
                    else "high_correlation_redundancy"
                ),
                "dropped_group": groups[column],
                "kept_group": groups[representative],
            }
        )

    kept_set = set(kept)
    kept_in_input_order = [c for c in columns if c in kept_set]
    return (
        x_fit.loc[:, kept_in_input_order].copy(),
        sorted(dropped, key=lambda item: item["dropped"]),
        threshold,
        min_observations,
        groups,
    )


def _group_audit(
    columns: Iterable[str],
    groups: dict[str, str],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for column in columns:
        grouped.setdefault(groups[column], []).append(column)
    return {
        group: sorted(group_columns)
        for group, group_columns in sorted(grouped.items())
    }


def _mi_scores(x: pd.DataFrame, y: pd.DataFrame, targets: list[str]):
    imp = SimpleImputer(strategy="median")
    x_imp = pd.DataFrame(imp.fit_transform(x), columns=x.columns, index=x.index)
    score_df = pd.DataFrame(index=x.columns)
    for t in targets:
        mi = mutual_info_regression(x_imp, y[t], random_state=RANDOM_STATE)
        score_df[f"mi_{t.lower()}"] = mi
    score_df["mi_mean"] = score_df.mean(axis=1)
    return score_df


def _correlation_filter(x: pd.DataFrame, score_df: pd.DataFrame,
                        force_keep_cols: list[str]):
    corr = x.corr().abs()
    cols = list(corr.columns)
    to_drop: set[str] = set()
    for i in range(len(cols)):
        ci = cols[i]
        if ci in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            cj = cols[j]
            if cj in to_drop:
                continue
            if corr.loc[ci, cj] >= CORR_THRESHOLD:
                ci_forced = ci in force_keep_cols
                cj_forced = cj in force_keep_cols
                if ci_forced and cj_forced:
                    continue
                if ci_forced and not cj_forced:
                    to_drop.add(cj)
                    continue
                if cj_forced and not ci_forced:
                    to_drop.add(ci)
                    break
                keep_ci = score_df.loc[ci, "mi_mean"] >= score_df.loc[cj, "mi_mean"]
                to_drop.add(cj if keep_ci else ci)
                if not keep_ci:
                    break
    kept = [c for c in cols if c not in to_drop]
    return x[kept].copy(), sorted(to_drop)


def _stability_selection_for_target(
    x: pd.DataFrame, y_series: pd.Series, target_name: str,
):
    """Optional supervised selector with strict evidence gates.

    The publication path no longer calls this function, but it remains
    available for API compatibility and exploratory work.  A feature counts
    in a fold only when its mean permutation importance is strictly positive
    and greater than its repeat-level standard error.  Consequently an
    all-zero importance vector can never be converted into an arbitrary top-k.
    """
    imp = SimpleImputer(strategy="median")
    x_imp = pd.DataFrame(imp.fit_transform(x), columns=x.columns, index=x.index)
    n_samples, n_features = x_imp.shape
    if n_samples < 4 or n_features == 0:
        empty = pd.Series(dtype=float)
        zeros = pd.Series(0.0, index=x.columns, dtype=float)
        empty.attrs["audit"] = {
            "informative": False,
            "reason": "insufficient_samples_or_features",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_splits": 0,
        }
        return empty, zeros, zeros.copy()

    # Keep at least two validation observations and avoid training folds that
    # are too small for a tree split.
    n_splits = min(KFOLD_SPLITS, max(2, n_samples // 4))
    n_splits = min(n_splits, n_samples // 2)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    freq = pd.Series(0.0, index=x.columns)
    fold_imp_table = pd.DataFrame(
        index=x.columns,
        columns=range(n_splits),
        dtype=float,
    )
    fold_noise_table = pd.DataFrame(
        index=x.columns,
        columns=range(n_splits),
        dtype=float,
    )
    informative_folds = 0
    adaptive_top_k: list[int] = []
    adaptive_min_child_samples: list[int] = []

    for fold_idx, (tr, va) in enumerate(kf.split(x_imp)):
        x_tr, x_va = x_imp.iloc[tr], x_imp.iloc[va]
        y_tr, y_va = y_series.iloc[tr], y_series.iloc[va]
        if y_tr.nunique(dropna=True) < 2 or y_va.nunique(dropna=True) < 2:
            fold_imp_table[fold_idx] = 0.0
            fold_noise_table[fold_idx] = 0.0
            adaptive_top_k.append(0)
            adaptive_min_child_samples.append(0)
            continue

        min_child_samples = max(2, min(20, len(tr) // 5))
        num_leaves = max(3, min(15, len(tr) // min_child_samples + 1))
        model = LGBMRegressor(
            objective="regression",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=RANDOM_STATE,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(x_tr, y_tr)
        result = permutation_importance(
            model, x_va, y_va,
            n_repeats=PERM_N_REPEATS, random_state=RANDOM_STATE,
            scoring="r2", n_jobs=1,
        )
        fold_imp = pd.Series(
            np.asarray(result.importances_mean, dtype=float),
            index=x.columns,
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        repeat_std = pd.Series(
            np.asarray(result.importances_std, dtype=float),
            index=x.columns,
        ).replace([np.inf, -np.inf], np.nan).fillna(np.inf)
        noise_floor = repeat_std / np.sqrt(max(1, PERM_N_REPEATS))
        fold_imp_table[fold_idx] = fold_imp
        fold_noise_table[fold_idx] = noise_floor

        eligible = fold_imp[
            (fold_imp > 0.0)
            & (fold_imp > noise_floor)
        ].sort_values(ascending=False, kind="mergesort")
        top_k = min(
            TOP_K_PER_TARGET,
            max(1, len(tr) // 3),
            n_features,
        )
        topk = eligible.head(top_k).index
        freq.loc[topk] += 1.0
        adaptive_top_k.append(int(top_k))
        adaptive_min_child_samples.append(int(min_child_samples))
        if len(topk):
            informative_folds += 1

    freq = freq / n_splits
    perm_mean = fold_imp_table.mean(axis=1)
    perm_std = fold_imp_table.std(axis=1)

    selected = freq[freq >= IMPORTANCE_FREQ_THRESHOLD].sort_values(ascending=False)
    selected.attrs["audit"] = {
        "informative": bool(len(selected)),
        "reason": (
            "strictly_positive_above_noise_features_found"
            if len(selected)
            else "no_feature_passed_positive_above_noise_gate"
        ),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_splits": int(n_splits),
        "informative_folds": int(informative_folds),
        "adaptive_top_k_per_fold": adaptive_top_k,
        "adaptive_min_child_samples_per_fold": adaptive_min_child_samples,
        "importance_gate": "mean > 0 and mean > repeat_std/sqrt(n_repeats)",
    }
    perm_mean.attrs["fold_noise_floor_mean"] = {
        c: float(v)
        for c, v in fold_noise_table.mean(axis=1).items()
    }
    print(
        f"[{target_name}] permutation-stability selected {len(selected)} features "
        f"(freq>={IMPORTANCE_FREQ_THRESHOLD}, "
        f"adaptive top-k over {n_splits} folds; positive-above-noise only)"
    )
    return selected, perm_mean, perm_std


# ── 主入口 (可编程调用) ───────────────────────────────────────────

def select_features(
    data_path: str | Path,
    targets: list[str],
    *,
    force_keep_cols: list[str] | None = None,
    force_drop_cols: list[str] | None = None,
    extra_drop_cols: list[str] | None = None,
    output_dir: str | Path | None = None,
    report_name: str | None = None,
) -> dict:
    """对全特征 CSV 做确定性、目标无关的冗余筛选。

    Parameters
    ----------
    data_path: 全特征 CSV 路径 (feature_construct 产出).
    targets: 目标列名列表, 如 ``["Strength"]`` 或 ``["Strength", "TC"]``.
    force_keep_cols: 原始输入锚点列。若仅在初始集常量则保留；若全池常量则删除。
    force_drop_cols: 直接丢弃的列.
    extra_drop_cols: 额外丢弃的列 (会与 force_drop_cols 合并).
    output_dir: 若提供, 写入 per-target 选定特征 CSV 及报告 JSON.
    report_name: JSON 报告文件名 (不含路径), 默认 ``feature_selection_report.json``.

    Returns
    -------
    dict 包含选定特征、筛选范围、常量/相关性决策和 feature-group 审计。
    """
    data_path = Path(data_path)

    if force_keep_cols is None:
        force_keep_cols = []
    if force_drop_cols is None:
        force_drop_cols = list(DEFAULT_FORCE_DROP_COLS)
    if extra_drop_cols:
        force_drop_cols = list(force_drop_cols) + list(extra_drop_cols)

    df = pd.read_csv(data_path)
    x, y = _prepare_xy(df, targets, DEFAULT_DROP_COLS, force_drop_cols)

    force_keep_existing = [c for c in force_keep_cols if c in x.columns]
    x_screen, screening_audit = _load_unsupervised_screening_matrix(
        data_path,
        x,
    )
    x_var, constant_audit = _unsupervised_constant_filter(
        x,
        x_screen,
        force_keep_existing,
        screening_scope=screening_audit["scope"],
    )
    force_keep_all_nan = [
        c
        for c in force_keep_existing
        if c in constant_audit["all_missing_dropped"]
    ]
    force_keep_recovered_after_variance = [
        c
        for c in force_keep_existing
        if c in constant_audit[
            "force_kept_initial_constants_without_pool_reference"
        ]
    ]
    force_keep_usable = [c for c in force_keep_existing if c in x_var.columns]
    x_corr, redundancy_decisions, corr_threshold, corr_min_observations, groups = (
        _unsupervised_redundancy_filter(
            x_var,
            x_screen.loc[:, x_var.columns],
            force_keep_usable,
            constant_audit["feature_diagnostics"],
        )
    )
    dropped_by_corr = [
        item["dropped"] for item in redundancy_decisions
    ]
    force_keep_in_corr = [
        c for c in force_keep_usable if c in x_corr.columns
    ]
    selected_features = sorted(x_corr.columns)
    groups_all = {
        c: _feature_group(c, force_keep_existing)
        for c in x.columns
    }
    group_columns_before = _group_audit(x.columns, groups_all)
    group_columns_after = _group_audit(selected_features, groups)

    per_target_report = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        if output_dir is not None:
            out_df = pd.concat(
                [x_corr[selected_features], y[[target]]],
                axis=1,
            )
            out_csv = output_dir / f"features_selected_{target.lower()}.csv"
            # The CALM runner may create fit directories concurrently or
            # resume after an interrupted run.  Reassert the immediate parent
            # here so the final artifact write cannot depend on an earlier
            # directory creation surviving that orchestration.
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_csv, index=False)
            print(
                f"[{target}] saved -> {out_csv}  "
                f"({out_df.shape[0]} rows, {out_df.shape[1]} cols)"
            )

        per_target_report[target] = {
            "n_selected": len(selected_features),
            "selected_features": selected_features,
            "selection_is_target_independent": True,
            "selected_via_unsupervised_screening": selected_features,
            # Compatibility fields are intentionally empty: no permutation
            # ranking is used by the publication path.
            "selected_via_permutation": [],
            "selected_via_force_keep_only": sorted(force_keep_in_corr),
            "stability_frequency": {},
            "perm_importance_oof_mean_top": {},
            "perm_importance_oof_std_top": {},
            "feature_groups": group_columns_after,
            **({"saved_csv": str(out_csv)} if output_dir is not None else {}),
        }

    report = {
        "input_shape": {
            "n_samples": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
        },
        "x_before_filter": int(x.shape[1]),
        "x_after_variance_filter": int(x_var.shape[1]),
        "x_after_constant_filter": int(x_var.shape[1]),
        "x_after_corr_filter": int(x_corr.shape[1]),
        "selection_method": UNSUPERVISED_SELECTION_METHOD,
        "scoring_method": "none_target_independent",
        "feature_selection_policy_version": FEATURE_SELECTION_POLICY_VERSION,
        "supervised_selector_used": False,
        "target_values_used_for_selection": False,
        # Retain the legacy report keys for readers of older artifacts.  They
        # describe the optional exploratory helper and are explicitly not
        # active in this selection run.
        "supervised_selector_config_status": (
            "retained_for_compatibility_not_used"
        ),
        "perm_n_repeats": PERM_N_REPEATS,
        "kfold_splits": KFOLD_SPLITS,
        "stability_topk_per_target": TOP_K_PER_TARGET,
        "stability_frequency_threshold": IMPORTANCE_FREQ_THRESHOLD,
        "variance_threshold": VARIANCE_THRESHOLD,
        "screening_matrix": screening_audit,
        "constant_filter": {
            "test": "finite span <= 1e-12 + 1e-10 * max(1, abs(x))",
            **constant_audit,
        },
        "configured_corr_threshold": CORR_THRESHOLD,
        "effective_corr_threshold": corr_threshold,
        "corr_threshold": corr_threshold,
        "corr_min_pairwise_observations": corr_min_observations,
        "target_columns": targets,
        "force_drop_columns": force_drop_cols,
        "force_keep_raw_columns": force_keep_existing,
        "force_keep_all_nan_dropped": force_keep_all_nan,
        "force_keep_recovered_after_variance": (
            force_keep_recovered_after_variance
        ),
        "force_keep_raw_columns_after_corr": force_keep_in_corr,
        "dropped_by_correlation": dropped_by_corr,
        "redundancy_decisions": redundancy_decisions,
        "feature_groups_before_filter": group_columns_before,
        "feature_group_counts_before_filter": {
            group: len(columns)
            for group, columns in group_columns_before.items()
        },
        "feature_groups_after_filter": group_columns_after,
        "feature_group_counts_after_filter": {
            group: len(columns)
            for group, columns in group_columns_after.items()
        },
        "per_target": per_target_report,
    }

    if output_dir is not None:
        rpt_name = report_name or "feature_selection_report.json"
        report_path = Path(output_dir) / rpt_name
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {report_path}")

    return report


# ── CLI ───────────────────────────────────────────────────────────

def main():
    """使用默认参数 (2xxx_mn, Strength+TC) 运行特征筛选."""
    select_features(
        data_path=DEFAULT_DATA_PATH,
        targets=DEFAULT_TARGETS,
        force_keep_cols=DEFAULT_FORCE_KEEP_COLS,
        output_dir=DEFAULT_OUTPUT_DIR,
        report_name="2xxx_mn_feature_selection_report.json",
    )


if __name__ == "__main__":
    main()
