"""Leakage-audited data and feature representations for CALM.

The experiment exposes three explicit representations:

* ``raw``: reported composition plus deterministically encoded process inputs;
* ``physics``: ``raw`` plus atom-correct Magpie, configurational entropy, and
  radius-mismatch descriptors;
* ``physics_llm``: ``physics`` plus validated algebraic candidates proposed by
  the optional feature-construction skill.

The publication default is ``raw`` so acquisition-policy comparisons share
one deterministic, target-independent representation. Pool covariates may be
used for constant/redundancy screening; unqueried response values may not.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from common.llm.config import get_api_base_url, get_llm_model, validate_api_base_url

COMPOSITION_UNIT_NORMALIZATION_VERSION = "fraction_to_percent_v2"
ATOMIC_DESCRIPTOR_COMPOSITION_VERSION = (
    "explicit_basis_reported_components_missing_masks_v2"
)
COMPLETE_COMPOSITION_POLICY = "complete_composition_required_v1"
REPORTED_COMPOSITION_POLICY = (
    "preserve_unknown_reported_components_with_missing_indicators_v1"
)
COMPOSITION_MISSING_POLICIES = frozenset(
    {COMPLETE_COMPOSITION_POLICY, REPORTED_COMPOSITION_POLICY}
)
MODEL_READY_DATA_GUARD_VERSION = (
    "manifest_hash_roles_candidate_keys_unit_missing_closure_v3"
)
FEATURE_REPRESENTATION_VERSION = "raw_physics_physics_llm_v1"
FEATURE_REPRESENTATIONS = frozenset({"raw", "physics", "physics_llm"})
COLUMN_ROLE_SCHEMA_VERSION = "all_property_targets_v2"
COLUMN_ROLE_PROMPT_VERSION = "four_roles_all_properties_hidden_v2"
LLM_FEATURE_LEAKAGE_GUARD_VERSION = "forbid_all_property_columns_v2"
LLM_FEATURE_MATERIAL_PROFILE_VERSION = (
    "registered_material_unit_source_missing_null_contract_v4"
)
PROCESS_FEATURE_ENCODING_VERSION = (
    "numeric_missing_masks_plus_deterministic_onehot_missing_category_v2"
)
RAW_INPUT_FORCE_KEEP_VERSION = "elements_missing_masks_plus_encoded_process_v2"
INITIAL_FEATURE_FIT_VERSION = (
    "initial_labels_visible_pool_x_unsupervised_atomic_representation_v12"
)


def _llm_feature_cache_name(
    unit_normalized: bool,
    composition_unit: str = "wt.%",
) -> str:
    """Return a short, policy-keyed filename safe under Windows MAX_PATH."""
    if composition_unit not in {"wt.%", "at.%"}:
        raise ValueError(
            "composition_unit must be exactly 'wt.%' or 'at.%', got "
            f"{composition_unit!r}"
        )
    policy = {
        "leakage_guard": LLM_FEATURE_LEAKAGE_GUARD_VERSION,
        "material_profile": LLM_FEATURE_MATERIAL_PROFILE_VERSION,
        "composition_units": (
            COMPOSITION_UNIT_NORMALIZATION_VERSION
            if unit_normalized
            else "not_normalized"
        ),
        "composition_basis": composition_unit,
    }
    raw = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    policy_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"_llmf_{policy_id}.pkl"


def _formula_uses_any_column(formula: str, columns: list[str] | set[str]) -> bool:
    """Return True when a formula references any response/property column."""
    forbidden = {str(c).casefold() for c in columns}
    try:
        tree = ast.parse(str(formula).strip(), mode="eval")
    except SyntaxError:
        # Formula validation elsewhere rejects invalid expressions.  Treat an
        # unparsable cached formula conservatively when a forbidden name occurs.
        text = str(formula)
        return any(col in text.casefold() for col in forbidden)
    tokens = {
        str(node.id).casefold()
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    tokens.update(
        str(node.value).casefold()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return bool(tokens & forbidden)


def assert_no_target_leakage(
    feature_cols: list[str] | pd.Index,
    target_cols: list[str] | set[str],
    *,
    context: str = "feature matrix",
) -> None:
    """Fail fast if a measured response/property is exposed as a model input."""
    targets_by_lower = {str(c).casefold(): str(c) for c in target_cols}
    leaked = [
        str(c)
        for c in feature_cols
        if str(c).casefold() in targets_by_lower
    ]
    if leaked:
        raise RuntimeError(
            f"Target leakage in {context}: response/property columns used as inputs: "
            f"{sorted(set(leaked))}"
        )


def _column_schema_fingerprint(df: pd.DataFrame) -> str:
    """Stable cache key for the ordered column schema (names + dtypes)."""
    payload = [
        {"name": str(col), "dtype": str(df[col].dtype)}
        for col in df.columns
    ]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def dataframe_fingerprint(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> str:
    """Hash ordered rows and visible values for cache/provenance validation."""
    selected = [str(c) for c in (columns if columns is not None else df.columns)]
    view = df.loc[:, selected].copy()
    for col in view.columns:
        if (
            pd.api.types.is_object_dtype(view[col])
            or pd.api.types.is_string_dtype(view[col])
            or isinstance(view[col].dtype, pd.CategoricalDtype)
        ):
            view[col] = view[col].astype("string").fillna("<NA>")
    row_hashes = pd.util.hash_pandas_object(view, index=True).to_numpy(dtype=np.uint64)
    header = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(header.encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def read_dataset_table(data_path: str | Path) -> pd.DataFrame:
    """Read a CALM source table without changing its visible values.

    Existing benchmark sources remain XLSX. Newly published model-ready alloy
    tables may use CSV so that cleaned values can be promoted without another
    spreadsheet serialization step. Both formats use a single header row and
    no exported index column.
    """
    path = Path(data_path)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(
        f"Unsupported CALM dataset format {path.suffix!r}: {path}. "
        "Expected CSV, XLSX, XLSM, or XLS."
    )


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict:
    """Load a JSON sidecar and reject malformed/non-object payloads."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _canonicalize_candidate_keys(
    df: pd.DataFrame,
    candidate_key_columns: list[str],
    *,
    numeric_rounding_decimals: int,
) -> pd.DataFrame:
    """Canonicalize visible inputs using the publication deduplication policy."""
    canonical = df.loc[:, candidate_key_columns].copy()
    for col in candidate_key_columns:
        raw = canonical[col]
        text = raw.astype("string").str.strip()
        semantic_non_missing = text.notna() & text.ne("")
        numeric = pd.to_numeric(
            text.where(semantic_non_missing),
            errors="coerce",
        )
        if (
            pd.api.types.is_numeric_dtype(raw)
            or int(numeric.notna().sum()) == int(semantic_non_missing.sum())
        ):
            numeric = numeric.astype(np.float64).round(numeric_rounding_decimals)
            numeric.loc[numeric == 0.0] = 0.0
            canonical[col] = numeric
        else:
            canonical[col] = text.mask(text.eq(""), pd.NA)
    return canonical


def validate_model_ready_dataset(
    data_path: str | Path,
    df: pd.DataFrame,
    *,
    target_col: str,
    element_cols: list[str],
    process_cols: list[str] | None,
    expected_target_n: int,
    expected_composition_unit: str | None = None,
    expected_composition_missing_policy: str = COMPLETE_COMPOSITION_POLICY,
    require_dedup_manifest: bool = True,
    require_roles_sidecar: bool = True,
) -> dict[str, object]:
    """Fail fast when a model-ready table drifts from its publication sidecars.

    Call this immediately after :func:`read_dataset_table`, before filtering
    target-missing rows or normalizing composition values.  The sibling
    ``.dedup.json`` verifies the processed-file hash, row count, target-valid
    count, and exact candidate-key definition.  The sibling ``.columns.json``
    verifies the ordered schema and ordered table-value fingerprints.

    Sidecars may be optional for ephemeral unit-test data.  If an optional
    sidecar is present, it is still validated rather than silently ignored.
    Candidate uniqueness and ``expected_target_n`` are always checked.
    """
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model-ready data file does not exist: {path}")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if (
        not isinstance(expected_target_n, (int, np.integer))
        or isinstance(expected_target_n, (bool, np.bool_))
        or int(expected_target_n) < 0
    ):
        raise ValueError("expected_target_n must be a non-negative integer")
    expected_target_n = int(expected_target_n)
    if (
        expected_composition_unit is not None
        and expected_composition_unit not in {"wt.%", "at.%"}
    ):
        raise ValueError(
            "expected_composition_unit must be exactly 'wt.%' or 'at.%'"
        )
    if expected_composition_missing_policy not in COMPOSITION_MISSING_POLICIES:
        raise ValueError(
            "expected_composition_missing_policy must be one of "
            f"{sorted(COMPOSITION_MISSING_POLICIES)}"
        )

    elements = [str(col) for col in (element_cols or [])]
    processes = [str(col) for col in (process_cols or [])]
    candidate_key_columns = [*elements, *processes]
    if not elements:
        raise ValueError("element_cols must contain at least one composition column")
    if len(candidate_key_columns) != len(set(candidate_key_columns)):
        raise ValueError(
            "element_cols and process_cols must define disjoint, duplicate-free "
            "candidate keys"
        )
    if target_col in candidate_key_columns:
        raise ValueError(
            f"Target column {target_col!r} must not be a candidate key/input"
        )
    required_columns = [*candidate_key_columns, target_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Model-ready table is missing required columns: {missing_columns}"
        )
    composition_raw = df.loc[:, elements]
    composition_numeric = composition_raw.apply(
        pd.to_numeric,
        errors="coerce",
    )
    semantic_non_missing = composition_raw.apply(
        lambda series: series.astype("string").str.strip().notna()
        & series.astype("string").str.strip().ne("")
    )
    non_numeric = semantic_non_missing & composition_numeric.isna()
    if non_numeric.any().any():
        locations = [
            f"{index}:{column}"
            for index, column in zip(*np.where(non_numeric.to_numpy()))
        ][:10]
        raise ValueError(
            "Model-ready element compositions contain non-numeric reported "
            f"values at positions {locations}"
        )
    composition_array = composition_numeric.to_numpy(dtype=np.float64)
    finite_or_missing = np.isfinite(composition_array) | np.isnan(composition_array)
    if not finite_or_missing.all():
        raise ValueError(
            "Model-ready reported element compositions must be finite"
        )
    missing_composition = np.isnan(composition_array)
    if (
        expected_composition_missing_policy == COMPLETE_COMPOSITION_POLICY
        and missing_composition.any()
    ):
        raise ValueError(
            "Model-ready element compositions must all be reported under the "
            "complete-composition policy"
        )
    if np.any(composition_array[np.isfinite(composition_array)] < 0.0):
        raise ValueError(
            "Model-ready element compositions must be non-negative"
        )
    composition_sums = composition_numeric.sum(axis=1, min_count=1)
    if (composition_sums <= 0.0).any():
        raise ValueError(
            "Every model-ready candidate must have positive total composition"
        )

    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    target_values = target_numeric.to_numpy(dtype=np.float64)
    actual_target_n = int(np.isfinite(target_values).sum())
    if actual_target_n != expected_target_n:
        raise ValueError(
            f"Target-valid row count mismatch for {target_col!r}: "
            f"table={actual_target_n}, expected={expected_target_n}"
        )

    dedup_path = path.with_suffix(".dedup.json")
    dedup_manifest: dict | None = None
    if dedup_path.exists():
        dedup_manifest = _load_json_object(
            dedup_path,
            label="deduplication manifest",
        )
    elif require_dedup_manifest:
        raise FileNotFoundError(
            f"Required deduplication manifest is missing: {dedup_path}"
        )

    processed_sha256 = _sha256_file(path)
    numeric_rounding_decimals = 12
    if dedup_manifest is not None:
        manifest_sha256 = str(dedup_manifest.get("processed_sha256", "")).strip()
        if not manifest_sha256:
            raise ValueError(
                f"Deduplication manifest lacks processed_sha256: {dedup_path}"
            )
        if manifest_sha256.casefold() != processed_sha256.casefold():
            raise ValueError(
                f"Processed SHA-256 mismatch for {path.name}: "
                f"manifest={manifest_sha256}, actual={processed_sha256}"
            )

        try:
            manifest_unique_rows = int(dedup_manifest["n_unique_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Deduplication manifest has invalid n_unique_rows: {dedup_path}"
            ) from exc
        if manifest_unique_rows != len(df):
            raise ValueError(
                f"Unique-row count mismatch for {path.name}: "
                f"manifest={manifest_unique_rows}, table={len(df)}"
            )

        manifest_keys = dedup_manifest.get("candidate_key_columns")
        if not isinstance(manifest_keys, list):
            raise ValueError(
                f"Deduplication manifest lacks candidate_key_columns: {dedup_path}"
            )
        manifest_keys = [str(col) for col in manifest_keys]
        if manifest_keys != candidate_key_columns:
            raise ValueError(
                f"Candidate-key role mismatch for {path.name}: "
                f"manifest={manifest_keys}, roles={candidate_key_columns}"
            )

        manifest_target_n = None
        manifest_target_field = None
        for field in ("target_valid_rows", "task_target_valid_rows"):
            counts = dedup_manifest.get(field)
            if isinstance(counts, dict) and target_col in counts:
                try:
                    manifest_target_n = int(counts[target_col])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Deduplication manifest has invalid {field}"
                        f"[{target_col!r}]: {dedup_path}"
                    ) from exc
                manifest_target_field = field
                break
        if manifest_target_n is None:
            raise ValueError(
                f"Deduplication manifest lacks a target-valid count for "
                f"{target_col!r}: {dedup_path}"
            )
        if manifest_target_n != expected_target_n:
            raise ValueError(
                f"Manifest target-valid count mismatch for {target_col!r}: "
                f"{manifest_target_field}={manifest_target_n}, "
                f"expected={expected_target_n}"
            )

        try:
            numeric_rounding_decimals = int(
                dedup_manifest.get("numeric_key_rounding_decimals", 12)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Deduplication manifest has invalid "
                f"numeric_key_rounding_decimals: {dedup_path}"
            ) from exc
        if numeric_rounding_decimals < 0:
            raise ValueError(
                f"numeric_key_rounding_decimals must be non-negative: {dedup_path}"
            )

    canonical_keys = _canonicalize_candidate_keys(
        df,
        candidate_key_columns,
        numeric_rounding_decimals=numeric_rounding_decimals,
    )
    duplicate_mask = canonical_keys.duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_indices = [
            int(index) if isinstance(index, (int, np.integer)) else str(index)
            for index in canonical_keys.index[duplicate_mask][:10]
        ]
        raise ValueError(
            f"Duplicate model-ready candidates remain after "
            f"{numeric_rounding_decimals}-decimal key canonicalization; "
            f"row indices={duplicate_indices}"
        )

    roles_path = path.with_suffix(".columns.json")
    roles_sidecar: dict | None = None
    if roles_path.exists():
        roles_sidecar = _load_json_object(
            roles_path,
            label="column-role sidecar",
        )
    elif require_roles_sidecar:
        raise FileNotFoundError(
            f"Required column-role sidecar is missing: {roles_path}"
        )

    if roles_sidecar is not None:
        sidecar_elements = [
            str(col) for col in roles_sidecar.get("element_cols", [])
        ]
        sidecar_processes = [
            str(col) for col in roles_sidecar.get("process_cols", [])
        ]
        if sidecar_elements != elements or sidecar_processes != processes:
            raise ValueError(
                f"Column-role sidecar input roles differ from requested roles: "
                f"{roles_path}"
            )
        sidecar_targets = roles_sidecar.get("target_cols")
        if not isinstance(sidecar_targets, list):
            raise ValueError(
                f"Column-role sidecar target_cols must be a list: {roles_path}"
            )
        if target_col not in sidecar_targets:
            raise ValueError(
                f"Column-role sidecar does not classify {target_col!r} as a "
                f"target/property: {roles_path}"
            )

        expected_schema_fingerprint = _column_schema_fingerprint(df)
        if (
            roles_sidecar.get("column_schema_fingerprint")
            != expected_schema_fingerprint
        ):
            raise ValueError(
                f"Column-role schema fingerprint mismatch for {path.name}"
            )
        expected_source_fingerprint = dataframe_fingerprint(df)
        if (
            roles_sidecar.get("source_order_fingerprint")
            != expected_source_fingerprint
        ):
            raise ValueError(
                f"Column-role source fingerprint mismatch for {path.name}"
            )

    closure_range: list[float] | None = None
    if expected_composition_unit is not None:
        if dedup_manifest is None or roles_sidecar is None:
            raise ValueError(
                "Composition-unit validation requires both deduplication and "
                "column-role sidecars"
            )
        manifest_unit = dedup_manifest.get("composition_unit")
        sidecar_unit = roles_sidecar.get("composition_unit")
        if manifest_unit != expected_composition_unit:
            raise ValueError(
                f"Deduplication manifest composition_unit mismatch for "
                f"{path.name}: manifest={manifest_unit!r}, "
                f"expected={expected_composition_unit!r}"
            )
        if sidecar_unit != expected_composition_unit:
            raise ValueError(
                f"Column-role sidecar composition_unit mismatch for "
                f"{path.name}: sidecar={sidecar_unit!r}, "
                f"expected={expected_composition_unit!r}"
            )
        manifest_missing_policy = dedup_manifest.get(
            "composition_missing_policy", COMPLETE_COMPOSITION_POLICY
        )
        sidecar_missing_policy = roles_sidecar.get(
            "composition_missing_policy", COMPLETE_COMPOSITION_POLICY
        )
        if (
            manifest_missing_policy != expected_composition_missing_policy
            or sidecar_missing_policy != expected_composition_missing_policy
        ):
            raise ValueError(
                f"Composition missingness policy mismatch for {path.name}: "
                f"manifest={manifest_missing_policy!r}, "
                f"sidecar={sidecar_missing_policy!r}, "
                f"expected={expected_composition_missing_policy!r}"
            )

        manifest_closure = dedup_manifest.get("composition_closure_audit")
        sidecar_closure = roles_sidecar.get("composition_closure_audit")
        if not isinstance(manifest_closure, dict) or not isinstance(
            sidecar_closure,
            dict,
        ):
            raise ValueError(
                f"Composition closure audit is missing for {path.name}"
            )
        manifest_range = manifest_closure.get("sum_range")
        sidecar_range = sidecar_closure.get("sum_range")
        if manifest_range is None or sidecar_range is None:
            if (
                manifest_range is not None
                or sidecar_range is not None
                or expected_composition_missing_policy != REPORTED_COMPOSITION_POLICY
                or manifest_closure.get("mode")
                != "reported_component_partial_sum_no_closure"
                or sidecar_closure.get("mode")
                != "reported_component_partial_sum_no_closure"
            ):
                raise ValueError(
                    f"Composition closure exemption is invalid for {path.name}"
                )
        else:
            if (
                not isinstance(manifest_range, list)
                or len(manifest_range) != 2
                or not isinstance(sidecar_range, list)
                or len(sidecar_range) != 2
            ):
                raise ValueError(
                    f"Composition closure audit has an invalid sum_range for "
                    f"{path.name}"
                )
            try:
                closure_range = [float(manifest_range[0]), float(manifest_range[1])]
                sidecar_range_numeric = [
                    float(sidecar_range[0]),
                    float(sidecar_range[1]),
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Composition closure range is non-numeric for {path.name}"
                ) from exc
            if (
                closure_range != sidecar_range_numeric
                or not 0.0 < closure_range[0] <= closure_range[1]
            ):
                raise ValueError(
                    f"Composition closure ranges disagree for {path.name}: "
                    f"manifest={closure_range}, sidecar={sidecar_range_numeric}"
                )
            within_closure = composition_sums.between(
                closure_range[0],
                closure_range[1],
                inclusive="both",
            )
            if not within_closure.all():
                bad = [
                    int(i) if isinstance(i, (int, np.integer)) else str(i)
                    for i in composition_sums.index[~within_closure][:10]
                ]
                raise ValueError(
                    f"Composition closure failure for {path.name}: expected sums "
                    f"within {closure_range}, row indices={bad}"
                )

    return {
        "guard_version": MODEL_READY_DATA_GUARD_VERSION,
        "status": "pass",
        "data_path": str(path),
        "processed_sha256": processed_sha256,
        "n_unique_rows": int(len(df)),
        "target_col": target_col,
        "target_valid_rows": actual_target_n,
        "composition_unit": expected_composition_unit,
        "composition_missing_policy": expected_composition_missing_policy,
        "composition_missing_cell_count": int(missing_composition.sum()),
        "composition_rows_with_missing": int(missing_composition.any(axis=1).sum()),
        "composition_sum_range": closure_range,
        "composition_sum_min": float(composition_sums.min()),
        "composition_sum_max": float(composition_sums.max()),
        "candidate_key_columns": candidate_key_columns,
        "numeric_key_rounding_decimals": numeric_rounding_decimals,
        "dedup_manifest": str(dedup_path) if dedup_manifest is not None else None,
        "roles_sidecar": str(roles_path) if roles_sidecar is not None else None,
    }


def _dedupe_known_columns(values, columns: list[str]) -> list[str]:
    known = set(columns)
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        col = str(value)
        if col in known and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _normalize_column_roles(
    result: dict,
    columns: list[str],
    *,
    target_col: str,
    known_target_cols: list[str] | None = None,
) -> dict[str, list[str]]:
    """Normalize LLM roles and force every known response into ``target_cols``."""
    known_targets = _dedupe_known_columns(
        [target_col, *(known_target_cols or [])],
        columns,
    )
    llm_targets = _dedupe_known_columns(result.get("target_cols", []), columns)
    target_cols = _dedupe_known_columns([*known_targets, *llm_targets], columns)
    target_set = set(target_cols)

    roles = {
        "target_cols": target_cols,
        "element_cols": _dedupe_known_columns(result.get("element_cols", []), columns),
        "process_cols": _dedupe_known_columns(result.get("process_cols", []), columns),
        "metadata_cols": _dedupe_known_columns(result.get("metadata_cols", []), columns),
    }

    # Priority is target > element > process > metadata.  This guarantees that
    # an LLM response cannot simultaneously expose a measured property as input.
    assigned = set(target_set)
    clean = {"target_cols": target_cols, "element_cols": [], "process_cols": [], "metadata_cols": []}
    for key in ("element_cols", "process_cols", "metadata_cols"):
        for col in roles[key]:
            if col not in assigned:
                clean[key].append(col)
                assigned.add(col)

    # Unknown/unassigned columns are excluded from modeling by default.
    clean["metadata_cols"].extend(col for col in columns if col not in assigned)
    return clean


def _validate_column_roles(
    roles: dict[str, list[str]],
    columns: list[str],
    *,
    context: str,
) -> None:
    """Reject syntactically valid but unusable LLM column classifications."""
    if not roles.get("element_cols"):
        raise ValueError(
            f"{context}: no element composition columns were identified"
        )

    role_names = ("target_cols", "element_cols", "process_cols", "metadata_cols")
    flattened = [col for role in role_names for col in roles.get(role, [])]
    counts = {col: flattened.count(col) for col in set(flattened)}
    duplicates = sorted(col for col, count in counts.items() if count > 1)
    missing = [col for col in columns if col not in flattened]
    unknown = [col for col in flattened if col not in columns]
    if duplicates or missing or unknown:
        raise ValueError(
            f"{context}: roles do not partition the source columns exactly "
            f"(duplicates={duplicates}, missing={missing}, unknown={unknown})"
        )


def _safe_input_roles(
    *,
    element_cols: list[str] | None,
    process_cols: list[str] | None,
    target_cols: list[str] | None,
    current_target: str,
) -> tuple[list[str], list[str], list[str]]:
    """Return deduplicated input roles with all responses removed."""
    targets = list(dict.fromkeys([current_target, *(target_cols or [])]))
    target_set = set(targets)
    elements = [
        c for c in dict.fromkeys(element_cols or [])
        if c not in target_set
    ]
    processes = [
        c for c in dict.fromkeys(process_cols or [])
        if c not in target_set and c not in elements
    ]
    assert_no_target_leakage(elements + processes, targets, context="column roles")
    return elements, processes, targets


def normalize_mixed_composition_units(
    df: pd.DataFrame,
    element_cols: list[str] | None,
    *,
    enabled: bool = False,
    label: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Normalize fraction-scale composition rows to percent scale.

    Enabled datasets may be entirely fraction-scale (row sum near 1) or may
    mix fraction-scale and percent-scale rows.  Every fraction-scale row is
    multiplied by 100; rows already summing near 100 are left unchanged.
    """
    elem_available = [c for c in (element_cols or []) if c in df.columns]
    meta = {
        "version": COMPOSITION_UNIT_NORMALIZATION_VERSION,
        "enabled": bool(enabled),
        "applied": False,
        "rows_scaled": 0,
        "rows_fraction_like": 0,
        "rows_percent_like": 0,
        "element_cols": elem_available,
        "cache_key": "disabled",
    }
    if not enabled or not elem_available:
        return df, meta

    out = df.copy()
    numeric = out[elem_available].apply(pd.to_numeric, errors="coerce")
    row_sum = numeric.fillna(0.0).sum(axis=1)
    fraction_like = row_sum.between(0.5, 1.5)
    percent_like = row_sum.between(80.0, 120.0)

    meta.update(
        {
            "rows_fraction_like": int(fraction_like.sum()),
            "rows_percent_like": int(percent_like.sum()),
            "cache_key": (
                f"{COMPOSITION_UNIT_NORMALIZATION_VERSION}:"
                f"frac={int(fraction_like.sum())}:"
                f"pct={int(percent_like.sum())}:"
                f"elements={','.join(elem_available)}"
            ),
        }
    )

    if not fraction_like.any():
        print(
            f"[data] Composition unit normalization enabled for {label or 'dataset'}, "
            "but no fraction-scale rows were detected"
        )
        return out, meta

    out.loc[:, elem_available] = numeric
    out.loc[fraction_like, elem_available] = (
        numeric.loc[fraction_like, elem_available] * 100.0
    )
    meta["applied"] = True
    meta["rows_scaled"] = int(fraction_like.sum())
    print(
        f"[data] Normalized composition units for {label or 'dataset'}: "
        f"scaled {meta['rows_scaled']} fraction-like rows to percent scale "
        f"({meta['rows_percent_like']} rows already percent-like; "
        f"mode={'mixed' if percent_like.any() else 'all-fraction'})"
    )
    return out, meta


# ── 通用特征构建 (base + Magpie) ──


def _encode_process_features(
    df: pd.DataFrame,
    process_cols: list[str] | None,
) -> pd.DataFrame:
    """Encode numeric processes directly and categorical processes one-hot.

    The full candidate pool is known in pool-based AL, so deriving the sorted
    category vocabulary from input-only columns does not expose any labels.
    Sorted categories also keep feature names stable across runs and machines.
    """
    encoded_columns: dict[str, pd.Series] = {}
    for col in process_cols or []:
        if col not in df.columns:
            continue
        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        non_missing = int(raw.notna().sum())
        if pd.api.types.is_numeric_dtype(raw) or int(numeric.notna().sum()) == non_missing:
            if int(numeric.notna().sum()) == 0:
                continue
            encoded_columns[col] = numeric
            if numeric.isna().any():
                encoded_columns[f"{col}__source_missing"] = numeric.isna().astype(float)
            continue

        values = raw.astype("string").fillna("__MISSING__")
        categories = sorted(str(value) for value in values.unique())
        for category in categories:
            encoded_columns[f"{col}={category}"] = (values == category).astype(float)
    return pd.DataFrame(encoded_columns, index=df.index)


def _force_keep_input_feature_columns(
    df: pd.DataFrame,
    features: pd.DataFrame,
    element_cols: list[str] | None,
    process_cols: list[str] | None,
) -> list[str]:
    """Return raw model inputs that must survive supervised feature selection.

    Elements keep their original composition columns. Numeric process columns
    keep their original names, while categorical process columns keep every
    deterministic one-hot column produced by :func:`_encode_process_features`.
    Targets and metadata are never supplied through either input-role list.
    """
    encoded_process_cols = list(_encode_process_features(df, process_cols).columns)
    missing_mask_cols = [
        f"{col}__source_missing"
        for col in (element_cols or [])
        if f"{col}__source_missing" in features.columns
    ]
    requested = [*(element_cols or []), *missing_mask_cols, *encoded_process_cols]
    return [
        col
        for col in dict.fromkeys(requested)
        if col in features.columns
    ]


def _build_atomic_compositions(
    comp_df: pd.DataFrame,
    element_cols: list[str],
    composition_unit: str,
    *,
    allow_missing: bool = False,
) -> pd.Series:
    """Build normalized atomic compositions for atom-based descriptors.

    Raw composition columns are never overwritten.  For ``wt.%`` inputs, each
    positive mass amount is divided by the element's atomic mass before
    normalization.  ``at.%`` inputs are normalized directly, so both 0--1 and
    0--100 atomic scales are accepted.

    Missing composition cells fail closed by default.  For an explicitly
    registered source-missingness policy, ``allow_missing=True`` computes
    descriptors from reported components only; the caller must also expose
    per-element source-missing indicators.  Missing values are never converted
    to zero or filled to a balance composition.
    """
    if composition_unit not in {"wt.%", "at.%"}:
        raise ValueError(
            "composition_unit must be exactly 'wt.%' or 'at.%', got "
            f"{composition_unit!r}"
        )

    from pymatgen.core.composition import Composition
    from pymatgen.core.periodic_table import Element

    elem_cols = [c for c in element_cols if c in comp_df.columns]
    if not elem_cols:
        raise ValueError("No element columns are available for composition conversion")

    atomic_masses: dict[str, float] = {}
    if composition_unit == "wt.%":
        for element in elem_cols:
            try:
                mass = float(Element(element).atomic_mass)
            except Exception as exc:
                raise ValueError(
                    f"Invalid element column {element!r} for atomic-mass conversion"
                ) from exc
            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError(
                    f"Invalid atomic mass for element column {element!r}: {mass}"
                )
            atomic_masses[element] = mass

    def _row_to_atomic_composition(row: pd.Series) -> Composition:
        atomic_amounts: dict[str, float] = {}
        for element in elem_cols:
            raw_value = row[element]
            if pd.isna(raw_value):
                if allow_missing:
                    continue
                raise ValueError(
                    f"Missing composition value at row {row.name!r}, "
                    f"element {element!r}"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric composition value at row {row.name!r}, "
                    f"element {element!r}: {raw_value!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite composition value at row {row.name!r}, "
                    f"element {element!r}: {raw_value!r}"
                )
            if value < 0.0:
                raise ValueError(
                    f"Negative composition value at row {row.name!r}, "
                    f"element {element!r}: {value}"
                )
            if value == 0.0:
                continue
            atomic_amounts[element] = (
                value / atomic_masses[element]
                if composition_unit == "wt.%"
                else value
            )

        total = float(sum(atomic_amounts.values()))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"Zero total composition at row {row.name!r}; "
                "cannot compute atom-based descriptors"
            )
        return Composition(atomic_amounts).fractional_composition

    return comp_df[elem_cols].apply(_row_to_atomic_composition, axis=1)


def build_generic_features(
    comp_df: pd.DataFrame,
    df: pd.DataFrame,
    element_cols: list[str],
    process_cols: list[str] | None = None,
    composition_unit: str = "wt.%",
    include_atomic_descriptors: bool = True,
    composition_missing_policy: str = COMPLETE_COMPOSITION_POLICY,
) -> pd.DataFrame:
    """构建 base + Magpie 特征。不经过 SimpleAlloyFeatureConstructor,
       因此从源头就不产生任何手工特征 (ratio/PSI/RII 等).

    跨体系通用: 接受元素列和工艺列作为参数, Magpie 自动适配任意元素。

    Parameters
    ----------
    comp_df : 含有元素列的 DataFrame
    df : 原始 DataFrame (用于提取工艺列)
    element_cols : 元素列名列表
    process_cols : 工艺列名列表 (可选)

    Returns
    -------
    pd.DataFrame: base 元素列 + 工艺列 + Magpie + comp_entropy + radius_mismatch.
    Raw element columns remain on ``composition_unit``; atom-based descriptors
    always use normalized atomic fractions. Set
    ``include_atomic_descriptors=False`` for the raw representation, which
    returns only composition columns plus encoded process inputs.
    """
    if process_cols is None:
        process_cols = []
    if not isinstance(include_atomic_descriptors, (bool, np.bool_)):
        raise TypeError("include_atomic_descriptors must be a boolean")
    if composition_missing_policy not in COMPOSITION_MISSING_POLICIES:
        raise ValueError(
            "composition_missing_policy must be one of "
            f"{sorted(COMPOSITION_MISSING_POLICIES)}"
        )
    allow_missing_composition = (
        composition_missing_policy == REPORTED_COMPOSITION_POLICY
    )

    feats = pd.DataFrame(index=comp_df.index)

    # 1) base 元素列
    elem_cols = [c for c in element_cols if c in comp_df.columns]
    if not elem_cols:
        raise ValueError(
            "No element composition columns are available for generic feature "
            "construction. Check the dataset .columns.json role cache."
        )
    for c in elem_cols:
        feats[c] = pd.to_numeric(comp_df[c], errors="coerce")
        if allow_missing_composition:
            feats[f"{c}__source_missing"] = feats[c].isna().astype(np.float64)

    # 2) 工艺列: numeric passthrough + deterministic one-hot categoricals
    process_features = _encode_process_features(df, process_cols)
    feats = pd.concat([feats, process_features], axis=1)

    if not include_atomic_descriptors:
        numeric_cols = feats.select_dtypes(include=[np.number]).columns
        all_missing_numeric = [col for col in numeric_cols if feats[col].isna().all()]
        if all_missing_numeric:
            feats = feats.drop(columns=all_missing_numeric)
            numeric_cols = feats.select_dtypes(include=[np.number]).columns
        feats[numeric_cols] = feats[numeric_cols].fillna(
            feats[numeric_cols].median()
        )
        print(
            f"[data] Built {len(feats.columns)} features: "
            f"base({len(elem_cols)}) + "
            f"process({len(process_features.columns)}) + "
            "atomic_descriptors(disabled)"
        )
        return feats

    from matminer.featurizers.composition import ElementProperty
    from matminer.utils.data import MagpieData
    from pymatgen.core.periodic_table import Element

    # 3) Magpie 特征.  Pymatgen ``Composition`` amounts are atomic, so wt.%
    # must be converted through atomic masses before any atom-based descriptor.
    compositions = _build_atomic_compositions(
        comp_df,
        elem_cols,
        composition_unit,
        allow_missing=allow_missing_composition,
    )
    ep = ElementProperty.from_preset(preset_name="magpie")
    # Matminer otherwise inherits a machine-dependent process count.  A fixed
    # single worker avoids Windows spawn recursion and keeps feature generation
    # deterministic across formal runs.
    ep.set_n_jobs(1)
    magpie_df = ep.featurize_dataframe(
        pd.DataFrame({"composition": compositions}),
        "composition", ignore_errors=True, pbar=False,
    )
    if "composition" in magpie_df.columns:
        magpie_df = magpie_df.drop(columns=["composition"])
    magpie_df.index = comp_df.index
    feats = pd.concat([feats, magpie_df], axis=1)

    # 4) 混合熵
    def _entropy(comp) -> float:
        vals = np.array(list(comp.get_el_amt_dict().values()), dtype=float)
        if vals.size == 0 or vals.sum() <= 0:
            return np.nan
        frac = vals / vals.sum()
        frac = frac[frac > 0]
        return float(-np.sum(frac * np.log(frac)))

    feats["comp_entropy"] = compositions.apply(_entropy)

    # 5) 原子半径失配
    md = MagpieData(impute_nan=True)

    def _radius_mismatch(comp) -> float:
        d = comp.get_el_amt_dict()
        if not d:
            return np.nan
        total = sum(d.values())
        radii, fracs = [], []
        for ele, amt in d.items():
            try:
                r = md.get_elemental_property(Element(ele), "AtomicRadius")
            except Exception:
                r = None
            if r is not None and np.isfinite(float(r)):
                radii.append(float(r))
                fracs.append(float(amt) / total)
        if not radii:
            return np.nan
        fracs = np.asarray(fracs, dtype=float)
        frac_total = float(fracs.sum())
        if not np.isfinite(frac_total) or frac_total <= 0.0:
            return np.nan
        fracs = fracs / frac_total
        r_mean = float(np.dot(fracs, radii))
        if abs(r_mean) < 1e-12:
            return np.nan
        return float(np.sqrt(np.dot(fracs, [(1.0 - r / r_mean) ** 2 for r in radii])))

    feats["radius_mismatch"] = compositions.apply(_radius_mismatch)

    # 6) 缺失值填充
    numeric_cols = feats.select_dtypes(include=[np.number]).columns
    all_missing_numeric = [col for col in numeric_cols if feats[col].isna().all()]
    if all_missing_numeric:
        feats = feats.drop(columns=all_missing_numeric)
        numeric_cols = feats.select_dtypes(include=[np.number]).columns
    feats[numeric_cols] = feats[numeric_cols].fillna(feats[numeric_cols].median())

    n_magpie = sum(1 for c in feats.columns if c.startswith("MagpieData "))
    descriptor_basis = (
        "wt.%->atomic_fraction"
        if composition_unit == "wt.%"
        else "at.%->atomic_fraction"
    )
    print(f"[data] Built {len(feats.columns)} features: "
          f"base({len(elem_cols)}) + process({len(process_features.columns)}) + "
          f"Magpie({n_magpie}) + entropy + mismatch "
          f"[atomic_basis={descriptor_basis}]")
    return feats


# ── LLM 列分类 ──────────────────────────────────────────────────────

_COLUMN_DETECT_PROMPT = """\
You are a materials informatics data curator. Given the column names and sample \
values from an alloy dataset Excel file, classify each column into exactly one of:

- **element_cols**: chemical element composition columns (wt.% or at.%). These are \
  elements like Cu, Mg, Al, Fe, C, Mn, etc. Usually 1-3 letter symbols starting \
  with an uppercase letter. Look at both the column name AND sample values — \
  element columns typically contain non-negative values on either a fraction
  scale or a 0-100 percent scale; do not infer the physical basis from the sum.

- **process_cols**: processing parameters (temperature, time, strain rate, \
  reduction ratio, extrusion speed, aging type, condition flags, \
  gauge length, thickness, test conditions, etc.). A measured material property \
  or response must NEVER be classified as a process/input column.

- **target_cols**: every measured response, predicted property, optimization \
  objective, or auxiliary material property. Examples include strength, yield or \
  tensile response, elongation, hardness, conductivity, modulus/MOE, fatigue, \
  corrosion, and any other measured output. Known target columns that must be in \
  this list: {known_target_cols}.

- **metadata_cols**: identifiers, labels, references, comments, alloy names, \
  DOIs, R2/R3 scores, captions, formula strings — not used for modeling.

Return a JSON object with keys "element_cols", "process_cols", "target_cols", \
and "metadata_cols", each mapping to a list of exact column names. Every column \
must appear exactly once across the four lists.

Column details (name + first 5 non-null values):
{column_samples}
"""


def _llm_detect_column_roles(
    df: pd.DataFrame,
    target_col: str,
    xlsx_path: str | Path | None = None,
    known_target_cols: list[str] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, list[str]]:
    """Classify all columns, explicitly isolating every measured response.

    Cache reuse requires the leakage-safe role schema and an exact source-column
    fingerprint.  Legacy ``.columns.json`` files are intentionally invalidated.
    """
    model = get_llm_model() if model is None else model
    base_url = validate_api_base_url(
        get_api_base_url() if base_url is None else base_url
    )
    columns = [str(c) for c in df.columns]
    schema_fingerprint = _column_schema_fingerprint(df)
    source_order_fingerprint = dataframe_fingerprint(df)
    forced_targets = _dedupe_known_columns(
        [target_col, *(known_target_cols or [])],
        columns,
    )
    cache_path = None
    if xlsx_path is not None:
        cache_path = Path(xlsx_path).with_suffix(".columns.json")
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached = {}
            curated_role_registry = (
                cached.get("role_source")
                in {
                    "curated_physical_dedup_registry_v1",
                    "official_v24_curated_registry_v1",
                    "audited_external_extension_registry_v1",
                }
            )
            cache_ok = (
                (
                    curated_role_registry
                    or (
                        cached.get("schema_version")
                        == COLUMN_ROLE_SCHEMA_VERSION
                        and cached.get("prompt_version")
                        == COLUMN_ROLE_PROMPT_VERSION
                        and cached.get("model") == model
                        and cached.get("api_base_url") == base_url
                    )
                )
                and cached.get("column_schema_fingerprint") == schema_fingerprint
                and cached.get("source_order_fingerprint") == source_order_fingerprint
                and set(forced_targets).issubset(set(cached.get("target_cols", [])))
            )
            if cache_ok:
                roles = _normalize_column_roles(
                    cached,
                    columns,
                    target_col=target_col,
                    known_target_cols=forced_targets,
                )
                try:
                    _validate_column_roles(
                        roles,
                        columns,
                        context=f"cached column roles in {cache_path.name}",
                    )
                except ValueError as exc:
                    print(f"[detect] Invalidating unusable column roles: {exc}")
                else:
                    role_source = (
                        "audited curated registry"
                        if curated_role_registry
                        else "leakage-safe LLM cache"
                    )
                    print(
                        f"[detect] Loaded {role_source} from "
                        f"{cache_path.name}"
                    )
                    return roles
            print(f"[detect] Invalidating legacy/stale column roles in {cache_path.name}")

    # 每列取前 5 个非空样本值
    sample_lines = []
    for col in df.columns:
        if str(col) in forced_targets:
            vals = ["<known measured response values hidden>"]
        else:
            vals = df[col].dropna().head(5).tolist()
        sample_lines.append(f"  {col}: {vals}")
    column_samples = "\n".join(sample_lines)

    prompt = _COLUMN_DETECT_PROMPT.format(
        known_target_cols=json.dumps(forced_targets, ensure_ascii=False),
        column_samples=column_samples,
    )

    from openai import OpenAI
    from common.llm.key_pool import require_ustc_api_keys
    from common.llm.response_utils import extract_message_text
    client = OpenAI(
        api_key=api_key or require_ustc_api_keys(1)[0],
        base_url=base_url,
    )
    import time as _time
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = extract_message_text(resp.choices[0].message)
            if not raw:
                raise ValueError("empty LLM response")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("LLM column response must be a JSON object")
            roles = _normalize_column_roles(
                result,
                columns,
                target_col=target_col,
                known_target_cols=forced_targets,
            )
            _validate_column_roles(
                roles,
                columns,
                context="LLM column response",
            )
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                delay = 2 ** attempt
                print(f"[detect] Retry {attempt+1}/3 in {delay}s: {e}")
                _time.sleep(delay)
            else:
                raise RuntimeError(
                    f"LLM column detection failed after 3 attempts: {last_err}"
                ) from last_err
    print(f"[detect] LLM classified {len(df.columns)} cols → "
          f"{len(roles['element_cols'])} elements, "
          f"{len(roles['process_cols'])} process, "
          f"{len(roles['target_cols'])} targets/properties, "
          f"{len(roles['metadata_cols'])} metadata")

    if cache_path is not None:
        cache_data = {
            "schema_version": COLUMN_ROLE_SCHEMA_VERSION,
            "prompt_version": COLUMN_ROLE_PROMPT_VERSION,
            "model": model,
            "api_base_url": base_url,
            "column_schema_fingerprint": schema_fingerprint,
            "source_order_fingerprint": source_order_fingerprint,
            "anchor_target_col": target_col,
            **roles,
        }
        tmp = cache_path.with_suffix(".columns.tmp")
        tmp.write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(cache_path)
        print(f"[detect] Cached to {cache_path.name}")

    return roles


def _filter_leaky_llm_features(
    llm_features: pd.DataFrame,
    cached: dict,
    *,
    target_col: str,
    forbidden_target_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Remove cached LLM features whose formula uses any measured response."""
    meta_target = cached.get("metadata", {}).get("per_target", {}).get(target_col, {})
    selected_formula_names = set(meta_target.get("selected_cols", []))
    formula_by_name = {
        item.get("name"): item.get("formula", "")
        for item in meta_target.get("all_formulas", [])
        if item.get("name")
    }
    leaky_names = {
        name
        for name in selected_formula_names
        if name not in formula_by_name
        or _formula_uses_any_column(
            formula_by_name[name], set(forbidden_target_cols)
        )
    }
    # Metadata is the dependency proof. A feature without a matching selected
    # formula is rejected rather than treated as safe.
    leaky_names.update(
        str(col)
        for col in llm_features.columns
        if str(col).startswith("LLM_") and str(col) not in selected_formula_names
    )
    leaky_names.discard(None)
    # Also reject direct response columns even if cache metadata is incomplete.
    leaky_names.update(c for c in forbidden_target_cols if c in llm_features.columns)
    dropped = sorted(str(c) for c in leaky_names if c in llm_features.columns)
    return llm_features.drop(columns=dropped, errors="ignore"), dropped


def _validated_llm_feature_frame(
    cached: dict,
    *,
    target_col: str,
    forbidden_target_cols: list[str],
    expected_index: pd.Index,
) -> tuple[pd.DataFrame, list[str]]:
    """Return one structurally aligned, finite, leakage-filtered LLM table.

    The feature designer validates scientific cache identity.  This second
    boundary validates the cached payload itself so that a truncated,
    reordered, duplicated, or non-numeric table cannot be silently aligned to
    the wrong candidates by ``concat``.
    """
    if not isinstance(cached, dict):
        raise ValueError("LLM feature result must be a dictionary")
    feature_tables = cached.get("llm_features")
    if not isinstance(feature_tables, dict):
        raise ValueError("LLM feature result lacks a per-target feature mapping")

    llm_features = feature_tables.get(target_col)
    if llm_features is None:
        return pd.DataFrame(index=expected_index.copy()), []
    if not isinstance(llm_features, pd.DataFrame):
        raise ValueError(
            f"LLM features for {target_col!r} must be a pandas DataFrame"
        )
    if len(llm_features) != len(expected_index):
        raise ValueError(
            f"LLM feature row-count mismatch for {target_col!r}: "
            f"cache={len(llm_features)}, expected={len(expected_index)}"
        )
    if not llm_features.index.equals(expected_index):
        raise ValueError(
            f"LLM feature index mismatch for {target_col!r}; cached rows are "
            "not aligned with the current candidate order"
        )
    if llm_features.columns.duplicated().any():
        duplicates = sorted(
            set(str(c) for c in llm_features.columns[llm_features.columns.duplicated()])
        )
        raise ValueError(
            f"Duplicate LLM feature columns for {target_col!r}: {duplicates}"
        )

    llm_features, dropped = _filter_leaky_llm_features(
        llm_features,
        cached,
        target_col=target_col,
        forbidden_target_cols=forbidden_target_cols,
    )
    if llm_features.empty:
        return llm_features.copy(), dropped

    numeric = llm_features.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(
            f"LLM features for {target_col!r} contain non-numeric or non-finite "
            "values; formula-specific imputation is prohibited"
        )
    return numeric, dropped


def _build_initial_design_frame(
    df: pd.DataFrame,
    designer_cols: list[str],
    target_col: str,
    initial_idx: np.ndarray,
) -> pd.DataFrame:
    """Expose full-pool inputs but labels only for the initial AL membership."""
    design = df.loc[:, designer_cols].copy()
    labeled = np.zeros(len(design), dtype=bool)
    labeled[np.asarray(initial_idx, dtype=int)] = True
    design.loc[~labeled, target_col] = np.nan
    return design


# ── 通用数据集加载 ──────────────────────────────────────────────────

def load_external_dataset(
    xlsx_path: str | Path,
    target_col: str,
    element_cols: list[str] | None = None,
    process_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    output_dir: str | Path | None = None,
    force_rerun_selection: bool = False,
    normalize_mixed_units: bool = False,
    material_system: str | None = None,
    composition_unit: str = "wt.%",
    composition_missing_policy: str = COMPLETE_COMPOSITION_POLICY,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """加载合金数据集, 构建完整特征矩阵 (Magpie + LLM + 筛选).

    适用于任意合金体系 (Al/Cu/Mg/Ti/Steel), 元素列和工艺列自动检测。

    Parameters
    ----------
    xlsx_path : 数据文件路径
    target_col : 目标列名
    element_cols : 元素成分列名 (None=自动检测)
    process_cols : 工艺参数列名 (None=自动检测)
    output_dir : 缓存输出目录 (LLM 特征 / 特征筛选 / 全矩阵)
    force_rerun_selection : 强制重新筛选

    Returns
    -------
    X : 选定特征矩阵
    y : 目标值 (float32)
    comp_df : 元素成分 + 目标列
    """
    from common.feature_engineer.feature_select import (
        FEATURE_SELECTION_POLICY_VERSION,
        select_features,
    )

    df = read_dataset_table(xlsx_path)
    print(f"[data] Loaded {len(df)} samples from {Path(xlsx_path).name}")

    # 自动检测列 (LLM 语义分类, 带缓存)
    if element_cols is None or process_cols is None or target_cols is None:
        roles = _llm_detect_column_roles(
            df,
            target_col,
            xlsx_path=xlsx_path,
            known_target_cols=target_cols,
        )
        if element_cols is None:
            element_cols = roles["element_cols"]
        if process_cols is None:
            process_cols = roles["process_cols"]
        if target_cols is None:
            target_cols = roles["target_cols"]

    element_cols, process_cols, target_cols = _safe_input_roles(
        element_cols=element_cols,
        process_cols=process_cols,
        target_cols=target_cols,
        current_target=target_col,
    )

    df, unit_meta = normalize_mixed_composition_units(
        df,
        element_cols,
        enabled=normalize_mixed_units,
        label=Path(xlsx_path).name,
    )

    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(y)
    if not valid.all():
        df = df.loc[valid].reset_index(drop=True)
        y = y[valid]

    # 成分
    elem_available = [c for c in element_cols if c in df.columns]
    comp_df = df[elem_available].copy()
    comp_df[target_col] = y

    # 特征构建: Magpie
    feats = build_generic_features(
        comp_df[elem_available],
        df,
        elem_available,
        process_cols or [],
        composition_unit=composition_unit,
        composition_missing_policy=composition_missing_policy,
    )
    force_keep_input_cols = _force_keep_input_feature_columns(
        df,
        feats,
        elem_available,
        process_cols,
    )
    assert_no_target_leakage(
        force_keep_input_cols,
        target_cols,
        context=f"{Path(xlsx_path).name} force-kept raw inputs",
    )

    # LLM 特征设计 (跨体系, 带缓存)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_cache_name = _llm_feature_cache_name(
            bool(unit_meta.get("applied")),
            composition_unit,
        )
        llm_cache = output_dir / llm_cache_name

        print(
            f"[data] Designing or identity-validating LLM features for "
            f"{Path(xlsx_path).name}..."
        )
        import asyncio
        from common.feature_engineer.llm_feature_designer import LLMFeatureDesigner

        designer = LLMFeatureDesigner(
            temperature=0.0,
        )
        designer_cols = [
            c for c in [*elem_available, *process_cols, target_col]
            if c in df.columns
        ]
        # This legacy entry point has no AL initial-set membership.  Keep the
        # target name for task semantics, but never expose any target values.
        df_design = df.loc[:, designer_cols].copy()
        df_design.loc[:, target_col] = np.nan
        cached = asyncio.run(designer.design_features(
            df=df_design,
            target_cols=[target_col],
            element_cols=elem_available,
            process_cols=process_cols,
            n_rounds=2,
            max_features_per_round=8,
            use_rag=False,
            cache_path=str(llm_cache),
            material_system=material_system,
            composition_unit=composition_unit,
        ))
        llm_features, dropped = _validated_llm_feature_frame(
            cached,
            target_col=target_col,
            forbidden_target_cols=target_cols,
            expected_index=df.index,
        )
        if llm_features is not None and not llm_features.empty:
            if dropped:
                print(
                    f"[data] Dropped {len(dropped)} LLM features that referenced "
                    f"response/property columns: {dropped}"
                )
            llm_new = llm_features[[c for c in llm_features.columns if c not in feats.columns]]
            if not llm_new.empty:
                feats = pd.concat([feats.reset_index(drop=True), llm_new.reset_index(drop=True)], axis=1)
            n_llm = len(llm_new.columns)
            print(f"[data] Added {n_llm} LLM-designed features")
        else:
            print("[data] No LLM features found in cache")

    # 全矩阵 + 特征筛选
    feats = feats.copy()
    feats[target_col] = y

    if output_dir is not None:
        full_path = output_dir / "features_full.csv"
        feats.to_csv(full_path, index=False)

        sel_csv = output_dir / f"features_selected_{target_col.lower()}.csv"
        selection_report = output_dir / "feature_selection_report.json"
        unit_sidecar = output_dir / f"{sel_csv.stem}.unit_meta.json"
        expected_unit_metadata = {
            "composition_unit_normalization": (
                unit_meta.get("cache_key")
                if unit_meta.get("applied")
                else "disabled"
            ),
            "composition_unit": composition_unit,
            "atomic_descriptor_composition": (
                ATOMIC_DESCRIPTOR_COMPOSITION_VERSION
            ),
        }
        selection_unit_ok = False
        if unit_sidecar.exists():
            try:
                cached_unit = json.loads(unit_sidecar.read_text(encoding="utf-8"))
                selection_unit_ok = cached_unit == expected_unit_metadata
            except json.JSONDecodeError:
                selection_unit_ok = False

        selection_policy_ok = False
        if selection_report.exists():
            try:
                cached_selection = json.loads(
                    selection_report.read_text(encoding="utf-8")
                )
                selection_policy_ok = (
                    cached_selection.get("feature_selection_policy_version")
                    == FEATURE_SELECTION_POLICY_VERSION
                    and cached_selection.get("force_keep_raw_columns")
                    == force_keep_input_cols
                )
            except json.JSONDecodeError:
                selection_policy_ok = False

        if (
            not sel_csv.exists()
            or force_rerun_selection
            or not selection_unit_ok
            or not selection_policy_ok
        ):
            select_features(
                data_path=full_path,
                targets=[target_col],
                force_keep_cols=force_keep_input_cols,
                output_dir=output_dir,
            )
            unit_sidecar.write_text(
                json.dumps(
                    expected_unit_metadata,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        sel_df = pd.read_csv(sel_csv, nrows=1)
        selected_cols = [c for c in sel_df.columns if c != target_col]
        print(f"[data] Feature selection: {len(selected_cols)} features cached")
    else:
        selected_cols = [c for c in feats.columns if c != target_col]

    # 对齐
    available = [c for c in selected_cols if c in feats.columns]
    assert_no_target_leakage(
        available,
        target_cols,
        context=f"{Path(xlsx_path).name} selected features",
    )
    X = feats[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    n_llm = sum(1 for c in available if c.startswith("LLM_"))
    n_magpie = sum(1 for c in available if c.startswith("MagpieData "))
    n_base = len(available) - n_llm - n_magpie
    print(f"[data] Feature matrix: {X.shape[1]} selected cols x {X.shape[0]} rows "
          f"(base:{n_base} Magpie:{n_magpie} LLM:{n_llm})")
    return X, y, comp_df


def load_external_dataset_initial_fit(
    xlsx_path: str | Path,
    target_col: str,
    initial_idx: np.ndarray,
    element_cols: list[str] | None = None,
    process_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    output_dir: str | Path | None = None,
    force_rerun_selection: bool = False,
    normalize_mixed_units: bool = False,
    material_system: str | None = None,
    composition_unit: str = "wt.%",
    composition_missing_policy: str = COMPLETE_COMPOSITION_POLICY,
    feature_representation: str = "raw",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load a dataset while fitting LLM feature design and selection on AL initial rows only.

    The returned X still covers every target-valid row, but any target-supervised step sees
    labels only from ``initial_idx``. This is the AL-safe path for CALM benchmarks.

    ``feature_representation`` controls the ablation boundary: ``raw`` keeps
    composition plus encoded process inputs, ``physics`` adds atom-correct
    Magpie/entropy/radius descriptors, and ``physics_llm`` additionally invokes
    the initial-label-only LLM feature designer.
    """
    from common.feature_engineer.feature_select import (
        FEATURE_SELECTION_POLICY_VERSION,
        select_features,
    )

    feature_representation = str(feature_representation).strip().casefold()
    if feature_representation not in FEATURE_REPRESENTATIONS:
        raise ValueError(
            "feature_representation must be one of "
            f"{sorted(FEATURE_REPRESENTATIONS)}, got "
            f"{feature_representation!r}"
        )

    initial_idx = np.asarray(initial_idx, dtype=int)
    if initial_idx.ndim != 1 or len(initial_idx) == 0:
        raise ValueError("initial_idx must be a non-empty 1D array")

    df = read_dataset_table(xlsx_path)
    print(f"[data] Loaded {len(df)} samples from {Path(xlsx_path).name}")

    if element_cols is None or process_cols is None or target_cols is None:
        roles = _llm_detect_column_roles(
            df,
            target_col,
            xlsx_path=xlsx_path,
            known_target_cols=target_cols,
        )
        if element_cols is None:
            element_cols = roles["element_cols"]
        if process_cols is None:
            process_cols = roles["process_cols"]
        if target_cols is None:
            target_cols = roles["target_cols"]

    element_cols, process_cols, target_cols = _safe_input_roles(
        element_cols=element_cols,
        process_cols=process_cols,
        target_cols=target_cols,
        current_target=target_col,
    )

    df, unit_meta = normalize_mixed_composition_units(
        df,
        element_cols,
        enabled=normalize_mixed_units,
        label=Path(xlsx_path).name,
    )

    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(y)
    if not valid.all():
        df = df.loc[valid].reset_index(drop=True)
        y = y[valid]

    if initial_idx.min() < 0 or initial_idx.max() >= len(df):
        raise IndexError(
            f"initial_idx out of bounds for target-valid data: "
            f"min={int(initial_idx.min())}, max={int(initial_idx.max())}, n={len(df)}"
        )
    if len(np.unique(initial_idx)) != len(initial_idx):
        raise ValueError("initial_idx contains duplicate rows")
    fit_initial_idx = np.sort(initial_idx)

    elem_available = [c for c in (element_cols or []) if c in df.columns]
    comp_df = df[elem_available].copy()
    comp_df[target_col] = y

    feats = build_generic_features(
        comp_df[elem_available],
        df,
        elem_available,
        process_cols or [],
        composition_unit=composition_unit,
        include_atomic_descriptors=feature_representation != "raw",
        composition_missing_policy=composition_missing_policy,
    )
    force_keep_input_cols = _force_keep_input_feature_columns(
        df,
        feats,
        elem_available,
        process_cols,
    )
    assert_no_target_leakage(
        force_keep_input_cols,
        target_cols,
        context=f"{Path(xlsx_path).name} initial-fit force-kept raw inputs",
    )

    initial_fingerprint = hashlib.sha1(
        np.asarray(fit_initial_idx, dtype=np.int64).tobytes()
    ).hexdigest()[:12]
    designer_cols = [
        c for c in [*elem_available, *process_cols, target_col]
        if c in df.columns
    ]
    input_fingerprint_cols = [
        c for c in [*elem_available, *process_cols]
        if c in df.columns
    ]
    pool_x_fingerprint = dataframe_fingerprint(df, input_fingerprint_cols)
    df_design = _build_initial_design_frame(
        df,
        designer_cols,
        target_col,
        fit_initial_idx,
    )
    if feature_representation == "physics_llm":
        visible_initial_label_fingerprint = dataframe_fingerprint(
            df_design.iloc[fit_initial_idx],
            [target_col],
        )
        source_fingerprint_scope = "full_pool_x_plus_initial_labels_only"
        source_fingerprint_payload = {
            "pool_x": pool_x_fingerprint,
            "visible_initial_labels": visible_initial_label_fingerprint,
        }
    else:
        visible_initial_label_fingerprint = "not_used"
        source_fingerprint_scope = "full_pool_x_only_target_independent"
        source_fingerprint_payload = {"pool_x": pool_x_fingerprint}
    source_fingerprint = hashlib.sha256(
        json.dumps(
            source_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    fit_signature = {
        "feature_fit_version": INITIAL_FEATURE_FIT_VERSION,
        "feature_representation_version": FEATURE_REPRESENTATION_VERSION,
        "feature_representation": feature_representation,
        "atomic_descriptor_composition": (
            ATOMIC_DESCRIPTOR_COMPOSITION_VERSION
            if feature_representation != "raw"
            else "disabled"
        ),
        "feature_selection_policy_version": FEATURE_SELECTION_POLICY_VERSION,
        "raw_input_force_keep_version": RAW_INPUT_FORCE_KEEP_VERSION,
        "force_keep_input_cols": force_keep_input_cols,
        "process_feature_encoding": PROCESS_FEATURE_ENCODING_VERSION,
        "column_role_schema_version": COLUMN_ROLE_SCHEMA_VERSION,
        "llm_feature_leakage_guard_version": LLM_FEATURE_LEAKAGE_GUARD_VERSION,
        "llm_feature_material_profile_version": LLM_FEATURE_MATERIAL_PROFILE_VERSION,
        "material_system": str(material_system or "").strip(),
        "composition_unit": composition_unit,
        "composition_missing_policy": composition_missing_policy,
        "column_schema_fingerprint": _column_schema_fingerprint(df),
        "source_fingerprint": source_fingerprint,
        "source_fingerprint_scope": source_fingerprint_scope,
        "pool_x_fingerprint": pool_x_fingerprint,
        "visible_initial_label_fingerprint": visible_initial_label_fingerprint,
        "target_col": target_col,
        "target_cols": target_cols,
        "element_cols": elem_available,
        "process_cols": process_cols,
    }
    signature_raw = json.dumps(
        fit_signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fit_signature_id = hashlib.sha1(signature_raw.encode("utf-8")).hexdigest()[:12]
    fit_dir = Path(output_dir) if output_dir is not None else None
    if fit_dir is not None:
        # Keep both 12-hex audit identities, but use compact labels.  The v26
        # output root is long enough that the former
        # ``initial_<id>_fit_<id>`` spelling pushed some Windows artifacts
        # beyond the traditional MAX_PATH limit (for example, the
        # FeCoNi__Coercivity selected-feature CSV reached 262 characters).
        fit_dir = fit_dir / f"i{initial_fingerprint}_f{fit_signature_id}"
        fit_dir.mkdir(parents=True, exist_ok=True)

    if feature_representation == "physics_llm":
        llm_cache = None
        if fit_dir is not None:
            llm_cache_name = _llm_feature_cache_name(
                bool(unit_meta.get("applied")),
                composition_unit,
            )
            llm_cache = fit_dir / llm_cache_name
        print(
            f"[data] Designing or identity-validating initial-fit LLM features "
            f"({len(initial_idx)}/{len(df)} labels visible)..."
        )
        import asyncio
        from common.feature_engineer.llm_feature_designer import LLMFeatureDesigner

        designer = LLMFeatureDesigner(
            temperature=0.0,
        )
        cached = asyncio.run(designer.design_features(
            df=df_design,
            target_cols=[target_col],
            element_cols=elem_available,
            process_cols=process_cols,
            n_rounds=2,
            max_features_per_round=8,
            use_rag=False,
            cache_path=str(llm_cache) if llm_cache is not None else None,
            material_system=material_system,
            composition_unit=composition_unit,
        ))

        llm_features, dropped = _validated_llm_feature_frame(
            cached,
            target_col=target_col,
            forbidden_target_cols=target_cols,
            expected_index=df.index,
        )
        if llm_features is not None and not llm_features.empty:
            if dropped:
                print(
                    f"[data] Dropped {len(dropped)} LLM features that referenced "
                    f"response/property columns: {dropped}"
                )

            llm_new = llm_features[[c for c in llm_features.columns if c not in feats.columns]]
            if not llm_new.empty:
                feats = pd.concat([feats.reset_index(drop=True), llm_new.reset_index(drop=True)], axis=1)
            print(f"[data] Added {len(llm_new.columns)} initial-fit LLM-designed features")
        else:
            print("[data] No LLM features found in initial-fit cache")

    feats = feats.copy()
    feats[target_col] = y

    if fit_dir is not None:
        full_path = fit_dir / "features_full_all_rows.csv"
        initial_path = fit_dir / "features_initial_fit.csv"
        initial_label_mask = np.zeros(len(feats), dtype=bool)
        initial_label_mask[fit_initial_idx] = True
        full_export = feats.copy()
        full_export.loc[~initial_label_mask, target_col] = np.nan
        full_export.to_csv(full_path, index=False)
        feats.iloc[fit_initial_idx].reset_index(drop=True).to_csv(initial_path, index=False)

        sel_csv = fit_dir / f"features_selected_{target_col.lower()}.csv"
        metadata_path = fit_dir / "initial_fit_metadata.json"
        metadata = {
            **fit_signature,
            "feature_fit_scope": "initial",
            "initial_fraction_realized": float(len(initial_idx) / len(df)),
            "initial_size": int(len(initial_idx)),
            "n_samples": int(len(df)),
            "initial_idx": [int(i) for i in fit_initial_idx.tolist()],
            "composition_unit_normalization": (
                unit_meta.get("cache_key") if unit_meta.get("applied") else "disabled"
            ),
        }
        metadata_ok = False
        if metadata_path.exists():
            try:
                metadata_ok = json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
            except json.JSONDecodeError:
                metadata_ok = False

        if not sel_csv.exists() or force_rerun_selection or not metadata_ok:
            select_features(
                data_path=initial_path,
                targets=[target_col],
                force_keep_cols=force_keep_input_cols,
                output_dir=fit_dir,
                report_name="feature_selection_report.initial_fit.json",
            )
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        sel_df = pd.read_csv(sel_csv, nrows=1)
        selected_cols = [c for c in sel_df.columns if c != target_col]
        print(
            f"[data] Initial-fit feature selection: {len(selected_cols)} features cached "
            f"from {len(initial_idx)} labeled rows"
        )
    else:
        selected_cols = [c for c in feats.columns if c != target_col]

    available = [c for c in selected_cols if c in feats.columns]
    assert_no_target_leakage(
        available,
        target_cols,
        context=f"{Path(xlsx_path).name} initial-fit selected features",
    )
    X = feats[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    n_llm = sum(1 for c in available if c.startswith("LLM_"))
    n_magpie = sum(1 for c in available if c.startswith("MagpieData "))
    n_base = len(available) - n_llm - n_magpie
    print(f"[data] Feature matrix: {X.shape[1]} selected cols x {X.shape[0]} rows "
          f"(base:{n_base} Magpie:{n_magpie} LLM:{n_llm})")
    return X, y, comp_df
