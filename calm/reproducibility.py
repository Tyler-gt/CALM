"""Compact source, environment and sample identities for safe checkpoint resume."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Iterable

PROJECT = Path(__file__).resolve().parent.parent
IMPLEMENTATION_FINGERPRINT_VERSION = "calm_method_source_sha256_v1"
ENVIRONMENT_FINGERPRINT_VERSION = "calm_runtime_versions_v1"
CANDIDATE_INDEX_AUDIT_VERSION = "explicit_initial_and_batch_indices_sha256_v1"

_SOURCE_PATTERNS = (
    "calm/*.py", "calm/strategies/*.py", "common/*.py",
    "common/feature_engineer/*.py", "common/llm/*.py",
    "common/nn_ensemble/*.py",
)
_REQUIRED_FILES = (
    "run.py", "requirements.txt", "calm/runner.py", "calm/data.py",
    "calm/experiment.py", "calm/objective.py", "calm/dynamic_scorer.py",
    "calm/metrics.py", "calm/plot.py", "calm/reproducibility.py",
    "common/llm/reviewer.py", "common/llm/key_pool.py",
    "common/nn_ensemble/model.py",
)
_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch",
    "matplotlib", "openpyxl", "openai", "python-dotenv", "joblib",
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def implementation_fingerprint(project: str | Path = PROJECT) -> dict:
    """Bind a checkpoint to method code, independent of Git or publication files."""
    root = Path(project).resolve()
    paths = {root / name for name in _REQUIRED_FILES}
    missing = sorted(path.relative_to(root).as_posix() for path in paths
                     if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing method files: " + ", ".join(missing))
    for pattern in _SOURCE_PATTERNS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    files = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        raw = path.read_bytes()
        files.append({"path": path.relative_to(root).as_posix(),
                      "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"version": IMPLEMENTATION_FINGERPRINT_VERSION,
            "sha256": _canonical_sha256(files), "files": files}


def environment_fingerprint() -> dict:
    """Record runtime versions so incompatible checkpoints are not resumed."""
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    identity = {"version": ENVIRONMENT_FINGERPRINT_VERSION,
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(), "packages": packages}
    return {**identity, "sha256": _canonical_sha256(identity)}


def candidate_index_fingerprint(indices: Iterable[int]) -> str:
    """Identify an ordered initial or queried candidate sequence."""
    return _canonical_sha256({"version": CANDIDATE_INDEX_AUDIT_VERSION,
                              "indices": [int(index) for index in indices]})
