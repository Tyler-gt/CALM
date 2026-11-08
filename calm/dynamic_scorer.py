"""Dynamic LLM scorer for CALM.

It injects current AL progress into the LLM prompt and periodically rescales
only the physical-score channel used by CALM acquisition.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from calm.experiment import (
    CANDIDATE_MEMORY_POLICY_VERSION,
    CANDIDATE_MEMORY_PREFILTER_K,
    DYNAMIC_FEEDBACK_POLICY_VERSION,
    DYNAMIC_PREQUERY_LEDGER_VERSION,
    LLM_EVIDENCE_VERSION,
)
from calm.objective import (
    to_raw_target,
    validate_objective_direction,
)

if TYPE_CHECKING:
    from common.llm.reviewer import LLMReviewer


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _input_records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    selected = frame.loc[:, columns].astype(object)
    selected = selected.where(pd.notna(selected), None)
    return selected.to_dict("records")


class DynamicLLMScorer:
    """Wrap LLMReviewer and build AL progress reports."""

    def __init__(
        self,
        reviewer: "LLMReviewer",
        comp_df: pd.DataFrame,
        element_cols: list[str],
        process_cols: list[str] | None = None,
        target_col: str = "target_property",
        objective_direction: str = "maximize",
        forbidden_target_cols: list[str] | None = None,
        anchor_best_k: int = 3,
        anchor_worst_k: int = 2,
        anchor_nearby_k: int = 3,
        candidate_memory_prefilter_k: int = CANDIDATE_MEMORY_PREFILTER_K,
        audit_path: Path | None = None,
    ):
        self.reviewer = reviewer
        self.llm_identity = {
            "llm_api_base_url": reviewer.base_url,
            "llm_scoring_model": reviewer.model,
        }
        self.composition_unit = str(
            getattr(reviewer, "composition_unit", "wt.%")
        )
        if self.composition_unit not in {"wt.%", "at.%"}:
            raise ValueError(
                "reviewer.composition_unit must be exactly 'wt.%' or 'at.%'"
            )
        self.comp_df = comp_df
        self.element_cols = [c for c in element_cols if c in comp_df.columns]
        self.process_cols = [
            c for c in (process_cols or []) if c in comp_df.columns
        ]
        blocked_targets = {
            str(col).casefold()
            for col in [target_col, *(forbidden_target_cols or [])]
        }
        self.input_cols = []
        for col in [*self.element_cols, *self.process_cols]:
            if str(col).casefold() not in blocked_targets and col not in self.input_cols:
                self.input_cols.append(col)
        leaked = [
            col for col in self.input_cols
            if str(col).casefold() in blocked_targets
        ]
        if leaked:
            raise RuntimeError(
                f"Dynamic LLM scorer received response/property inputs: {leaked}"
            )
        self.target_col = target_col
        self.objective_direction = validate_objective_direction(
            objective_direction
        )
        self.anchor_best_k = max(0, int(anchor_best_k))
        self.anchor_worst_k = max(0, int(anchor_worst_k))
        self.anchor_nearby_k = max(0, int(anchor_nearby_k))
        if self.anchor_best_k + self.anchor_worst_k + self.anchor_nearby_k <= 0:
            raise ValueError("At least one feedback anchor must be configured")
        self.candidate_memory_prefilter_k = max(
            1, int(candidate_memory_prefilter_k)
        )
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self._run_index: int | None = None
        self._strategy_name = "CALM"
        self._seen_audit_ids: set[str] = set()
        self._source_evidence_ids: dict[tuple[str, int], str] = {}
        self._last_anchor_manifest: list[dict] = []
        self._last_anchor_manifest_sha256 = ""
        self._candidate_memory: dict[int, dict] = {}
        self._last_candidate_memory_manifest: list[dict] = []
        self._last_candidate_memory_manifest_sha256 = ""
        self._last_candidate_memory_by_index: dict[int, dict] = {}
        self._load_audit_ids()

    def _load_audit_ids(self) -> None:
        """Recover completed audit events and remove any interrupted JSONL line."""
        if self.audit_path is None or not self.audit_path.exists():
            return
        valid_records: list[dict] = []
        needs_repair = False
        try:
            raw_text = self.audit_path.read_text(encoding="utf-8")
            needs_repair = bool(raw_text and not raw_text.endswith("\n"))
            for line in raw_text.splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    needs_repair = True
                    continue
                valid_records.append(record)
                event_id = record.get("evidence_id")
                if event_id:
                    self._seen_audit_ids.add(str(event_id))
                    if record.get("phase") == "dynamic":
                        try:
                            source_key = (
                                str(record["feedback_fingerprint"]),
                                int(record["candidate_index"]),
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                        self._source_evidence_ids[source_key] = str(event_id)
        except (OSError, UnicodeDecodeError):
            return
        if needs_repair:
            tmp = self.audit_path.with_suffix(".repair.tmp")
            with tmp.open("w", encoding="utf-8", newline="\n") as stream:
                for record in valid_records:
                    stream.write(json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=_json_scalar,
                    ))
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(self.audit_path)

    def set_run_context(self, run_index: int, strategy_name: str = "CALM") -> None:
        self._run_index = int(run_index)
        self._strategy_name = str(strategy_name)

    def reset_run_state(self) -> None:
        """Clear candidate memories between independently seeded AL runs."""
        self._candidate_memory = {}
        self._last_candidate_memory_manifest = []
        self._last_candidate_memory_manifest_sha256 = ""
        self._last_candidate_memory_by_index = {}
        self._last_anchor_manifest = []
        self._last_anchor_manifest_sha256 = ""

    def export_run_state(self) -> dict:
        """Serialize target-blind selections for iteration checkpointing."""
        return {
            **self.llm_identity,
            "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
            "candidate_memories": [
                dict(self._candidate_memory[index])
                for index in sorted(self._candidate_memory)
            ],
        }

    def import_run_state(self, state: dict) -> None:
        """Restore persistent memories without re-querying the reviewer."""
        if not state:
            self.reset_run_state()
            return
        if any(state.get(key) != value for key, value in self.llm_identity.items()):
            raise ValueError("Candidate-memory LLM endpoint/model mismatch")
        if state.get("candidate_memory_policy") != CANDIDATE_MEMORY_POLICY_VERSION:
            raise ValueError("Candidate-memory checkpoint policy mismatch")
        restored: dict[int, dict] = {}
        for raw in state.get("candidate_memories", []):
            memory = dict(raw)
            candidate_index = int(memory["candidate_index"])
            anchor_index = int(memory["anchor_candidate_index"])
            if (
                candidate_index in restored
                or candidate_index < 0
                or candidate_index >= len(self.comp_df)
                or anchor_index < 0
                or anchor_index >= len(self.comp_df)
            ):
                raise ValueError("Invalid candidate-memory checkpoint entry")
            memory["candidate_index"] = candidate_index
            memory["anchor_candidate_index"] = anchor_index
            memory["distance_to_candidate"] = float(
                memory["distance_to_candidate"]
            )
            memory["created_iteration"] = int(memory["created_iteration"])
            memory["revision"] = int(memory.get("revision", 0))
            restored[candidate_index] = memory
        self._candidate_memory = restored
        self._last_candidate_memory_manifest = []
        self._last_candidate_memory_manifest_sha256 = ""
        self._last_candidate_memory_by_index = {}

    def _write_audit_records(self, records: list[dict]) -> None:
        if self.audit_path is None or not records:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as stream:
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

    def _append_feedback_record(
        self,
        *,
        feedback_fingerprint: str,
        iteration: int | None,
        progress_report: str,
    ) -> None:
        if self.audit_path is None:
            return
        trace = {
            **self.llm_identity,
            "evidence_version": LLM_EVIDENCE_VERSION,
            "phase": "dynamic_feedback",
            "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
            "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
            "composition_unit": self.composition_unit,
            "objective_direction": self.objective_direction,
            "run_index": self._run_index,
            "strategy": self._strategy_name,
            "iteration": None if iteration is None else int(iteration),
            "feedback_fingerprint": str(feedback_fingerprint),
            "progress_report": str(progress_report),
            "progress_report_sha256": hashlib.sha256(
                str(progress_report).encode("utf-8")
            ).hexdigest(),
            "anchors": self._last_anchor_manifest,
            "anchor_manifest_sha256": self._last_anchor_manifest_sha256,
            "candidate_memories": self._last_candidate_memory_manifest,
            "candidate_memory_manifest_sha256": (
                self._last_candidate_memory_manifest_sha256
            ),
        }
        raw = json.dumps(trace, ensure_ascii=False, sort_keys=True)
        event_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        if event_id in self._seen_audit_ids:
            return
        trace["evidence_id"] = event_id
        self._seen_audit_ids.add(event_id)
        self._write_audit_records([trace])

    def _append_audit_records(
        self,
        *,
        pool_indices: np.ndarray,
        candidates: list[dict],
        results: list[dict],
        feedback_fingerprint: str,
        iteration: int | None,
        candidate_progress_reports: list[str] | None = None,
    ) -> None:
        if self.audit_path is None:
            return
        records: list[dict] = []
        reports = (
            [None] * len(candidates)
            if candidate_progress_reports is None
            else list(candidate_progress_reports)
        )
        if len(reports) != len(candidates):
            raise ValueError("Candidate progress reports do not match results")
        for candidate_index, candidate, result, candidate_report in zip(
            pool_indices, candidates, results, reports
        ):
            memory = self._last_candidate_memory_by_index.get(
                int(candidate_index)
            )
            trace = {
                **self.llm_identity,
                "evidence_version": LLM_EVIDENCE_VERSION,
                "phase": "dynamic",
                "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
                "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
                "composition_unit": self.composition_unit,
                "objective_direction": self.objective_direction,
                "run_index": self._run_index,
                "strategy": self._strategy_name,
                "iteration": None if iteration is None else int(iteration),
                "feedback_fingerprint": str(feedback_fingerprint),
                "anchor_manifest_sha256": self._last_anchor_manifest_sha256,
                "candidate_memory_manifest_sha256": (
                    self._last_candidate_memory_manifest_sha256
                ),
                "candidate_index": int(candidate_index),
                "candidate_input": candidate,
                "s_phys": float(result.get("s_phys", 0.0)),
                "s_target_raw": float(result.get("s_target_raw", 0.0)),
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
            if memory is not None:
                memory_prompt = self._candidate_memory_suffix(memory)
                trace.update({
                    "candidate_memory": memory,
                    "candidate_memory_sha256": self._manifest_sha256(memory),
                    "candidate_memory_prompt": memory_prompt,
                    "candidate_memory_prompt_sha256": hashlib.sha256(
                        memory_prompt.encode("utf-8")
                    ).hexdigest(),
                })
            if candidate_report is not None:
                trace["candidate_progress_report_sha256"] = hashlib.sha256(
                    str(candidate_report).encode("utf-8")
                ).hexdigest()
            raw = json.dumps(
                trace,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_scalar,
            )
            event_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            if event_id in self._seen_audit_ids:
                continue
            trace["evidence_id"] = event_id
            records.append(trace)
            self._seen_audit_ids.add(event_id)
            self._source_evidence_ids[
                (str(feedback_fingerprint), int(candidate_index))
            ] = event_id
        self._write_audit_records(records)

    @staticmethod
    def _format_value(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.4g}"
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    @staticmethod
    def _is_near_zero(value) -> bool:
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        return abs(float(value)) <= 1e-12

    def _input_summary(self, row: pd.Series) -> str:
        parts = []
        for col in self.input_cols:
            if col not in row.index or col == self.target_col:
                continue
            value = row[col]
            if col in self.element_cols and self._is_near_zero(value):
                continue
            formatted = self._format_value(value)
            if formatted:
                suffix = (
                    f" {self.composition_unit}"
                    if col in self.element_cols
                    else ""
                )
                parts.append(f"{col}={formatted}{suffix}")
        return ", ".join(parts) if parts else "(no non-target input variables)"

    @staticmethod
    def _fingerprint(payload: dict) -> str:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _manifest_sha256(value: object) -> str:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_scalar,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _candidate_input(self, candidate_index: int) -> dict:
        return _input_records(
            self.comp_df.iloc[[int(candidate_index)]],
            self.input_cols,
        )[0]

    def _candidate_distance(self, left_index: int, right_index: int) -> float:
        """Mixed-type X-only distance used only to choose nearby anchors."""
        if not self.input_cols:
            return 0.0
        distances: list[float] = []
        left = self.comp_df.iloc[int(left_index)]
        right = self.comp_df.iloc[int(right_index)]
        for col in self.input_cols:
            left_value = left[col]
            right_value = right[col]
            left_missing = bool(pd.isna(left_value))
            right_missing = bool(pd.isna(right_value))
            if left_missing and right_missing:
                distances.append(0.0)
                continue
            if left_missing or right_missing:
                distances.append(1.0)
                continue

            series = self.comp_df[col]
            numeric = pd.to_numeric(series, errors="coerce")
            reported = int(series.notna().sum())
            is_numeric = reported > 0 and int(numeric.notna().sum()) == reported
            if is_numeric:
                low = float(numeric.min())
                high = float(numeric.max())
                scale = high - low
                if np.isfinite(scale) and scale > 1e-12:
                    distance = abs(float(left_value) - float(right_value)) / scale
                else:
                    distance = 0.0 if float(left_value) == float(right_value) else 1.0
                distances.append(float(min(max(distance, 0.0), 1.0)))
            else:
                distances.append(
                    0.0
                    if str(left_value).strip().casefold()
                    == str(right_value).strip().casefold()
                    else 1.0
                )
        return float(np.mean(distances)) if distances else 0.0

    def _candidate_memory_options(
        self,
        candidate_index: int,
        train_idx: np.ndarray,
    ) -> list[dict]:
        """Return deterministic X-only nearest measured options with hidden y."""
        ordered = sorted(
            (
                self._candidate_distance(int(candidate_index), int(anchor_index)),
                int(anchor_index),
            )
            for anchor_index in np.asarray(train_idx, dtype=int)
        )[: self.candidate_memory_prefilter_k]
        return [
            {
                "option_id": f"P{ordinal}",
                "candidate_index": int(anchor_index),
                "candidate_input": self._candidate_input(anchor_index),
                "distance_to_candidate": float(distance),
            }
            for ordinal, (distance, anchor_index) in enumerate(ordered, 1)
        ]

    def _append_candidate_memory_selection_record(
        self,
        *,
        candidate_index: int,
        options: list[dict],
        result: dict,
        iteration: int,
    ) -> str:
        """Audit the LLM relevance choice while proving target values were hidden."""
        selected_option_id = str(result["selected_option_id"])
        selected = next(
            option for option in options
            if str(option["option_id"]) == selected_option_id
        )
        trace = {
            **self.llm_identity,
            "evidence_version": LLM_EVIDENCE_VERSION,
            "phase": "candidate_memory_select",
            "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
            "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
            "composition_unit": self.composition_unit,
            "objective_direction": self.objective_direction,
            "run_index": self._run_index,
            "strategy": self._strategy_name,
            "iteration": int(iteration),
            "candidate_index": int(candidate_index),
            "candidate_input": self._candidate_input(candidate_index),
            "prefilter_k": int(self.candidate_memory_prefilter_k),
            "target_values_hidden": True,
            "anchor_options": options,
            "selected_option_id": selected_option_id,
            "selected_anchor_candidate_index": int(selected["candidate_index"]),
            "selection_reason": str(result["reason"]),
            "key_alias": str(result.get("_key_alias", "unknown")),
        }
        raw = json.dumps(
            trace,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_scalar,
        )
        event_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        if event_id not in self._seen_audit_ids:
            trace["evidence_id"] = event_id
            self._seen_audit_ids.add(event_id)
            self._write_audit_records([trace])
        return event_id

    def prepare_candidate_memories(
        self,
        *,
        train_idx: np.ndarray,
        query_indices: np.ndarray,
        iteration: int,
    ) -> None:
        """Create each shortlist candidate's immutable target-blind M1 memory."""
        train = np.asarray(train_idx, dtype=int)
        queries = np.asarray(query_indices, dtype=int)
        if (
            train.ndim != 1
            or queries.ndim != 1
            or len(train) == 0
            or len(np.unique(train)) != len(train)
            or len(np.unique(queries)) != len(queries)
        ):
            raise ValueError("Candidate-memory indices must be unique 1D arrays")
        if np.any(train < 0) or np.any(train >= len(self.comp_df)):
            raise IndexError("Candidate-memory train index is out of range")
        if np.any(queries < 0) or np.any(queries >= len(self.comp_df)):
            raise IndexError("Candidate-memory query index is out of range")
        if set(train.tolist()) & set(queries.tolist()):
            raise ValueError("Candidate memory may only be created for unmeasured rows")

        missing = [
            int(index) for index in queries
            if int(index) not in self._candidate_memory
        ]
        if not missing:
            return
        if not hasattr(self.reviewer, "select_relevant_anchor_batch"):
            raise RuntimeError(
                "Reviewer does not implement target-blind candidate-memory selection"
            )

        option_sets = [
            self._candidate_memory_options(candidate_index, train)
            for candidate_index in missing
        ]
        requests = [
            {
                "candidate_input": self._candidate_input(candidate_index),
                "anchor_options": options,
            }
            for candidate_index, options in zip(missing, option_sets)
        ]

        async def _select() -> list[dict]:
            return await self.reviewer.select_relevant_anchor_batch(requests)

        results = asyncio.run(_select())
        if len(results) != len(missing):
            raise RuntimeError("Candidate-memory reviewer returned wrong result count")
        for candidate_index, options, result in zip(missing, option_sets, results):
            selected_option_id = str(result.get("selected_option_id", ""))
            matches = [
                option for option in options
                if str(option["option_id"]) == selected_option_id
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Candidate-memory reviewer selected an unknown option"
                )
            selected = matches[0]
            evidence_id = self._append_candidate_memory_selection_record(
                candidate_index=candidate_index,
                options=options,
                result=result,
                iteration=int(iteration),
            )
            self._candidate_memory[candidate_index] = {
                "candidate_index": int(candidate_index),
                "anchor_candidate_index": int(selected["candidate_index"]),
                "distance_to_candidate": float(
                    selected["distance_to_candidate"]
                ),
                "created_iteration": int(iteration),
                "revision": 0,
                "selection_reason": str(result["reason"]),
                "selection_key_alias": str(
                    result.get("_key_alias", "unknown")
                ),
                "selection_evidence_id": str(evidence_id),
            }
        print(
            f"[DynamicLLMScorer] Created {len(missing)} persistent M1 "
            f"candidate memories at iteration {int(iteration)}",
            flush=True,
        )

    def _build_candidate_memory_manifest(
        self,
        query_indices: np.ndarray,
        train_idx: np.ndarray,
        y_all: np.ndarray,
        shared_anchors: list[dict],
    ) -> list[dict]:
        y = np.asarray(y_all, dtype=float)
        measured = {int(index) for index in np.asarray(train_idx, dtype=int)}
        shared_by_index = {
            int(anchor["candidate_index"]): str(anchor["anchor_id"])
            for anchor in shared_anchors
        }
        manifest: list[dict] = []
        for candidate_index in np.asarray(query_indices, dtype=int):
            idx = int(candidate_index)
            if idx not in self._candidate_memory:
                raise RuntimeError(
                    f"Candidate {idx} has no persistent M1 memory"
                )
            stored = self._candidate_memory[idx]
            anchor_index = int(stored["anchor_candidate_index"])
            if anchor_index not in measured or anchor_index == idx:
                raise RuntimeError(
                    f"Candidate {idx} has an invalid unmeasured M1 anchor"
                )
            item = {
                "candidate_index": idx,
                "memory_anchor_id": "M1",
                "anchor_candidate_index": anchor_index,
                "target_value": float(
                    to_raw_target(
                        [y[anchor_index]],
                        self.objective_direction,
                    )[0]
                ),
                "anchor_input": self._candidate_input(anchor_index),
                "distance_to_candidate": float(
                    stored["distance_to_candidate"]
                ),
                "created_iteration": int(stored["created_iteration"]),
                "revision": int(stored.get("revision", 0)),
                "selection_evidence_id": str(stored["selection_evidence_id"]),
            }
            if anchor_index in shared_by_index:
                item["duplicates_shared_anchor_id"] = shared_by_index[anchor_index]
            manifest.append(item)
        return manifest

    @staticmethod
    def _candidate_memory_suffix(memory: dict) -> str:
        duplicate = memory.get("duplicates_shared_anchor_id")
        if duplicate:
            reference = (
                f"M1 is the same measured entry already shown as {duplicate}."
            )
        else:
            reference = (
                "M1 measured input: "
                + json.dumps(
                    memory["anchor_input"],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_scalar,
                )
            )
        return (
            "\n\nCandidate-specific persistent measured memory M1 "
            "(chosen target-blind for mechanism relevance):\n"
            f"{reference}\n"
            f"M1 measured target: {memory['target_value']:.6g}.\n"
            "Use M1 as the primary local comparator and preserve this evidence "
            "across re-evaluations. M1 is not a hard upper or lower bound: the "
            "candidate may rank better or worse when composition/process "
            "differences imply a defensible target mechanism change. Prefer a "
            "near-exact measured comparison over a generic alloy prior, and "
            "state the mechanism supporting any departure from M1."
        )

    def _candidate_progress_report(
        self,
        shared_progress_report: str,
        candidate_index: int,
    ) -> str:
        memory = self._last_candidate_memory_by_index.get(int(candidate_index))
        if memory is None:
            raise RuntimeError(
                f"Candidate {int(candidate_index)} is missing its M1 prompt memory"
            )
        return shared_progress_report + self._candidate_memory_suffix(memory)

    def _select_feedback_anchors(
        self,
        train_idx: np.ndarray,
        y_all: np.ndarray,
        query_indices: np.ndarray | None,
    ) -> list[dict]:
        """Select unique objective-best, objective-worst, and nearby anchors."""
        train = np.asarray(train_idx, dtype=int)
        if train.ndim != 1 or len(train) == 0 or len(np.unique(train)) != len(train):
            raise ValueError("train_idx must contain unique measured candidates")
        y = np.asarray(y_all, dtype=float)
        if np.any(train < 0) or np.any(train >= len(y)):
            raise IndexError("train_idx contains an out-of-range candidate")
        measured = y[train]
        if not np.all(np.isfinite(measured)):
            raise ValueError("Feedback anchors require finite measured targets")

        selected: list[tuple[int, str, float | None]] = []
        selected_set: set[int] = set()

        def _add(local_order: np.ndarray, limit: int, role: str) -> None:
            if limit <= 0:
                return
            for local_index in local_order:
                candidate_index = int(train[int(local_index)])
                if candidate_index in selected_set:
                    continue
                selected.append((candidate_index, role, None))
                selected_set.add(candidate_index)
                if sum(1 for _idx, item_role, _dist in selected if item_role == role) >= limit:
                    return

        # ``y_all`` is the oriented acquisition utility: larger is always
        # better. Candidate index is the deterministic secondary key for ties.
        _add(np.lexsort((train, -measured)), self.anchor_best_k, "best")
        _add(np.lexsort((train, measured)), self.anchor_worst_k, "worst")

        queries = np.asarray(
            [] if query_indices is None else query_indices,
            dtype=int,
        )
        if len(queries) and self.anchor_nearby_k:
            if np.any(queries < 0) or np.any(queries >= len(self.comp_df)):
                raise IndexError("query_indices contains an out-of-range candidate")
            nearby: list[tuple[float, int]] = []
            for candidate_index in train:
                idx = int(candidate_index)
                if idx in selected_set:
                    continue
                distance = min(
                    self._candidate_distance(idx, int(query_index))
                    for query_index in queries
                )
                nearby.append((float(distance), idx))
            for distance, idx in sorted(nearby)[: self.anchor_nearby_k]:
                selected.append((idx, "nearby", distance))
                selected_set.add(idx)

        anchors: list[dict] = []
        for ordinal, (candidate_index, role, distance) in enumerate(selected, 1):
            anchor = {
                "anchor_id": f"A{ordinal}",
                "role": role,
                "candidate_index": int(candidate_index),
                "target_value": float(
                    to_raw_target(
                        [y[int(candidate_index)]],
                        self.objective_direction,
                    )[0]
                ),
                "candidate_input": self._candidate_input(candidate_index),
            }
            if distance is not None:
                anchor["distance_to_shortlist"] = float(distance)
            anchors.append(anchor)
        return anchors

    def build_feedback(
        self,
        train_idx: np.ndarray,
        y_all: np.ndarray,
        current_scores: dict[str, np.ndarray] | None = None,
        score_overrides: dict[str, dict[int, float]] | None = None,
        n_calibration: int = 5,
        query_indices: np.ndarray | None = None,
    ) -> tuple[str, str]:
        """Build one shared anchor-relative prompt for the current shortlist."""
        del current_scores, score_overrides, n_calibration
        train_idx = np.asarray(train_idx, dtype=int)
        n_tested = len(train_idx)
        queries = np.asarray(
            [] if query_indices is None else query_indices,
            dtype=int,
        )
        anchors = self._select_feedback_anchors(
            train_idx,
            y_all,
            queries,
        )
        self._last_anchor_manifest = anchors
        self._last_anchor_manifest_sha256 = self._manifest_sha256(anchors)
        memories = self._build_candidate_memory_manifest(
            queries,
            train_idx,
            y_all,
            anchors,
        ) if len(queries) else []
        self._last_candidate_memory_manifest = memories
        self._last_candidate_memory_manifest_sha256 = self._manifest_sha256(
            memories
        )
        self._last_candidate_memory_by_index = {
            int(memory["candidate_index"]): memory
            for memory in memories
        }
        fingerprint_payload = {
            **self.llm_identity,
            "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
            "target_col": self.target_col,
            "objective_direction": self.objective_direction,
            "composition_unit": self.composition_unit,
            "n_tested": int(n_tested),
            "input_cols": self.input_cols,
            "anchors": anchors,
            "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
            "candidate_memories": memories,
            "query_indices": [
                int(index)
                for index in queries
            ],
        }

        lines = [
            f"Tested {n_tested} candidates. The following anchors are the only "
            "measured target values available for feedback calibration.",
            f"Optimization direction: {self.objective_direction.upper()} the raw "
            f"numerical value of {self.target_col}.",
            (
                "Lower measured values are better for this task; a higher "
                "s_target_raw must indicate a lower expected raw target value."
                if self.objective_direction == "minimize"
                else
                "Higher measured values are better for this task; a higher "
                "s_target_raw must indicate a higher expected raw target value."
            ),
            "\nShared measured anchors (best/worst/nearby roles are deterministic):",
        ]
        for anchor in anchors:
            global_i = int(anchor["candidate_index"])
            comp = self.comp_df.iloc[global_i]
            lines.append(
                f"  {anchor['anchor_id']} [{anchor['role']}]: "
                f"{self.target_col}={float(anchor['target_value']):.6g} | "
                f"{self._input_summary(comp)}"
            )

        lines.append(
            "\nScore the candidate RELATIVE TO THIS SHARED ANCHOR SET. First "
            "identify the composition/process mechanisms controlling the target; "
            "then state whether it should rank better than, between, or worse "
            "than specific anchors under the registered direction. Use "
            "s_target_raw on 0-100 as an ordinal ranking coordinate, "
            "not as a prediction of the target's physical numeric value. Candidates "
            "judged against this same anchor set must be mutually rankable. Avoid "
            "identical scores unless they are indistinguishable for the target. Do "
            "not reward broad alloy plausibility unless it improves the registered "
            "target objective."
        )
        progress_report = "\n".join(lines)
        fingerprint_payload["progress_report_sha256"] = hashlib.sha256(
            progress_report.encode("utf-8")
        ).hexdigest()
        return progress_report, self._fingerprint(fingerprint_payload)

    def build_progress_report(
        self,
        train_idx: np.ndarray,
        y_all: np.ndarray,
        current_scores: dict[str, np.ndarray] | None = None,
        score_overrides: dict[str, dict[int, float]] | None = None,
        n_calibration: int = 5,
        query_indices: np.ndarray | None = None,
    ) -> str:
        """Build a concise progress report from measured candidates."""
        report, _ = self.build_feedback(
            train_idx,
            y_all,
            current_scores=current_scores,
            score_overrides=score_overrides,
            n_calibration=n_calibration,
            query_indices=query_indices,
        )
        return report

    def rescore(
        self,
        pool_indices: np.ndarray,
        train_idx: np.ndarray,
        y_train: np.ndarray,
        current_scores: dict[str, np.ndarray] | None = None,
        score_overrides: dict[str, dict[int, float]] | None = None,
        progress_report: str | None = None,
        feedback_fingerprint: str = "unspecified",
        iteration: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Rescore the current pool subset, returning only s_phys."""
        train_idx = np.asarray(train_idx)
        progress = progress_report
        if progress is None:
            progress = self.build_progress_report(
                train_idx,
                y_train,
                current_scores=current_scores,
                score_overrides=score_overrides,
                query_indices=np.asarray(pool_indices, dtype=int),
            )
        candidate_cols = [
            col for col in self.input_cols
            if col in self.comp_df.columns and col != self.target_col
        ]
        candidates = _input_records(self.comp_df.iloc[pool_indices], candidate_cols)
        candidate_progress_reports = [
            self._candidate_progress_report(progress, int(candidate_index))
            for candidate_index in np.asarray(pool_indices, dtype=int)
        ]

        async def _score():
            return await self.reviewer.review_batch(
                candidates,
                progress_report=candidate_progress_reports,
            )

        results = asyncio.run(_score())
        self._append_feedback_record(
            feedback_fingerprint=feedback_fingerprint,
            iteration=iteration,
            progress_report=progress,
        )
        self._append_audit_records(
            pool_indices=np.asarray(pool_indices, dtype=int),
            candidates=candidates,
            results=results,
            feedback_fingerprint=feedback_fingerprint,
            iteration=iteration,
            candidate_progress_reports=candidate_progress_reports,
        )
        key_alias_counts = Counter(
            str(result.get("_key_alias", "unknown")) for result in results
        )
        print(
            f"[DynamicLLMScorer] Key usage: "
            f"{dict(sorted(key_alias_counts.items()))}",
            flush=True,
        )
        return {
            "s_phys": np.array(
                [r.get("s_phys", 0.0) for r in results], dtype=np.float64
            )
        }

    def record_prequery_ledger(
        self,
        *,
        pool_indices: np.ndarray,
        scores: np.ndarray,
        feedback_fingerprint: str,
        iteration: int | None,
        progress_report: str,
    ) -> None:
        """Seal the final shortlist knowledge order before acquisition selects."""
        indices = np.asarray(pool_indices, dtype=int)
        values = np.asarray(scores, dtype=np.float64)
        if (
            indices.ndim != 1
            or values.shape != indices.shape
            or len(indices) == 0
            or len(np.unique(indices)) != len(indices)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("Pre-query ledger received an invalid shortlist")
        source_ids: list[str] = []
        for candidate_index in indices:
            source_id = self._source_evidence_ids.get(
                (str(feedback_fingerprint), int(candidate_index))
            )
            if source_id is None:
                raise RuntimeError(
                    "Pre-query ledger cannot be written before score evidence "
                    f"exists for candidate {int(candidate_index)}"
                )
            source_ids.append(source_id)
        self._append_feedback_record(
            feedback_fingerprint=feedback_fingerprint,
            iteration=iteration,
            progress_report=progress_report,
        )
        trace = {
            **self.llm_identity,
            "evidence_version": LLM_EVIDENCE_VERSION,
            "phase": "dynamic_prequery",
            "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
            "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
            "dynamic_prequery_ledger": DYNAMIC_PREQUERY_LEDGER_VERSION,
            "composition_unit": self.composition_unit,
            "objective_direction": self.objective_direction,
            "run_index": self._run_index,
            "strategy": self._strategy_name,
            "iteration": None if iteration is None else int(iteration),
            "feedback_fingerprint": str(feedback_fingerprint),
            "anchor_manifest_sha256": self._last_anchor_manifest_sha256,
            "candidate_memory_manifest_sha256": (
                self._last_candidate_memory_manifest_sha256
            ),
            "shortlist_candidate_indices": [int(index) for index in indices],
            "shortlist_scores": [float(value) for value in values],
            "score_evidence_ids": source_ids,
            "shortlist_index_sha256": hashlib.sha256(
                indices.astype(np.int64).tobytes(order="C")
            ).hexdigest(),
            "shortlist_score_sha256": hashlib.sha256(
                values.astype(np.float64).tobytes(order="C")
            ).hexdigest(),
        }
        raw = json.dumps(
            trace,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_scalar,
        )
        event_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        if event_id in self._seen_audit_ids:
            return
        trace["evidence_id"] = event_id
        self._seen_audit_ids.add(event_id)
        self._write_audit_records([trace])

    def record_cache_reuse(
        self,
        *,
        pool_indices: np.ndarray,
        scores: np.ndarray,
        feedback_fingerprint: str,
        iteration: int | None,
        progress_report: str,
    ) -> None:
        """Write an explicit current-iteration link for every reused score."""
        indices = np.asarray(pool_indices, dtype=int)
        values = np.asarray(scores, dtype=float)
        if len(indices) != len(values) or not np.all(np.isfinite(values)):
            raise ValueError("Dynamic cache-reuse audit received invalid scores")
        source_evidence_ids: list[str] = []
        for candidate_index in indices:
            source_key = (str(feedback_fingerprint), int(candidate_index))
            source_evidence_id = self._source_evidence_ids.get(source_key)
            if source_evidence_id is None:
                raise RuntimeError(
                    "Dynamic score cache lacks its source evidence record for "
                    f"feedback/candidate {source_key}"
                )
            source_evidence_ids.append(source_evidence_id)
        self._append_feedback_record(
            feedback_fingerprint=feedback_fingerprint,
            iteration=iteration,
            progress_report=progress_report,
        )
        records: list[dict] = []
        candidate_cols = [
            col for col in self.input_cols
            if col in self.comp_df.columns and col != self.target_col
        ]
        for candidate_index, score, source_evidence_id in zip(
            indices,
            values,
            source_evidence_ids,
        ):
            candidate = _input_records(
                self.comp_df.iloc[[int(candidate_index)]],
                candidate_cols,
            )[0]
            trace = {
                **self.llm_identity,
                "evidence_version": LLM_EVIDENCE_VERSION,
                "phase": "dynamic_cache_reuse",
                "dynamic_feedback_policy": DYNAMIC_FEEDBACK_POLICY_VERSION,
                "candidate_memory_policy": CANDIDATE_MEMORY_POLICY_VERSION,
                "composition_unit": self.composition_unit,
                "objective_direction": self.objective_direction,
                "run_index": self._run_index,
                "strategy": self._strategy_name,
                "iteration": None if iteration is None else int(iteration),
                "feedback_fingerprint": str(feedback_fingerprint),
                "anchor_manifest_sha256": self._last_anchor_manifest_sha256,
                "candidate_memory_manifest_sha256": (
                    self._last_candidate_memory_manifest_sha256
                ),
                "candidate_index": int(candidate_index),
                "candidate_input": candidate,
                "s_phys": float(score),
                "source_evidence_id": source_evidence_id,
                "source_cache_key": (
                    f"{feedback_fingerprint}:{int(candidate_index)}"
                ),
                "reason": (
                    "Exact feedback-prompt/candidate score reused; the source "
                    "dynamic evidence has the same feedback fingerprint and "
                    "candidate index."
                ),
                "retrieval_queries": {},
                "references": [],
            }
            memory = self._last_candidate_memory_by_index.get(
                int(candidate_index)
            )
            if memory is not None:
                memory_prompt = self._candidate_memory_suffix(memory)
                candidate_report = str(progress_report) + memory_prompt
                trace.update({
                    "candidate_memory": memory,
                    "candidate_memory_sha256": self._manifest_sha256(memory),
                    "candidate_memory_prompt": memory_prompt,
                    "candidate_memory_prompt_sha256": hashlib.sha256(
                        memory_prompt.encode("utf-8")
                    ).hexdigest(),
                    "candidate_progress_report_sha256": hashlib.sha256(
                        candidate_report.encode("utf-8")
                    ).hexdigest(),
                })
            raw = json.dumps(
                trace,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_scalar,
            )
            event_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            if event_id in self._seen_audit_ids:
                continue
            trace["evidence_id"] = event_id
            self._seen_audit_ids.add(event_id)
            records.append(trace)
        self._write_audit_records(records)
