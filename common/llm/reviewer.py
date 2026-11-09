"""Single-objective CALM knowledge scoring with measured feedback and no retrieval.

Use set_dataset() to supply input-only context, review_batch() to score candidates,
and select_relevant_anchor_batch() to choose target-blind persistent references.
"""

from __future__ import annotations

import asyncio
import json
import math
import random

import pandas as pd
from openai import APIStatusError

from common.llm.prompts import (
    build_dataset_profile,
    build_history_stats,
    get_messages,
)
from common.llm.response_utils import extract_message_text, parse_json_object
from common.llm.config import get_llm_model
from common.llm.key_pool import (
    AsyncAPIKeyPool,
    NoAvailableAPIKeyError,
    require_api_keys,
)


_COMPOSITION_UNIT_LABELS = {
    "wt.%": "weight percent",
    "at.%": "atomic percent",
}


def _validate_composition_unit(composition_unit: str) -> str:
    unit = str(composition_unit).strip()
    if unit not in _COMPOSITION_UNIT_LABELS:
        raise ValueError(
            "composition_unit must be exactly 'wt.%' or 'at.%', got "
            f"{composition_unit!r}"
        )
    return unit


def _is_permanent_llm_error(exc: Exception) -> bool:
    """Errors that cannot become successful without changing configuration."""
    return isinstance(exc, NoAvailableAPIKeyError) or (
        isinstance(exc, APIStatusError)
        and not AsyncAPIKeyPool.is_retryable_status_error(exc)
    )


async def _gather_batch(coroutines) -> list[dict]:
    """Finish every task before a failed/cancelled batch releases its session."""
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _validate_scoring_response(result: dict) -> None:
    """Validate the fields requested by the paper's scoring prompt."""
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("LLM scoring response requires a non-empty string reason")
    for field in ("principles", "off_target_notes"):
        if field in result and (
            not isinstance(result[field], list)
            or any(not isinstance(item, str) or not item.strip() for item in result[field])
        ):
            raise ValueError(f"LLM scoring response {field} must be a list of strings")
    if "target_mechanism" in result and (
        not isinstance(result["target_mechanism"], str)
        or not result["target_mechanism"].strip()
    ):
        raise ValueError("LLM scoring response target_mechanism must be a non-empty string")
    for field in ("s_target_raw", "s_phys_raw", "s_general_plausibility_raw"):
        if field in result and (
            type(result[field]) is not int or not 0 <= result[field] <= 100
        ):
            raise ValueError(f"LLM scoring response {field} must be an integer in [0, 100]")
    if "s_phys" in result and (
        type(result["s_phys"]) not in (int, float)
        or not 0 <= result["s_phys"] <= 1
        or not math.isfinite(result["s_phys"])
    ):
        raise ValueError("LLM scoring response s_phys must be a finite number in [0, 1]")

def _jsonable_sample(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


# ═══════════════════════════════════════════════════════════════════
# Reviewer
# ═══════════════════════════════════════════════════════════════════

class LLMReviewer:
    """LLM reviewer for multiple alloy systems.

    Call set_dataset() when switching datasets to update profiles and statistics;
    then call review_one() or review_batch() to score candidate alloys.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        material_system: str = "alloy",
        composition_unit: str = "wt.%",
        element_cols: list[str] | None = None,
        process_cols: list[str] | None = None,
        target_col: str = "",
        objective_direction: str = "maximize",
        objective_description: str = "",
        max_concurrency: int = 20,
        rate_limit_batch_size: int = 20,
        rate_limit_sleep_seconds: float = 10.0,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 60.0,
        api_keys: list[str] | None = None,
        min_api_keys: int = 1,
        per_key_concurrency: int = 4,
        key_pool: AsyncAPIKeyPool | None = None,
    ) -> None:
        self.model = get_llm_model() if model is None else model
        self.temperature = temperature
        self.material_system = material_system
        self.composition_unit = _validate_composition_unit(composition_unit)
        self.target_col = target_col
        self.objective_direction = str(objective_direction).strip().lower()
        if self.objective_direction not in {"maximize", "minimize"}:
            raise ValueError(
                "objective_direction must be exactly 'maximize' or 'minimize'"
            )
        self.objective_description = str(objective_description).strip()
        self.max_concurrency = max(1, int(max_concurrency))
        self.rate_limit_batch_size = max(1, int(rate_limit_batch_size))
        self.rate_limit_sleep_seconds = max(0.0, float(rate_limit_sleep_seconds))
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(retry_max_seconds),
        )

        if key_pool is not None:
            if key_pool.key_count < int(min_api_keys):
                raise RuntimeError(
                    f"LLMReviewer requires {int(min_api_keys)} independent API keys"
                )
            self.key_pool = key_pool
        else:
            resolved_keys = (
                list(api_keys)
                if api_keys is not None
                else require_api_keys(min_api_keys)
            )
            if len(set(resolved_keys)) < int(min_api_keys):
                raise RuntimeError(
                    f"LLMReviewer requires {int(min_api_keys)} independent API keys"
                )
            self.key_pool = AsyncAPIKeyPool(
                resolved_keys,
                per_key_concurrency=per_key_concurrency,
            )
        self.base_url = self.key_pool.base_url
        # Precomputed profile text
        self._element_profile: str = ""
        self._history_stats: str = ""
        self._target_info: str = ""

        # Column definitions
        self.element_cols: list[str] = element_cols or []
        self.process_cols: list[str] = process_cols or []
        self.target_cols: list[str] = [target_col] if target_col else []
        self.metadata_cols: list[str] = []
        self._explicit_element_cols = element_cols is not None
        self._explicit_process_cols = process_cols is not None

    # ── Dataset registration ──────────────────────────────────────────────────

    def set_dataset(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "",
        element_cols: list[str] | None = None,
        process_cols: list[str] | None = None,
        composition_unit: str | None = None,
    ) -> None:
        """Register the current dataset and automatically build profiles and statistics.

        Subsequent review_one() user messages automatically use this dataset context.
        Supply audited element_cols and process_cols here or in the constructor;
        an empty process_cols list is valid. Registration never infers roles or
        makes an API request.
        """
        self.target_col = target_col or self.target_col
        if composition_unit is not None:
            self.composition_unit = _validate_composition_unit(composition_unit)

        # Column roles come from the caller's audited dataset registration.
        if element_cols is not None:
            self.element_cols = element_cols
            self._explicit_element_cols = True
        if process_cols is not None:
            self.process_cols = process_cols
            self._explicit_process_cols = True

        if not self._explicit_element_cols or not self._explicit_process_cols:
            raise ValueError(
                "LLMReviewer.set_dataset requires explicit element_cols and "
                "process_cols from the constructor or set_dataset; use "
                "process_cols=[] when the dataset has no process inputs"
            )
        self.target_cols = [self.target_col] if self.target_col else []

        # Prebuild context text
        self._element_profile = build_dataset_profile(
            df,
            self.element_cols,
            self.process_cols,
            composition_unit=self.composition_unit,
        )
        self._history_stats = build_history_stats(
            df, self.element_cols
        )

        # Expose target metadata without the full-dataset target-value distribution.
        if self.target_col:
            objective_direction = self.objective_direction
            objective_description = self.objective_description
            direction = objective_direction.upper()
            relation = (
                "higher raw numerical values are better"
                if objective_direction == "maximize"
                else "lower raw numerical values are better"
            )
            target_lines = [
                f"- Target column label: {self.target_col}",
                f"- Optimization direction: {direction} the raw numerical target value.",
                f"- Objective utility contract: {relation}; a higher s_target_raw "
                "must always mean a better candidate under this direction.",
            ]
            if objective_description:
                target_lines.append(
                    f"- Scientific objective: {objective_description}"
                )
            self._target_info = "\n".join(target_lines)
        else:
            self._target_info = ""

        n = len(df)
        print(
            f"[LLMReviewer] Dataset loaded: {n} samples, "
            f"{len(self.element_cols)} elements, {len(self.process_cols)} process, "
            f"target={self.target_col}"
        )

    def _candidate_inputs(self, candidate: dict) -> dict:
        """Keep only model-visible input columns, never target or metadata columns."""
        blocked = set(self.target_cols or [])
        if self.target_col:
            blocked.add(self.target_col)
        blocked.update(self.metadata_cols or [])

        allowed: list[str] = []
        seen = set()
        for col in [*self.element_cols, *self.process_cols]:
            if not col or col in blocked or col in seen:
                continue
            if col in candidate:
                allowed.append(col)
                seen.add(col)

        return {col: _jsonable_sample(candidate.get(col)) for col in allowed}


    # ── Single-candidate review ────────────────────────────────────────────────────

    async def review_one(self, candidate: dict, progress_report: str = "") -> dict:
        """Score one candidate inside a loop-local pooled-client session."""
        await self.key_pool.open_session()
        try:
            return await self._review_one_impl(
                candidate,
                progress_report=progress_report,
            )
        finally:
            await self.key_pool.close_session()

    async def _review_one_impl(self, candidate: dict, progress_report: str = "") -> dict:
        """Score one candidate alloy.

        Parameters
        ----------
        candidate : Dictionary of candidate composition and other input values
        progress_report : Optional experiment progress report

        Returns
        -------
        dict with keys: principles, s_target_raw, s_general_plausibility_raw,
        s_phys_raw, s_phys, reason, _refs, _retrieval_queries,
        _retrieval_mode
        """
        candidate_inputs = self._candidate_inputs(candidate)

        candidate_json = json.dumps(candidate_inputs, ensure_ascii=False)

        messages = get_messages(
            material_system=self.material_system,
            element_profile=self._element_profile,
            target_info=self._target_info,
            history_stats=self._history_stats,
            candidate_json=candidate_json,
            progress_report=progress_report,
        )

        # Call the LLM.
        resp, key_alias = await self.key_pool.create_chat_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            timeout=240,
        )
        raw = extract_message_text(resp.choices[0].message)
        if not raw:
            raise ValueError("empty LLM response")

        try:
            result = parse_json_object(raw)
        except ValueError as exc:
            raise ValueError("LLM scoring response must be one valid JSON object") from exc
        _validate_scoring_response(result)

        result.setdefault("principles", [])
        result.setdefault("s_general_plausibility_raw", 0)
        result.setdefault("off_target_notes", [])
        if "s_target_raw" in result or "s_phys_raw" in result:
            score = result.get("s_target_raw", result.get("s_phys_raw"))
            result["s_target_raw"] = score
            result["s_phys_raw"] = score
            result["s_phys"] = score / 100.0
        elif "s_phys" in result:
            result["s_phys"] = float(result["s_phys"])
            result["s_target_raw"] = int(round(result["s_phys"] * 100.0))
            result["s_phys_raw"] = result["s_target_raw"]
        else:
            raise ValueError("LLM scoring response is missing a target score")
        result["_key_alias"] = key_alias

        result["_refs"] = []
        result["_retrieval_queries"] = {}
        result["_retrieval_mode"] = "disabled"
        return result

    async def _select_relevant_anchor_one(self, request: dict) -> dict:
        """Choose one target-blind, mechanism-relevant measured reference."""
        candidate = self._candidate_inputs(dict(request["candidate_input"]))
        raw_options = list(request["anchor_options"])
        if not raw_options:
            raise ValueError("candidate-memory selection requires anchor options")

        options: list[dict] = []
        option_ids: set[str] = set()
        for raw_option in raw_options:
            option_id = str(raw_option["option_id"])
            if option_id in option_ids:
                raise ValueError("candidate-memory option IDs must be unique")
            option_ids.add(option_id)
            options.append({
                "option_id": option_id,
                "candidate_input": self._candidate_inputs(
                    dict(raw_option["candidate_input"])
                ),
            })

        system = (
            "You select one already-measured alloy entry as a persistent local "
            "reference for an unmeasured active-learning candidate. Select by "
            "mechanistic comparability for the registered target: composition, "
            "processing route, heat treatment, test condition, strengthening or "
            "transport mechanism, and likely response regime. You are not given "
            "any measured target values and must not estimate, rank, or request "
            "them. Return strict JSON only."
        )
        user = (
            f"Target column: {self.target_col}\n"
            f"Optimization direction: {self.objective_direction}\n"
            f"Composition basis: {self.composition_unit}\n"
            "Target values are intentionally hidden for every option.\n\n"
            "Unmeasured candidate inputs:\n"
            + json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            + "\n\nX-only prefiltered measured options:\n"
            + json.dumps(options, ensure_ascii=False, sort_keys=True)
            + "\n\nChoose exactly one option. Prefer the entry whose measured "
            "target would be the most scientifically informative comparator, "
            "not the one you imagine has the best target. Return: "
            '{"selected_option_id":"P1","reason":"brief mechanism-based reason"}'
        )
        resp, key_alias = await self.key_pool.create_chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            timeout=240,
        )
        raw = extract_message_text(resp.choices[0].message)
        if not raw:
            raise ValueError("empty candidate-memory selection response")
        try:
            result = parse_json_object(raw)
        except ValueError as exc:
            raise ValueError(
                "candidate-memory selection response is not valid JSON"
            ) from exc
        selected = result.get("selected_option_id")
        if not isinstance(selected, str) or selected not in option_ids:
            raise ValueError(
                "candidate-memory selection returned an unknown option ID"
            )
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("candidate-memory selection requires a non-empty string reason")
        reason = reason.strip()
        return {
            "selected_option_id": selected,
            "reason": reason,
            "_key_alias": key_alias,
        }

    async def select_relevant_anchor_batch(
        self,
        requests: list[dict],
    ) -> list[dict]:
        """Select persistent references with the production retry scheduler."""
        total = len(requests)
        if total == 0:
            return []
        done = 0
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(request: dict) -> dict:
            nonlocal done
            async with sem:
                failures = 0
                while True:
                    try:
                        result = await self._select_relevant_anchor_one(request)
                        break
                    except Exception as exc:
                        failures += 1
                        if _is_permanent_llm_error(exc):
                            print(
                                " [fatal candidate-memory request error after "
                                f"{failures} attempt(s): {exc}]",
                                flush=True,
                            )
                            raise
                        exponent = min(failures - 1, 10)
                        delay = min(
                            self.retry_max_seconds,
                            self.retry_base_seconds * (2 ** exponent),
                        )
                        if delay > 0:
                            delay += random.uniform(0.0, min(1.0, delay * 0.10))
                        print(
                            f" [memory retry {failures} in "
                            f"{delay:.1f}s: {exc}]",
                            flush=True,
                        )
                        await asyncio.sleep(delay)
            async with lock:
                done += 1
                if done % 10 == 0 or done == total:
                    print(
                        f"[MemoryProgress] {done}/{total} "
                        f"({100 * done / total:.1f}%)",
                        flush=True,
                    )
            return result

        await self.key_pool.open_session()
        try:
            results: list[dict] = []
            batch_size = self.rate_limit_batch_size
            for start in range(0, total, batch_size):
                batch = requests[start:start + batch_size]
                results.extend(await _gather_batch(_one(item) for item in batch))
                if start + batch_size < total and self.rate_limit_sleep_seconds > 0:
                    print(
                        f"[MemoryRateLimit] batch {start // batch_size + 1} done; "
                        f"sleeping {self.rate_limit_sleep_seconds:g}s before next "
                        f"{batch_size} requests",
                        flush=True,
                    )
                    await asyncio.sleep(self.rate_limit_sleep_seconds)
            return results
        finally:
            await self.key_pool.close_session()

    # ── Batch review ────────────────────────────────────────────────────

    async def review_batch(
        self,
        candidates: list[dict],
        progress_report: str | list[str] = "",
    ) -> list[dict]:
        total = len(candidates)
        if isinstance(progress_report, str):
            progress_reports = [progress_report] * total
        else:
            progress_reports = [str(value) for value in progress_report]
            if len(progress_reports) != total:
                raise ValueError(
                    "progress_report list must match the candidate count"
                )
        done = 0
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(c, candidate_progress_report: str):
            nonlocal done
            async with sem:
                failures = 0
                while True:
                    try:
                        result = await self.review_one(
                            c,
                            progress_report=candidate_progress_report,
                        )
                        break
                    except Exception as e:
                        failures += 1
                        if _is_permanent_llm_error(e):
                            print(
                                f" [fatal LLM configuration/request error after "
                                f"{failures} attempt(s): {e}]",
                                flush=True,
                            )
                            raise

                        exponent = min(failures - 1, 10)
                        delay = min(
                            self.retry_max_seconds,
                            self.retry_base_seconds * (2 ** exponent),
                        )
                        if delay > 0:
                            delay += random.uniform(0.0, min(1.0, delay * 0.10))
                        print(
                            f" [retry {failures} in {delay:.1f}s: {e}]",
                            flush=True,
                        )
                        await asyncio.sleep(delay)
            async with lock:
                done += 1
                if done % 10 == 0 or done == total:
                    print(
                        f"[Progress] {done}/{total} ({100*done/total:.1f}%)",
                        flush=True,
                    )
                print(
                    f"  [{done:>3}] "
                    f"phys={result['s_phys']:.2f} "
                    f"key={result.get('_key_alias', 'unknown')} "
                    f"| {str(result.get('reason', ''))[:80]}",
                    flush=True,
                )
            return result

        await self.key_pool.open_session()
        try:
            results = []
            batch_size = self.rate_limit_batch_size
            for start in range(0, total, batch_size):
                batch = candidates[start:start + batch_size]
                batch_reports = progress_reports[start:start + batch_size]
                results.extend(await _gather_batch(
                    _one(candidate, report)
                    for candidate, report in zip(batch, batch_reports)
                ))

                if start + batch_size < total and self.rate_limit_sleep_seconds > 0:
                    print(
                        f"[RateLimit] batch {start // batch_size + 1} done; "
                        f"sleeping {self.rate_limit_sleep_seconds:g}s before next "
                        f"{batch_size} requests",
                        flush=True,
                    )
                    await asyncio.sleep(self.rate_limit_sleep_seconds)
            return results
        finally:
            await self.key_pool.close_session()
