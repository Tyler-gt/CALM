"""Single-objective CALM knowledge scoring with measured feedback and no retrieval.

Use set_dataset() to supply input-only context, review_batch() to score candidates,
and select_relevant_anchor_batch() to choose target-blind persistent references.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time

import pandas as pd
from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    UnprocessableEntityError,
)

from common.llm.prompts import (
    build_dataset_profile,
    build_history_stats,
    get_messages,
)
from common.llm.response_utils import extract_message_text
from common.llm.config import get_api_base_url, get_llm_model, validate_api_base_url
from common.llm.key_pool import (
    AsyncUSTCKeyPool,
    NoAvailableAPIKeyError,
    require_ustc_api_keys,
)


NO_RETRIEVAL_PROMPT_PLACEHOLDER = (
    "[NO-RETRIEVAL CONTROL] Literature retrieval is disabled for this run. "
    "No paper snippets or source-derived target values were supplied. "
    "Use only the dataset profile, candidate inputs, measured active-learning "
    "feedback, and general prior knowledge; do not imply that a paper was "
    "retrieved or cited."
)

_COMPOSITION_UNIT_LABELS = {
    "wt.%": "weight percent",
    "at.%": "atomic percent",
}


def _validate_retrieval_mode(retrieval_mode: str) -> str:
    if str(retrieval_mode).strip() != "disabled":
        raise ValueError("CALM uses no literature retrieval; retrieval_mode must be 'disabled'")
    return "disabled"


def _validate_composition_unit(composition_unit: str) -> str:
    unit = str(composition_unit).strip()
    if unit not in _COMPOSITION_UNIT_LABELS:
        raise ValueError(
            "composition_unit must be exactly 'wt.%' or 'at.%', got "
            f"{composition_unit!r}"
        )
    return unit


_PERMANENT_LLM_ERRORS = (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
    NoAvailableAPIKeyError,
)


def _is_permanent_llm_error(exc: Exception) -> bool:
    """Errors that cannot become successful without changing configuration."""
    return isinstance(exc, _PERMANENT_LLM_ERRORS)

# ═══════════════════════════════════════════════════════════════════
# Column and input helpers
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

_COLUMN_ROLE_PROMPT = """\
You are classifying columns in an alloy dataset for a general active-learning
pipeline. Use semantic judgment from the column names and sample values; do not
apply a hard-coded keyword list or alloy-system-specific assumptions.

Classify every input column into exactly one of:
- element_cols: direct alloy composition amount columns for chemical elements.
- process_cols: processing, heat treatment, deformation, test condition, or
  other experimental/input variables that should be usable by the model.
- target_cols: measured objective/property columns. The intended target column
  is "{target_col}", but classify any other property columns as target_cols too.
- metadata_cols: identifiers, references, comments, alloy names, free-text
  formula/composition descriptions, derived audit fields, or columns that should
  not be used as model inputs.

Return only a JSON object with keys:
"element_cols", "process_cols", "target_cols", "metadata_cols".
Each value must be a list of exact column names copied from the input.
Every input column must appear exactly once across the four lists.

Column records:
{column_records}
"""


def _jsonable_sample(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _column_records(
    df: pd.DataFrame,
    target_col: str = "",
    max_values: int = 5,
) -> str:
    records = []
    for col in df.columns:
        if target_col and str(col) == str(target_col):
            values = ["<known target column values hidden>"]
        else:
            values = [_jsonable_sample(v) for v in df[col].dropna().head(max_values)]
        records.append({"name": str(col), "sample_values": values})
    return json.dumps(records, ensure_ascii=False, default=str, indent=2)


def _dedupe_known_columns(values, columns: list[str]) -> list[str]:
    known = set(columns)
    out = []
    seen = set()
    for value in values or []:
        col = str(value)
        if col in known and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _classify_columns_with_llm(
    df: pd.DataFrame,
    target_col: str,
    *,
    model: str,
    base_url: str | None = None,
) -> dict[str, list[str]]:
    """Classify all columns with the LLM; no regex/keyword role detection."""
    columns = [str(c) for c in df.columns]
    prompt = _COLUMN_ROLE_PROMPT.format(
        target_col=target_col or "",
        column_records=_column_records(df, target_col=target_col),
    )

    client = OpenAI(
        api_key=require_ustc_api_keys(1)[0],
        base_url=validate_api_base_url(
            get_api_base_url() if base_url is None else base_url
        ),
    )

    last_err: Exception | None = None
    result = None
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
            break
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                delay = 2 ** attempt
                print(f"[LLMReviewer] Column classification retry {attempt + 1}/3 in {delay}s: {exc}")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"LLM column classification failed after 3 attempts: {last_err}"
                ) from last_err

    roles = {
        "element_cols": _dedupe_known_columns(result.get("element_cols", []), columns),
        "process_cols": _dedupe_known_columns(result.get("process_cols", []), columns),
        "target_cols": _dedupe_known_columns(result.get("target_cols", []), columns),
        "metadata_cols": _dedupe_known_columns(result.get("metadata_cols", []), columns),
    }

    assigned = set()
    clean_roles = {key: [] for key in roles}
    for key in ("element_cols", "process_cols", "target_cols", "metadata_cols"):
        for col in roles[key]:
            if col not in assigned:
                clean_roles[key].append(col)
                assigned.add(col)

    missing = [col for col in columns if col not in assigned]
    if missing:
        clean_roles["metadata_cols"].extend(missing)

    if target_col and target_col in columns:
        for key in ("element_cols", "process_cols", "metadata_cols"):
            clean_roles[key] = [c for c in clean_roles[key] if c != target_col]
        if target_col not in clean_roles["target_cols"]:
            clean_roles["target_cols"].insert(0, target_col)

    print(
        "[LLMReviewer] LLM column roles: "
        f"{len(clean_roles['element_cols'])} elements, "
        f"{len(clean_roles['process_cols'])} process, "
        f"{len(clean_roles['target_cols'])} targets, "
        f"{len(clean_roles['metadata_cols'])} metadata"
    )
    return clean_roles


# ═══════════════════════════════════════════════════════════════════
# Reviewer
# ═══════════════════════════════════════════════════════════════════

class LLMReviewer:
    """多体系合金 LLM 评审器.

    每次切换数据集时调用 set_dataset() 更新 profile/统计;
    然后 review_one() 或 review_batch() 对候选合金打分.
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
        retry_until_success: bool = False,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 60.0,
        strict_scoring: bool = False,
        api_keys: list[str] | None = None,
        min_api_keys: int = 1,
        per_key_concurrency: int = 4,
        key_pool: AsyncUSTCKeyPool | None = None,
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
        self.retry_until_success = bool(retry_until_success)
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(retry_max_seconds),
        )
        self.strict_scoring = bool(strict_scoring)
        self.retrieval_mode = "disabled"

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
                else require_ustc_api_keys(min_api_keys)
            )
            if len(set(resolved_keys)) < int(min_api_keys):
                raise RuntimeError(
                    f"LLMReviewer requires {int(min_api_keys)} independent API keys"
                )
            self.key_pool = AsyncUSTCKeyPool(
                resolved_keys,
                per_key_concurrency=per_key_concurrency,
            )
        self.base_url = self.key_pool.base_url
        # 预计算的 profile 文本
        self._element_profile: str = ""
        self._process_profile: str = ""
        self._history_stats: str = ""
        self._target_info: str = ""

        # 列定义
        self.element_cols: list[str] = element_cols or []
        self.process_cols: list[str] = process_cols or []
        self.target_cols: list[str] = [target_col] if target_col else []
        self.metadata_cols: list[str] = []
        self._explicit_element_cols = element_cols is not None
        self._explicit_process_cols = process_cols is not None

    # ── 数据集注册 ──────────────────────────────────────────────────

    def set_dataset(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "",
        element_cols: list[str] | None = None,
        process_cols: list[str] | None = None,
        composition_unit: str | None = None,
    ) -> None:
        """注册当前数据集, 自动提取 profile 和统计.

        调用此方法后, review_one() 的 user message 将自动使用该数据集的信息.
        """
        self.target_col = target_col or self.target_col
        if composition_unit is not None:
            self.composition_unit = _validate_composition_unit(composition_unit)
        elif not hasattr(self, "composition_unit"):
            self.composition_unit = "wt.%"

        # 自动列分类: 未显式传入时, 将全部列名和样本值交给 LLM 判断.
        if element_cols is not None:
            self.element_cols = element_cols
            self._explicit_element_cols = True
        if process_cols is not None:
            self.process_cols = process_cols
            self._explicit_process_cols = True

        need_llm_roles = (
            (element_cols is None and not self._explicit_element_cols)
            or (process_cols is None and not self._explicit_process_cols)
        )
        if need_llm_roles:
            roles = _classify_columns_with_llm(
                df,
                self.target_col,
                model=self.model,
                base_url=self.base_url,
            )
            if element_cols is None and not self._explicit_element_cols:
                self.element_cols = roles["element_cols"]
            if process_cols is None and not self._explicit_process_cols:
                self.process_cols = roles["process_cols"]
            self.target_cols = roles["target_cols"]
            self.metadata_cols = roles["metadata_cols"]
        else:
            self.target_cols = [self.target_col] if self.target_col else []

        # 预构建文本
        self._element_profile = build_dataset_profile(
            df,
            self.element_cols,
            self.process_cols,
            self.target_col,
            composition_unit=self.composition_unit,
        )
        self._history_stats = build_history_stats(
            df, self.element_cols, self.target_col
        )

        # 目标信息只暴露列名, 不暴露全数据目标值分布.
        if self.target_col:
            # Production construction always supplies the registered direction.
            # A few focused diagnostics use ``object.__new__``, so retain a
            # maximize default for those deliberately partial reviewer objects.
            objective_direction = getattr(
                self, "objective_direction", "maximize"
            )
            objective_description = getattr(
                self, "objective_description", ""
            )
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


    # ── 单条评审 ────────────────────────────────────────────────────

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
        """对一条候选合金打分.

        Parameters
        ----------
        candidate : 候选成分字典
        progress_report : 实验进展报告 (可选)

        Returns
        -------
        dict with keys: principles, s_target_raw, s_general_plausibility_raw,
        s_phys_raw, s_phys, reason, _refs, _retrieval_queries,
        _retrieval_mode
        """
        candidate_inputs = self._candidate_inputs(candidate)

        # Keep the original no-retrieval prompt content used by the paper.
        _validate_retrieval_mode(self.retrieval_mode)
        snippets = NO_RETRIEVAL_PROMPT_PLACEHOLDER

        candidate_json = json.dumps(candidate_inputs, ensure_ascii=False)

        messages = get_messages(
            material_system=self.material_system,
            element_profile=self._element_profile,
            process_profile=self._process_profile,
            target_info=self._target_info,
            history_stats=self._history_stats,
            candidate_json=candidate_json,
            reference_snippets=snippets,
            progress_report=progress_report,
        )

        # ③ LLM 调用
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

        # ④ 解析
        fallback = {
            "principles": [],
            "s_target_raw": 0,
            "s_general_plausibility_raw": 0,
            "s_phys_raw": 0,
            "s_phys": 0.0,
            "off_target_notes": [],
            "reason": "parse error",
        }
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            try:
                decoder = json.JSONDecoder()
                result, _ = decoder.raw_decode(raw)
            except json.JSONDecodeError:
                if self.strict_scoring:
                    raise ValueError("LLM scoring response is not valid JSON")
                result = fallback

        if not isinstance(result, dict):
            raise ValueError("LLM scoring response must be a JSON object")

        result.setdefault("principles", [])
        has_target_score = "s_target_raw" in result
        has_raw_score = "s_phys_raw" in result
        has_normalized_score = "s_phys" in result
        result.setdefault("s_target_raw", fallback["s_target_raw"])
        result.setdefault(
            "s_general_plausibility_raw",
            fallback["s_general_plausibility_raw"],
        )
        result.setdefault("s_phys_raw", fallback["s_phys_raw"])
        result.setdefault("s_phys", fallback["s_phys"])
        result.setdefault("off_target_notes", fallback["off_target_notes"])

        raw_score = (
            result.get("s_target_raw")
            if has_target_score
            else result.get("s_phys_raw")
        )
        if has_target_score or has_raw_score:
            try:
                raw_float = float(raw_score)
                if not math.isfinite(raw_float):
                    raise ValueError("raw score is not finite")
                if 0.0 <= raw_float <= 1.0:
                    raw_float *= 100.0
                if self.strict_scoring and not 0.0 <= raw_float <= 100.0:
                    raise ValueError("raw score is outside [0, 100]")
                raw_float = min(max(raw_float, 0.0), 100.0)
                result["s_target_raw"] = int(round(raw_float))
                result["s_phys_raw"] = result["s_target_raw"]
                result["s_phys"] = result["s_target_raw"] / 100.0
            except (ValueError, TypeError):
                if self.strict_scoring:
                    raise ValueError("LLM scoring response contains an invalid raw score")
                has_target_score = False
                has_raw_score = False

        if not has_target_score and not has_raw_score:
            if self.strict_scoring and not has_normalized_score:
                raise ValueError("LLM scoring response is missing a target score")
            try:
                score = float(result["s_phys"])
                if not math.isfinite(score):
                    raise ValueError("normalized score is not finite")
                if self.strict_scoring and not 0.0 <= score <= 1.0:
                    raise ValueError("normalized score is outside [0, 1]")
                score = min(max(score, 0.0), 1.0)
                result["s_phys"] = score
                result["s_target_raw"] = int(round(score * 100.0))
                result["s_phys_raw"] = result["s_target_raw"]
            except (ValueError, TypeError):
                if self.strict_scoring:
                    raise ValueError(
                        "LLM scoring response contains an invalid normalized score"
                    )
                result["s_target_raw"] = fallback["s_target_raw"]
                result["s_phys_raw"] = fallback["s_phys_raw"]
                result["s_phys"] = fallback["s_phys"]

        try:
            plaus = float(result["s_general_plausibility_raw"])
            if 0.0 <= plaus <= 1.0:
                plaus *= 100.0
            result["s_general_plausibility_raw"] = int(round(min(max(plaus, 0.0), 100.0)))
        except (ValueError, TypeError):
            result["s_general_plausibility_raw"] = result["s_target_raw"]
        result.setdefault("reason", fallback["reason"])
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
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "candidate-memory selection response is not valid JSON"
            ) from exc
        if not isinstance(result, dict):
            raise ValueError("candidate-memory selection must be a JSON object")
        selected = str(result.get("selected_option_id", ""))
        if selected not in option_ids:
            raise ValueError(
                "candidate-memory selection returned an unknown option ID"
            )
        reason = str(result.get("reason", "")).strip()
        if not reason:
            raise ValueError("candidate-memory selection requires a reason")
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
                        if not self.retry_until_success and failures >= 4:
                            raise
                        exponent = min(failures - 1, 10)
                        delay = min(
                            self.retry_max_seconds,
                            self.retry_base_seconds * (2 ** exponent),
                        )
                        if delay > 0:
                            delay += random.uniform(0.0, min(1.0, delay * 0.10))
                        retry_limit = "∞" if self.retry_until_success else "3"
                        print(
                            f" [memory retry {failures}/{retry_limit} in "
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
                results.extend(await asyncio.gather(*(_one(item) for item in batch)))
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

    # ── 批量评审 ────────────────────────────────────────────────────

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
                        if not self.retry_until_success and failures >= 4:
                            raise

                        exponent = min(failures - 1, 10)
                        delay = min(
                            self.retry_max_seconds,
                            self.retry_base_seconds * (2 ** exponent),
                        )
                        if delay > 0:
                            delay += random.uniform(0.0, min(1.0, delay * 0.10))
                        retry_limit = "∞" if self.retry_until_success else "3"
                        print(
                            f" [retry {failures}/{retry_limit} in {delay:.1f}s: {e}]",
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
                results.extend(await asyncio.gather(*(
                    _one(candidate, report)
                    for candidate, report in zip(batch, batch_reports)
                )))

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
