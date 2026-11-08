"""LLM 评审客户端: HyDE 查询改写 + RAG 文献检索 + 冶金学评分 — 多体系通用版.

检索流程 (HyDE — Hypothetical Document Embeddings):
  候选成分+原始工艺值 → LLM 重写为自然语言段落 → text-embedding-v4 → 查论文 chunks
  消除元素裸数和带有正确 at.%/wt.% 语义的文献表述之间的嵌入鸿沟.

用法:
    reviewer = LLMReviewer(material_system="alloy")
    reviewer.set_dataset(df, target_col="target_property")
    scores = await reviewer.review_batch(candidates)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
import time
from typing import Optional

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
from common.rag.retrieve import Retriever
from common.rag.process_normalization import render_process_condition


RAG_SECTION_FILTER = (
    "results",
    "discussion",
    "conclusion",
    "experimental",
    "methods",
    "body",
)
RAG_MAX_PER_SOURCE = 2
RAG_SNIPPET_MAX_CHARS = 1200
RETRIEVAL_MODES = frozenset({"full", "disabled"})
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
    mode = str(retrieval_mode).strip()
    if mode not in RETRIEVAL_MODES:
        allowed = ", ".join(sorted(RETRIEVAL_MODES))
        raise ValueError(
            f"retrieval_mode must be one of {{{allowed}}}, got "
            f"{retrieval_mode!r}"
        )
    return mode


def _validate_composition_unit(composition_unit: str) -> str:
    unit = str(composition_unit).strip()
    if unit not in _COMPOSITION_UNIT_LABELS:
        raise ValueError(
            "composition_unit must be exactly 'wt.%' or 'at.%', got "
            f"{composition_unit!r}"
        )
    return unit


def _composition_unit_search_terms(composition_unit: str) -> str:
    unit = _validate_composition_unit(composition_unit)
    return f"{unit} {_COMPOSITION_UNIT_LABELS[unit]}"


class RAGEvidenceError(RuntimeError):
    """A deterministic RAG evidence failure that retries cannot repair."""


_PERMANENT_LLM_ERRORS = (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
    NoAvailableAPIKeyError,
    RAGEvidenceError,
)


def _is_permanent_llm_error(exc: Exception) -> bool:
    """Errors that cannot become successful without changing configuration."""
    return isinstance(exc, _PERMANENT_LLM_ERRORS)

# ═══════════════════════════════════════════════════════════════════
# HyDE 查询生成 — 体系无关
# ═══════════════════════════════════════════════════════════════════

def _target_label(target_col: str) -> str:
    """Format the target column name without interpreting its meaning."""
    label = str(target_col or "").strip()
    label = label.replace("_", " ").replace("-", " ")
    label = label.replace("(", " ").replace(")", " ")
    label = label.replace("[", " ").replace("]", " ")
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    label = " ".join(label.split())
    return label or "target property"


def _build_hyde_prompt(
    material_system: str,
    target_col: str = "",
    prompt_profile: str = "general",
    composition_unit: str = "wt.%",
) -> str:
    """Build a generic HyDE prompt with no alloy-family priors."""
    target_label = _target_label(target_col)
    composition_unit = _validate_composition_unit(composition_unit)
    unit_label = _COMPOSITION_UNIT_LABELS[composition_unit]
    if prompt_profile == "multi_objective_pareto":
        return (
            "Write one dense retrieval paragraph in the style of a materials "
            "science results summary for multi-objective alloy design. Do not "
            "assume alloy-specific grades, phases, or mechanisms unless they "
            "are directly implied by the given composition, processing fields, "
            "target objective label, or retrieved evidence.\n\n"
            f"Material system label: {material_system or 'alloy'}\n"
            f"Target objective label: {target_label}\n"
            f"Alloy composition ({composition_unit}; {unit_label}):\n"
            "{composition}\n\n"
            f"Interpret every elemental value as {unit_label} ({composition_unit}); "
            "do not reinterpret or convert it to the other composition basis.\n\n"
            "Processing/input fields (raw values; units may be encoded in "
            "the field names):\n"
            "{processing}\n\n"
            "Infer the objective semantics yourself from the label as written. "
            "Include the nonzero elements, the supplied processing conditions "
            "and their values, all objectives, and neutral "
            "composition-processing-property language about Pareto trade-offs, "
            "frontier candidates, and cross-objective synergies suitable for "
            "RAG retrieval. Do not score the alloy. No preamble, no labels:"
        )
    return (
        "Write one dense retrieval paragraph in the style of a materials "
        "science results summary. Do not assume alloy-specific grades, phases, "
        "or mechanisms unless they are directly implied by the given composition, "
        "processing fields, target column label, or retrieved evidence.\n\n"
        f"Material system label: {material_system or 'alloy'}\n"
        f"Target column label: {target_label}\n"
        f"Alloy composition ({composition_unit}; {unit_label}):\n"
        "{composition}\n\n"
        f"Interpret every elemental value as {unit_label} ({composition_unit}); "
        "do not reinterpret or convert it to the other composition basis.\n\n"
        "Processing/input fields (raw values; units may be encoded in the "
        "field names):\n"
        "{processing}\n\n"
        "Infer the target-column semantics yourself from the label as written. "
        "Do not rely on any hard-coded mapping from abbreviations to properties. "
        "The paragraph should include the nonzero elements, target column label, "
        "the supplied processing conditions and their exact values, and neutral "
        "composition-processing-property language suitable for RAG "
        "retrieval. Do not score the alloy. No preamble, no labels:"
    )


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


def _canonical_query_value(value):
    """Return a stable scalar for process-aware query text and cache keys."""
    sample = _jsonable_sample(value)
    if sample is None:
        return None
    if isinstance(sample, bool):
        return sample
    if isinstance(sample, (int, float)):
        numeric = float(sample)
        if not math.isfinite(numeric):
            return None
        if numeric.is_integer():
            return int(numeric)
        return float(f"{numeric:.10g}")
    text = str(sample).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    return text[:160]


def _format_query_value(value) -> str | None:
    canonical = _canonical_query_value(value)
    if canonical is None:
        return None
    if isinstance(canonical, bool):
        return "true" if canonical else "false"
    return str(canonical)


def _process_query_items(
    candidate: dict,
    process_cols: list[str],
    binary_indicator_cols: set[str] | None = None,
    binary_state_cols: set[str] | None = None,
) -> list[tuple[str, object, str, str | None]]:
    items: list[tuple[str, object, str, str | None]] = []
    binary_indicator_cols = binary_indicator_cols or set()
    binary_state_cols = binary_state_cols or set()
    for col in process_cols:
        if col not in candidate:
            continue
        canonical = _canonical_query_value(candidate.get(col))
        if canonical is None:
            continue
        formatted = _format_query_value(canonical)
        if formatted is None:
            continue
        indicator_state = None
        if col in binary_indicator_cols and canonical in {0, 1}:
            indicator_state = "active" if canonical == 1 else "inactive"
        elif col in binary_state_cols and canonical in {0, 1}:
            # Ordinary Boolean process fields are not mutually exclusive
            # one-hot columns.  Their zero state is scientifically meaningful
            # (for example Cu ``Aging=0`` means aging was not applied), so it
            # must remain in both the retrieval query and HyDE cache key.
            indicator_state = "applied" if canonical == 1 else "not_applied"
        items.append((col, canonical, formatted, indicator_state))
    return items


def _render_process_items(
    items: list[tuple[str, object, str, str | None]],
) -> list[tuple[str, str, str]]:
    """Return ``(raw field, raw text, normalized search text)`` triples."""
    rendered: list[tuple[str, str, str]] = []
    for col, value, value_text, indicator_state in items:
        search_text = render_process_condition(
            col,
            value,
            indicator_state=indicator_state,
        )
        if search_text:
            rendered.append((col, value_text, search_text))
    return rendered


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


_SNIPPET_GENERIC_WORDS = {
    "active",
    "alloy",
    "approximately",
    "condition",
    "conditions",
    "input",
    "processing",
    "property",
    "raw",
    "relationship",
}
_SNIPPET_PROCESS_WORDS = {
    "ageing",
    "aging",
    "annealing",
    "cooled",
    "cooling",
    "extrusion",
    "furnace",
    "homogenization",
    "preheat",
    "quench",
    "quenched",
    "quenching",
    "reduction",
    "solution",
    "strain",
    "temperature",
    "thermomechanical",
    "treatment",
}
_SNIPPET_COMPOUND_WEIGHT = 12
_SNIPPET_QUERY_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_SNIPPET_QUERY_UNIT = (
    r"(?:met(?:er|re)s?\s+per\s+minute|m\s*/\s*min|"
    r"per\s+second|s\s*\^\s*-\s*1|"
    r"degrees?\s+c(?:elsius)?|celsius|°\s*c|c|"
    r"kelvin|k|hours?|hrs?|h|minutes?|mins?|min|"
    r"percentage|percent|%)"
)


def _snippet_numeric_pattern(raw: str) -> str | None:
    """Return a complete, sign-preserving regex for one query number."""
    canonical = raw.rstrip("0").rstrip(".") if "." in raw else raw
    if not canonical or canonical in {"+", "-"}:
        return None
    sign = ""
    magnitude = canonical
    if canonical[0] in {"+", "-"}:
        sign, magnitude = canonical[0], canonical[1:]
    if not magnitude:
        return None

    # Match the complete signed numeric token.  A positive query must not
    # match the magnitude inside a negative number.
    prefix = r"(?<![\d.])" if sign else r"(?<![\d.+-])"
    signed_magnitude = re.escape(sign + magnitude)
    if "." in magnitude:
        numeric_body = signed_magnitude + r"0*"
    else:
        numeric_body = signed_magnitude + r"(?:\.0+)?"
    return prefix + numeric_body + r"(?![\d.])"


def _snippet_compound_unit_pattern(raw_unit: str) -> str | None:
    """Map a query unit to compatible adjacent-unit spellings in papers."""
    key = re.sub(r"\s+", "", raw_unit.casefold())
    if key in {"k", "kelvin"}:
        return r"(?:K|kelvin)\b"
    if key in {"c", "°c", "celsius"} or key.startswith("degree"):
        return r"(?:°\s*C|C|degrees?\s+C(?:elsius)?|celsius)\b"
    if key in {"h", "hr", "hrs", "hour", "hours"}:
        return r"(?:h|hr|hrs|hour|hours)\b"
    if key in {"min", "mins", "minute", "minutes"}:
        return r"(?:min|mins|minute|minutes)\b"
    if "/" in key or "perminute" in key:
        return r"(?:m\s*/\s*min|met(?:er|re)s?\s+per\s+minute)\b"
    if key.startswith("s^") or key == "persecond":
        return r"(?:s\s*\^?\s*-\s*1|per\s+second)\b"
    if key in {"%", "percent", "percentage"}:
        return r"(?:%|percent\b|percentage\b)"
    return None


def _snippet_anchors(query_text: str) -> list[tuple[re.Pattern, int]]:
    """Compile process/numeric anchors used to center an evidence window."""
    query_text = str(query_text or "").replace("−", "-").replace("–", "-")
    anchors: list[tuple[re.Pattern, int]] = []
    seen: set[tuple[str, int]] = set()

    # Prefer a process value together with its immediately adjacent unit.
    # This prevents, for example, a query for ``300 °C`` from centering the
    # evidence window on an unrelated ``300 MPa`` strength value.  Bare
    # numeric anchors remain as a lower-priority fallback for papers that omit
    # or OCR-corrupt the unit.
    compound_re = re.compile(
        rf"(?<![A-Za-z0-9_.^])(?P<number>{_SNIPPET_QUERY_NUMBER})\s*"
        rf"(?P<unit>{_SNIPPET_QUERY_UNIT})(?![A-Za-z])",
        re.IGNORECASE,
    )
    for match in compound_re.finditer(query_text):
        number_pattern = _snippet_numeric_pattern(match.group("number"))
        unit_pattern = _snippet_compound_unit_pattern(match.group("unit"))
        if number_pattern is None or unit_pattern is None:
            continue
        pattern_text = number_pattern + r"\s*" + unit_pattern
        key = (pattern_text.casefold(), _SNIPPET_COMPOUND_WEIGHT)
        if key not in seen:
            anchors.append(
                (
                    re.compile(pattern_text, re.IGNORECASE),
                    _SNIPPET_COMPOUND_WEIGHT,
                )
            )
            seen.add(key)

    for raw in re.findall(
        r"(?<![A-Za-z0-9_.^+\-])[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?![A-Za-z0-9_.])",
        query_text,
    ):
        pattern_text = _snippet_numeric_pattern(raw)
        if pattern_text is None:
            continue
        key = (pattern_text, 6)
        if key not in seen:
            anchors.append((re.compile(pattern_text, re.IGNORECASE), 6))
            seen.add(key)

    for word in re.findall(r"[A-Za-z]{4,}", query_text):
        lowered = word.casefold()
        if lowered in _SNIPPET_GENERIC_WORDS:
            continue
        weight = 4 if lowered in _SNIPPET_PROCESS_WORDS else 1
        pattern_text = rf"\b{re.escape(word)}\b"
        key = (pattern_text.casefold(), weight)
        if key not in seen:
            anchors.append((re.compile(pattern_text, re.IGNORECASE), weight))
            seen.add(key)
    return anchors


def _query_centered_snippet(
    text: str,
    query_text: str,
    max_chars: int = RAG_SNIPPET_MAX_CHARS,
) -> str:
    """Extract a window centered on the densest process/numeric query matches."""
    clean = " ".join(
        str(text or "")
        .replace("\n", " ")
        .replace("−", "-")
        .replace("–", "-")
        .split()
    )
    max_chars = max(80, int(max_chars))
    if len(clean) <= max_chars:
        return clean

    occurrences: list[tuple[int, int, int]] = []
    for pattern, weight in _snippet_anchors(query_text):
        for match in pattern.finditer(clean):
            occurrences.append((match.start(), match.end(), weight))

    if occurrences:
        best_start = 0
        best_score = (-1, -1, -1, 0)
        for anchor_start, anchor_end, _weight in occurrences:
            center = (anchor_start + anchor_end) // 2
            start = max(0, min(center - max_chars // 2, len(clean) - max_chars))
            end = start + max_chars
            inside = [item for item in occurrences if item[0] < end and item[1] > start]
            score = (
                sum(1 for item in inside if item[2] >= _SNIPPET_COMPOUND_WEIGHT),
                sum(item[2] for item in inside),
                sum(1 for item in inside if item[2] >= 4),
                -start,
            )
            if score > best_score:
                best_score = score
                best_start = start
    else:
        best_start = 0

    prefix = "…" if best_start > 0 else ""
    suffix_room = 1 if best_start + max_chars < len(clean) else 0
    content_budget = max_chars - len(prefix) - suffix_room
    end = min(len(clean), best_start + content_budget)
    suffix = "…" if end < len(clean) else ""
    return prefix + clean[best_start:end].strip() + suffix


def _prepare_hit_snippets(
    hits: list[dict],
    query_text: str,
    *,
    max_chars_per_hit: int = RAG_SNIPPET_MAX_CHARS,
) -> list[dict]:
    prepared: list[dict] = []
    for hit in hits:
        copied = dict(hit)
        copied["_snippet"] = _query_centered_snippet(
            hit.get("text", ""),
            query_text,
            max_chars=max_chars_per_hit,
        )
        prepared.append(copied)
    return prepared


def _format_snippets(
    hits: list[dict],
    max_chars_per_hit: int = RAG_SNIPPET_MAX_CHARS,
) -> str:
    if not hits:
        return "(no relevant literature snippets)"
    lines = []
    for i, h in enumerate(hits, 1):
        title = (h.get("paper_title") or h.get("source_file") or "").strip()
        section = h.get("section", "")
        score = h.get("score", 0.0)
        text = h.get("_snippet")
        if text is None:
            text = _query_centered_snippet(
                h.get("text", ""),
                "",
                max_chars=max_chars_per_hit,
            )
        if len(text) > max_chars_per_hit:
            text = text[:max_chars_per_hit] + "…"
        lines.append(
            f"[{i}] ({section}, sim={score:.3f}) {title}\n    {text}"
        )
    return "\n\n".join(lines)


def _build_evidence_refs(hits: list[dict], *, strict: bool) -> list[dict]:
    """Build replayable evidence records from the exact snippets shown to the LLM."""
    refs: list[dict] = []
    seen_chunk_ids: set[str] = set()
    for rank, hit in enumerate(hits, 1):
        chunk_id = str(hit.get("chunk_id", "") or "").strip()
        snippet = str(hit.get("_snippet", "") or "").strip()
        full_text = str(hit.get("text", "") or "")
        text_sha256 = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        if strict:
            if not chunk_id:
                raise RAGEvidenceError(
                    "RAG hit is missing a chunk_id in strict scoring mode"
                )
            if chunk_id in seen_chunk_ids:
                raise RAGEvidenceError(
                    f"duplicate RAG chunk_id in strict scoring mode: {chunk_id}"
                )
            if not snippet:
                raise RAGEvidenceError(
                    f"RAG hit {chunk_id} has no auditable snippet in strict scoring mode"
                )
        seen_chunk_ids.add(chunk_id)

        retrieval_score = float(hit.get("score", 0.0) or 0.0)
        fused_score = float(
            hit.get("fused_score", retrieval_score) or 0.0
        )
        refs.append(
            {
                "retrieval_rank": int(rank),
                "chunk_id": chunk_id,
                "chunk_idx": hit.get("chunk_idx"),
                "paper_title": hit.get("paper_title", ""),
                "source_file": hit.get("source_file", ""),
                "section": hit.get("section", ""),
                "text_sha256": text_sha256,
                "snippet": snippet,
                "dense_score": float(hit.get("dense_score", 0.0) or 0.0),
                "bm25_score": float(hit.get("bm25_score", 0.0) or 0.0),
                "fused_score": fused_score,
                # Final ranking score after section weighting; retained under
                # the backward-compatible name used by evidence readers.
                "retrieval_score": retrieval_score,
            }
        )
    return refs


# ═══════════════════════════════════════════════════════════════════
# Reviewer
# ═══════════════════════════════════════════════════════════════════

# 全局共享 Retriever, 避免重复加载
_RETRIEVER: Optional[Retriever] = None


def _get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


class LLMReviewer:
    """多体系合金 LLM 评审器.

    每次切换数据集时调用 set_dataset() 更新 profile/统计;
    然后 review_one() 或 review_batch() 对候选合金打分.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        top_k: int = 5,
        use_hyde: bool = True,
        material_system: str = "alloy",
        composition_unit: str = "wt.%",
        element_cols: list[str] | None = None,
        process_cols: list[str] | None = None,
        target_col: str = "",
        objective_direction: str = "maximize",
        objective_description: str = "",
        prompt_profile: str = "general",
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
        retrieval_mode: str = "full",
    ) -> None:
        self.model = get_llm_model() if model is None else model
        self.temperature = temperature
        self.top_k = top_k
        self.use_hyde = use_hyde
        self.material_system = material_system
        self.composition_unit = _validate_composition_unit(composition_unit)
        self.target_col = target_col
        self.objective_direction = str(objective_direction).strip().lower()
        if self.objective_direction not in {"maximize", "minimize"}:
            raise ValueError(
                "objective_direction must be exactly 'maximize' or 'minimize'"
            )
        self.objective_description = str(objective_description).strip()
        self.prompt_profile = prompt_profile
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
        self.retrieval_mode = _validate_retrieval_mode(retrieval_mode)

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
        # The no-RAG control must not load embeddings/BM25 or instantiate the
        # shared Retriever at all.
        self.retriever = (
            _get_retriever()
            if self.retrieval_mode == "full"
            else None
        )
        self._hyde_cache: dict[str, str] = {}
        self._binary_indicator_process_cols: set[str] = set()
        self._binary_state_process_cols: set[str] = set()

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

        # A reviewer may be reused across datasets.  Query text depends on the
        # target, roles and process semantics, so no prior HyDE entry is valid.
        self._hyde_cache.clear()

        # Only explicitly named ``cond_*`` binary columns are treated as
        # mutually exclusive one-hot condition flags (for example Ti
        # ``cond_ST-WQ``).  A generic numeric {0, 1} field may instead encode a
        # scientifically meaningful Boolean process state (for example Cu
        # ``Aging``), whose zero state must remain visible to retrieval.
        self._binary_indicator_process_cols = set()
        self._binary_state_process_cols = set()
        for col in self.process_cols:
            if col not in df.columns:
                continue
            raw = df[col].dropna()
            if raw.empty:
                continue
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().sum() != len(raw):
                continue
            unique = set(float(value) for value in numeric.unique())
            explicit_condition = bool(
                re.match(r"^cond(?:[_\- ])", str(col), flags=re.IGNORECASE)
            )
            if explicit_condition and unique.issubset({0.0, 1.0}):
                # A filtered dataset may contain only the active or only the
                # inactive state of an otherwise valid one-hot condition.
                self._binary_indicator_process_cols.add(col)
            elif unique == {0.0, 1.0}:
                self._binary_state_process_cols.add(col)

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

    # ── HyDE 查询 ───────────────────────────────────────────────────

    async def _build_hyde_query(self, candidate: dict) -> str:
        """Rewrite composition and raw processing conditions as a HyDE query."""
        process_items = _process_query_items(
            candidate,
            self.process_cols,
            getattr(self, "_binary_indicator_process_cols", set()),
            getattr(self, "_binary_state_process_cols", set()),
        )
        rendered_process_items = _render_process_items(process_items)
        cache_key = json.dumps(
            {
                "api_base_url": self.base_url,
                "model": self.model,
                "material_system": self.material_system,
                "target_col": self.target_col,
                "prompt_profile": self.prompt_profile,
                "composition_unit": getattr(self, "composition_unit", "wt.%"),
                "composition": {
                    el: _canonical_query_value(candidate.get(el, 0))
                    for el in self.element_cols
                },
                "processing": {
                    col: value
                    for col, value, _text, state in process_items
                    if state != "inactive"
                },
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if cache_key in self._hyde_cache:
            return self._hyde_cache[cache_key]

        known_elements: list[tuple[str, float]] = []
        for element in self.element_cols:
            canonical = _canonical_query_value(candidate.get(element))
            if isinstance(canonical, (int, float)) and abs(float(canonical)) > 0.001:
                known_elements.append((element, float(canonical)))
        comp_lines = "\n".join(
            f"  {element}: {value:.2f} "
            f"{getattr(self, 'composition_unit', 'wt.%')}"
            for element, value in known_elements
        )
        # 纯成分字符串 fallback: "Cu 4.40, Mg 1.50, Mn 0.60 ..."
        comp_flat = ", ".join(
            f"{element} {value:.2f}" for element, value in known_elements
        )
        process_lines = "\n".join(
            f"  {col}: {raw_text}; normalized search forms: {search_text}"
            for col, raw_text, search_text in rendered_process_items
        ) or "  (none provided)"
        process_flat = "; ".join(
            search_text for _col, _raw_text, search_text in rendered_process_items
        )

        text = ""
        try:
            prompt = _build_hyde_prompt(
                self.material_system,
                self.target_col,
                self.prompt_profile,
                getattr(self, "composition_unit", "wt.%"),
            ).format(
                composition=comp_lines,
                processing=process_lines,
            )
            resp, _key_alias = await self.key_pool.create_chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                timeout=60,
            )
            text = extract_message_text(resp.choices[0].message).strip()
        except Exception:
            if self.strict_scoring:
                raise
            text = ""

        # HyDE 失败或返回空 → 使用强力 keyword-dense fallback
        if not text:
            if self.strict_scoring:
                raise ValueError("empty HyDE response in strict scoring mode")
            text = (
                f"{self.material_system} alloy composition "
                f"({_composition_unit_search_terms(getattr(self, 'composition_unit', 'wt.%'))}): "
                f"{comp_flat}. "
                f"Relevant composition, processing, and target-property relationships."
            )

        # Preserve the composition basis even if the non-deterministic HyDE
        # rewrite omits it.  Dense and sparse retrieval therefore receive the
        # same explicit scientific unit contract.
        text = (
            text.rstrip()
            + " Deterministic composition basis: "
            + _composition_unit_search_terms(
                getattr(self, "composition_unit", "wt.%")
            )
            + "."
        )

        # LLM rewriting is non-deterministic and can silently drop exact
        # conditions.  Always append the raw process fields plus deterministic
        # aliases/unit equivalents after HyDE generation.  Inactive one-hot
        # conditions were removed by ``_render_process_items`` above.
        if process_flat:
            text = (
                text.rstrip()
                + " Deterministic processing conditions (raw and normalized "
                + f"equivalents): {process_flat}."
            )

        self._hyde_cache[cache_key] = text
        return text

    def _build_sparse_query(self, candidate: dict) -> str:
        """Build a sparse query from target, composition and raw process values."""
        parts: list[str] = []
        if self.material_system:
            parts.append(self.material_system)
        parts.append("alloy")
        parts.append(_target_label(self.target_col))
        if self.prompt_profile == "multi_objective_pareto":
            parts.append("multi objective Pareto trade-off frontier")
        parts.append(
            _composition_unit_search_terms(
                getattr(self, "composition_unit", "wt.%")
            )
        )

        nonzero: list[tuple[str, float]] = []
        for el in self.element_cols:
            try:
                value = float(candidate.get(el, 0))
            except (TypeError, ValueError):
                continue
            if abs(value) > 0.001:
                nonzero.append((el, value))

        if nonzero:
            # Element symbols are deliberately generic; alloy-family meaning comes from RAG.
            symbols = [el for el, _ in nonzero]
            parts.append(" ".join(symbols))
            parts.append(" ".join(f"{el} {value:.3g}" for el, value in nonzero))

            major = sorted(nonzero, key=lambda item: abs(item[1]), reverse=True)[:6]
            pair_terms = []
            for i, (a, _) in enumerate(major):
                for b, _ in major[i + 1:]:
                    pair_terms.append(f"{a}-{b}")
            if pair_terms:
                parts.append(" ".join(pair_terms))

        process_terms = []
        items = _process_query_items(
            candidate,
            self.process_cols,
            getattr(self, "_binary_indicator_process_cols", set()),
            getattr(self, "_binary_state_process_cols", set()),
        )
        for _col, _raw_text, search_text in _render_process_items(items):
            process_terms.append(search_text)
        if process_terms:
            parts.append("processing conditions " + " ; ".join(process_terms))

        if self.prompt_profile == "multi_objective_pareto":
            parts.append("composition processing property trade-off relationship")
        else:
            parts.append("composition processing property relationship")
        return " ".join(dict.fromkeys(p for p in parts if p))

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

        # ① HyDE 查询改写 → RAG 检索
        retrieval_mode = _validate_retrieval_mode(
            getattr(self, "retrieval_mode", "full")
        )
        if retrieval_mode == "disabled":
            # Empty internal query strings keep this branch type-stable; the
            # result serializes an empty query mapping for this control arm.
            # In particular, do not call HyDE, sparse-query construction,
            # Retriever.asearch(), or snippet preparation.
            query = ""
            bm25_query = ""
            hits: list[dict] = []
            snippets = NO_RETRIEVAL_PROMPT_PLACEHOLDER
        else:
            if self.use_hyde:
                query = await self._build_hyde_query(candidate_inputs)
            else:
                query = json.dumps(
                    {
                        "composition_unit": getattr(
                            self, "composition_unit", "wt.%"
                        ),
                        "candidate_inputs": candidate_inputs,
                    },
                    ensure_ascii=False,
                )

            bm25_query = self._build_sparse_query(candidate_inputs)

            hits = await self.retriever.asearch(
                query, top_k=self.top_k,
                section_filter=list(RAG_SECTION_FILTER),
                bm25_query=bm25_query, bm25_weight=0.4,
                max_per_source=RAG_MAX_PER_SOURCE,
            )
            if self.strict_scoring and not hits:
                raise RAGEvidenceError(
                    "RAG retrieval returned no hits in strict scoring mode"
                )

            process_items = _process_query_items(
                candidate_inputs,
                self.process_cols,
                getattr(self, "_binary_indicator_process_cols", set()),
                getattr(self, "_binary_state_process_cols", set()),
            )
            process_anchor_query = " ".join(
                search_text
                for _col, _raw_text, search_text
                in _render_process_items(process_items)
            )
            hits = _prepare_hit_snippets(
                hits,
                process_anchor_query or bm25_query,
                max_chars_per_hit=RAG_SNIPPET_MAX_CHARS,
            )
            snippets = _format_snippets(hits)

        # ② 构建 messages (新多体系格式)
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
            prompt_profile=self.prompt_profile,
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

        # ⑤ 附上 RAG 证据溯源
        result["_refs"] = _build_evidence_refs(
            hits,
            strict=self.strict_scoring,
        )
        result["_retrieval_queries"] = (
            {
                "dense": query,
                "sparse": bm25_query,
            }
            if retrieval_mode == "full"
            else {}
        )
        result["_retrieval_mode"] = retrieval_mode
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
