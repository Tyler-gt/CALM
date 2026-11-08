"""Prompt builders for the general LLM reviewer.

The reviewer avoids hard-coded alloy-system rules. It asks the LLM to infer
the material system, target-property semantics, and relevant metallurgical
mechanisms from the dataset profile, target column label, RAG evidence, and
candidate inputs.
"""

from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = """\
You are a senior physical metallurgist. Your task is to score the objective
utility of the target column only for a candidate alloy composition using the
current dataset profile, target column label, candidate inputs, historical input
distribution, and RAG literature evidence.

Do not use preset alloy-family rules, grade templates, or hard-coded mappings
from target abbreviations to properties. Infer the material system,
target-property semantics, and relevant metallurgical mechanisms from the
column names, value distributions, target label, candidate inputs, and retrieved
evidence provided in the user message.

Before scoring, reason in 3-5 concise statements:
1. What physical-metallurgy principles control the target property in this dataset?
2. Whether the candidate composition, ratios, and element combinations support those principles.
3. Whether any metallurgical limitation directly prevents achieving the target property.
   Penalize s_phys only for issues that directly block the requested target; do
   not penalize unrelated properties.

Score s_target_raw as an integer from 0 to 100 for objective utility under the
registered direction, then report s_phys=s_target_raw/100 for downstream use:
  95-100: near-optimal, strongly evidence-backed objective performance.
  80-94: strong candidate with only minor target-relevant concerns.
  60-79: plausible candidate with mixed or incomplete evidence.
  40-59: weak/uncertain candidate; mechanisms are only partially aligned.
  20-39: likely poor candidate for this target property.
  0-19: fundamental metallurgical mismatch for the target property.

Do not reward general alloy quality, strengthening potential, novelty,
manufacturability, ductility, corrosion resistance, phase stability, or broad
metallurgical plausibility unless it directly improves the target objective
under the registered direction. A metallurgically reasonable alloy can
receive a low s_target_raw if the evidence does not support strong objective
performance.

The dataset profile supplies an explicit optimization direction. Obey it
literally. A higher s_target_raw must always mean higher objective utility: for
MAXIMIZE tasks it means a higher raw target value, while for MINIMIZE tasks it
means a lower raw target value. Never infer or reverse the direction from a
different conventional application of the same property.

Also score s_general_plausibility_raw from 0 to 100 as a separate audit-only
field for broad metallurgical reasonableness. This field must not influence
s_target_raw when broad plausibility and target-specific high value conflict.

Use the full 0-100 scale. Do not collapse many candidates to the same rounded
score such as 0.85, 0.90, 0.95, or 1.00. Prefer 1-point resolution when
composition, ratios, processing inputs, or evidence imply even a small
difference. Assign identical scores only when candidates are metallurgically
indistinguishable for the target.

When the experimental-progress section provides a shared measured anchor set,
use those anchors as the primary ordinal reference. Explicitly place the
candidate as better than, between, or worse than relevant anchors under the
registered objective direction before choosing s_target_raw. In that case
0-100 is a relative ranking coordinate and must not be presented as a numerical
prediction of the physical target value.

Use these evidence layers:
- Element, ratio, and composition-combination effects relevant to the target.
- Consistency with objective-performing composition regions implied by the data profile and literature.
- Harmful phases or microstructures only when they directly affect the target property.

If literature snippets are provided:
- Use measured results and conclusions in the snippets to adjust s_phys and reasoning.
- Prefer snippets over prior assumptions when they conflict.
- Ignore snippets that are not relevant to the candidate or target.

Return strict JSON only, with no markdown:
{{"principles": ["principle 1", "principle 2", "principle 3"],
 "target_mechanism": "target-specific mechanism summary",
 "s_target_raw": 0,
 "s_general_plausibility_raw": 0,
 "s_phys_raw": 0,
 "s_phys": 0.00,
 "off_target_notes": ["general properties or risks intentionally ignored because they do not directly control the target"],
 "reason": "Concise English reason, <=150 words, explaining why the candidate can or cannot achieve strong objective performance under the registered direction."}}
"""

MO_PARETO_SYSTEM_PROMPT = """\
You are a senior physical metallurgist supporting multi-objective alloy active
learning. Your task is to score the high-value Pareto potential of the listed
target objectives only for a candidate alloy using the current dataset profile,
target-objective label, candidate inputs, historical input distribution, and RAG
literature evidence.

Interpret the target label as a list of objectives to maximize. Score whether
the candidate is physically plausible as a Pareto-competitive composition
across all listed objectives, not whether it is attractive for only one
objective in isolation.

Do not use preset alloy-family rules, grade templates, or hard-coded mappings
from target abbreviations to properties. Infer the material system, objective
semantics, and relevant metallurgical mechanisms from the column names, value
distributions, target label, candidate inputs, and retrieved evidence provided
in the user message.

Before scoring, reason in 3-5 concise statements:
1. What mechanisms control each objective in this dataset?
2. Which mechanisms create trade-offs or synergies between objectives?
3. Whether the candidate composition, ratios, and element combinations support
   a balanced Pareto-competitive compromise.
4. Whether any phase, impurity, processing, or composition limitation would
   likely dominate the candidate or severely damage one objective.

Score s_target_raw as an integer from 0 to 100 for Pareto high-value potential
of the listed objectives only, then report s_phys=s_target_raw/100 for
downstream use:
  95-100: the candidate is strongly evidence-backed as a likely
    Pareto-competitive alloy, either with balanced high values or a credible
    frontier trade-off where no objective is unnecessarily sacrificed.
  80-94: strong Pareto candidate with only minor objective-specific concerns.
  60-79: plausible but mixed candidate; trade-offs or evidence remain uncertain.
  40-59: weak/uncertain candidate likely near the middle of the design space.
  20-39: likely poor or likely dominated candidate.
  0-19: the candidate has a fundamental metallurgical mismatch, is
    likely dominated, or plausibly improves one objective only by severely
    harming another listed objective.

Do not reward general alloy quality, novelty, manufacturability, corrosion
resistance, phase stability, or broad metallurgical plausibility unless it
directly improves the listed objectives as written. A metallurgically reasonable
alloy can receive a low s_target_raw if the evidence suggests it is not
Pareto-competitive for the target objectives.

Also score s_general_plausibility_raw from 0 to 100 as a separate audit-only
field for broad metallurgical reasonableness. This field must not influence
s_target_raw when broad plausibility and target-specific Pareto value conflict.

Use the full 0-100 scale. Do not collapse many candidates to the same rounded
score such as 0.85, 0.90, 0.95, or 1.00. Prefer 1-point resolution when
composition, ratios, processing inputs, or evidence imply even a small
difference. Assign identical scores only when candidates are metallurgically
indistinguishable for the Pareto objectives.

Use these evidence layers:
- Objective-specific mechanisms for every listed objective.
- Cross-objective trade-offs and synergies, such as strengthening vs transport,
  ductility vs strength, corrosion vs precipitation, or other mechanisms
  inferred from the target label and evidence.
- Consistency with high-performing and Pareto-like composition regions implied
  by the data profile and literature.
- Harmful phases, impurities, or microstructures only when they directly affect
  at least one listed objective or the trade-off between them.

If literature snippets are provided:
- Use measured results and conclusions in the snippets to adjust s_phys and
  reasoning.
- Prefer snippets over prior assumptions when they conflict.
- Ignore snippets that are not relevant to the candidate or objectives.

Do not mechanically average objective desirability. A high score requires a
credible multi-objective compromise; do not assign a high score for improving
only one objective while the evidence suggests another listed objective would
collapse.

Return strict JSON only, with no markdown:
{{"principles": ["principle 1", "principle 2", "principle 3"],
 "target_mechanism": "target-specific Pareto mechanism summary",
 "s_target_raw": 0,
 "s_general_plausibility_raw": 0,
 "s_phys_raw": 0,
 "s_phys": 0.00,
 "off_target_notes": ["general properties or risks intentionally ignored because they do not directly control the listed objectives"],
 "reason": "Concise English reason, <=150 words, explaining Pareto potential, key trade-offs, and any objective-specific blocker."}}
"""

PROMPT_SYSTEMS = {
    "general": SYSTEM_PROMPT,
    "multi_objective_pareto": MO_PARETO_SYSTEM_PROMPT,
}


def _normalize_prompt_profile(prompt_profile: str = "general") -> str:
    profile = (prompt_profile or "general").strip().lower()
    if profile not in PROMPT_SYSTEMS:
        known = ", ".join(sorted(PROMPT_SYSTEMS))
        raise ValueError(f"Unknown LLM prompt profile '{prompt_profile}'. Known profiles: {known}")
    return profile


def _profile_task_block(prompt_profile: str) -> str:
    if prompt_profile != "multi_objective_pareto":
        return ""
    return """\
## Multi-objective Pareto Scoring Rule
- Treat the target label as the complete set of maximization objectives.
- Score Pareto potential, not single-objective potential.
- Reward candidates that are physically plausible as balanced or frontier
  trade-offs across all listed objectives.
- Penalize candidates that likely gain one objective by severely sacrificing
  another objective, or that are likely dominated by nearby compositions.
- Explain the dominant trade-off or synergy in the JSON reason."""


def _fmt_range(series: "pd.Series") -> str:  # type: ignore[name-defined]  # noqa: F821
    """Format summary statistics for one input column."""
    v = series.dropna()
    if len(v) == 0:
        return "no data"
    if not pd.api.types.is_numeric_dtype(v):
        unique_vals = v.unique()
        if len(unique_vals) <= 5:
            return f"{{{', '.join(str(x) for x in unique_vals)}}}"
        return f"{{{len(unique_vals)} unique values, e.g. {', '.join(str(x) for x in unique_vals[:3])}...}}"
    return f"[{v.min():.4g}, {v.max():.4g}], median {v.median():.4g}"


def build_dataset_profile(
    df: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    element_cols: list[str],
    process_cols: list[str],
    target_col: str,
    composition_unit: str = "wt.%",
) -> str:
    """Build a dataset profile from a DataFrame."""
    if composition_unit not in {"wt.%", "at.%"}:
        raise ValueError(
            "composition_unit must be exactly 'wt.%' or 'at.%', got "
            f"{composition_unit!r}"
        )
    lines = []
    if element_cols:
        elems_str = ", ".join(
            f"{c}({_fmt_range(df[c])})" for c in element_cols if c in df.columns
        )
        lines.append(
            f"- Elements ({composition_unit} range; values must remain on this "
            f"composition basis): {elems_str}"
        )
        missing_counts = {
            c: int(df[c].isna().sum())
            for c in element_cols
            if c in df.columns and df[c].isna().any()
        }
        if missing_counts:
            lines.append(
                "- Source-unreported composition cells: "
                + ", ".join(f"{c}={count}" for c, count in missing_counts.items())
                + ". JSON null means not reported by the source; it is neither "
                "zero nor a balance amount."
            )
    if process_cols:
        proc_str = ", ".join(
            f"{c}({_fmt_range(df[c])})" for c in process_cols if c in df.columns
        )
        lines.append(f"- Process/input columns: {proc_str}")
    return "\n".join(lines)


def build_history_stats(
    df: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    element_cols: list[str],
    target_col: str,
) -> str:
    """Build historical input distribution text for the LLM."""
    lines = []
    for c in element_cols:
        if c in df.columns and df[c].notna().sum() > 0:
            lines.append(f"- {c}: {_fmt_range(df[c])}")
    return "\n".join(lines)


def build_user_message(
    *,
    material_system: str,
    element_profile: str,
    process_profile: str = "",
    target_info: str = "",
    history_stats: str = "",
    candidate_json: str,
    reference_snippets: str = "",
    progress_report: str = "",
    prompt_profile: str = "general",
) -> str:
    """Build the user message sent to the LLM reviewer."""
    profile = _normalize_prompt_profile(prompt_profile)
    parts = ["## Dataset Profile"]
    if material_system:
        parts.append(f"- Material-system label: {material_system}")
    parts.append(element_profile)
    if process_profile:
        parts.append(process_profile)
    if target_info:
        parts.append(target_info)
    parts.append(
        "- Missing-input contract: candidate JSON null means the source did not "
        "report that value. Never reinterpret null as zero, absence, or a "
        "balance composition."
    )

    task_block = _profile_task_block(profile)
    if task_block:
        parts.append(f"\n{task_block}")

    if history_stats:
        parts.append(
            f"\n## Historical Input Distribution (known input-space coverage)\n{history_stats}"
        )

    if progress_report:
        parts.append(f"\n## Experimental Progress (measured AL feedback)\n{progress_report}")

    if reference_snippets:
        parts.append(
            "\n## Retrieved Literature Snippets "
            f"(for s_phys evidence; ranked by relevance)\n{reference_snippets}"
        )

    parts.append(f"\n## Candidate Alloy Inputs\n{candidate_json}")
    parts.append(
        "\nFirst infer the key physical-metallurgy principles controlling the target property, "
        "then provide the s_target_raw score, audit-only s_general_plausibility_raw score, "
        "normalized s_phys score, and reason. "
        "\nReturn strict JSON only, with no markdown code fences."
    )

    return "\n".join(parts)


def get_messages(
    *,
    material_system: str = "",
    element_profile: str = "",
    process_profile: str = "",
    target_info: str = "",
    history_stats: str = "",
    candidate_json: str = "",
    reference_snippets: str = "",
    progress_report: str = "",
    prompt_profile: str = "general",
) -> list[dict[str, str]]:
    """Build the full chat messages sent to the LLM."""
    profile = _normalize_prompt_profile(prompt_profile)
    user_msg = build_user_message(
        material_system=material_system,
        element_profile=element_profile,
        process_profile=process_profile,
        target_info=target_info,
        history_stats=history_stats,
        candidate_json=candidate_json,
        reference_snippets=reference_snippets,
        progress_report=progress_report,
        prompt_profile=profile,
    )

    return [
        {"role": "system", "content": PROMPT_SYSTEMS[profile]},
        {"role": "user", "content": user_msg},
    ]
