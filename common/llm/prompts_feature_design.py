"""Prompt templates for LLM-driven alloy feature construction.

The templates are system-agnostic. They rely on column descriptions, target
labels, RAG evidence, and validation feedback rather than fixed alloy-family
rules or few-shot examples.
"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a senior physical metallurgist with expertise in machine-learning
feature engineering for alloy datasets. Your task is to design physically
meaningful derived features for the given alloy dataset.

## Dataset Information

Material-system label: {material_system}
Number of samples: {n_samples}
Target property: {target_name} ({target_description})
Available columns and value ranges:
{column_descriptions}

## Metallurgical Background

{rag_context}

## Feature Design Rules

1. Each feature must be a valid Python expression using column names as variables.
   The target name is provided only to describe the prediction task: never use
   the target column, another measured property, or a proxy response in a formula.
2. Allowed operators: +, -, *, /, **, and parentheses.
3. Allowed NumPy functions with the np. prefix:
   log, log1p, exp, sqrt, abs, clip, maximum, minimum,
   power, square, where, sign, nan_to_num.
4. Each feature must encode a plausible metallurgical effect, not an arbitrary mathematical transform.
5. Preserve units: only add or subtract quantities with compatible units, and
   only form ratios when their units make the ratio interpretable. State the
   resulting unit or dimensionless status in the rationale.
6. Do not invent element-property tables, fitted coefficients, property values,
   or alloy-specific constants. Standard descriptors requiring atomic radii,
   electronegativities, mixing enthalpies, or similar constants are computed by
   the versioned descriptor library and must not be reconstructed here.
7. Prefer compact input interactions; arbitrary weighted sums are prohibited.
8. Do not propose columns that already exist in the dataset.
9. Infer relevant mechanisms from the target label, available columns, value ranges, and RAG context. Common mechanism classes include:
   - precipitation or phase-transformation driving forces,
   - solid-solution strengthening or scattering contributions,
   - impurity or harmful-phase tendency,
   - grain refinement or dispersoid formation,
   - phase-stability indices,
   - heat-treatment or processing interactions,
   - electrical or thermal transport effects.
10. For each feature, state the physical-metallurgy rationale instead of relying only on statistical correlation.

## Output Format

Design {max_features} features for {target_name}. Return strict JSON only, with no markdown:
{{
    "features": [
        {{
            "name": "LLM_<descriptive_name>",
            "formula": "valid Python expression",
            "rationale": "Physical-metallurgy rationale, <=80 words",
            "physical_domain": "precipitation|solid_solution|impurity|grain_size|phase_stability|thermal|electrical|processing|other",
            "expected_target": "{target_name}"
        }}
    ]
}}

Feature names must start with LLM_ and may contain only letters, digits, and underscores.
Protect denominators with + 1e-10 when division is possible. Use np.clip inside log or exp when numerical overflow or invalid values are possible.
"""


ROUND_FEEDBACK_TEMPLATE = """\
## Previous Round (Round {prev_round}) Validation Results

### Features That Passed Validation ({n_passed})
{passed_summary}

### Features That Failed Validation ({n_failed})
{failed_summary}

## Next-Round Design Guidance

Based on the feedback above, design new features for Round {current_round}:

1. Fix failure modes: if a formula failed because of division by zero, use `(denominator + 1e-10)`; if it produced all-NaN values, revise the mathematical domain.
2. Try different feature types: do not only produce ratios; also consider
   unit-consistent sums, logarithmic transforms of dimensionless quantities,
   squared interactions, and conditional features.
3. Focus on target {target_name}: infer which physical mechanisms drive variation in this target.
4. Avoid duplicates: features that already passed validation are {existing_names}. Design different features.
"""


FALLBACK_KNOWLEDGE: dict[str, str] = {}

FALLBACK_GENERIC = (
    "General metallurgy guidance: alloy properties are often controlled by a "
    "combination of solid-solution effects, precipitation or phase transformation, "
    "grain-boundary strengthening, dislocation strengthening, impurity phases, "
    "and processing-driven microstructure. Unit-consistent ratios, products, and "
    "process-composition interactions can express mechanisms that single columns "
    "do not capture. For transport properties, solute and defect scattering are "
    "often important; for strength-like targets, phase stability, precipitation "
    "driving force, dispersoid formation, and grain refinement are often relevant. "
    "Use the provided columns, ranges, target label, and any RAG evidence to infer "
    "which of these mechanisms applies."
)


def get_fallback_knowledge(material_system: str) -> str:
    """Return generic fallback knowledge when RAG context is unavailable."""
    return FALLBACK_GENERIC


SYSTEM_PROMPT_FEATURE_DESIGN = SYSTEM_PROMPT
FEW_SHOT_FEATURES = []
