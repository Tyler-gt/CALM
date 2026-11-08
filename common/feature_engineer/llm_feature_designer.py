"""LLM 驱动的特征构建器。

4 阶段流水线:
  Phase 1 — 数据集感知 (列统计、材料体系推断)
  Phase 2 — LLM 特征设计 (2-3 轮迭代, RAG 注入冶金学上下文)
  Phase 3 — 公式验证与计算 (AST 安全白名单 + 统计校验)
  Phase 4 — 可选效用过滤 (默认关闭；统一选择器在流水线外执行)

用法:
    designer = LLMFeatureDesigner()
    result = await designer.design_features(
        df, target_cols=["Strength"], element_cols=["Cu","Mg","Mn",...]
    )
    # result["llm_features"]  → pd.DataFrame of validated candidate features
    # result["metadata"]       → full trace dict
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

# ── 项目路径与 env ───────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_PROJECT = _THIS_FILE.parent.parent.parent
sys.path.insert(0, str(_PROJECT))

# ── 延迟导入 (避免循环依赖) ──────────────────────────────────────────────────
from common.llm.prompts_feature_design import (
    SYSTEM_PROMPT_FEATURE_DESIGN,
    FEW_SHOT_FEATURES,
    ROUND_FEEDBACK_TEMPLATE,
    FALLBACK_KNOWLEDGE,
    FALLBACK_GENERIC,
)
from common.llm.response_utils import extract_message_text
from common.llm.config import get_llm_model
from common.llm.key_pool import AsyncUSTCKeyPool, require_ustc_api_keys

FEATURE_FORMULA_VALIDATOR_VERSION = "input_only_ast_finite_missing_v3"
FEATURE_DESIGN_CACHE_VERSION = "full_prompt_response_identity_v2"

# ── RAG 全局单例 ────────────────────────────────────────────────────────────
_RETRIEVER = None

def _get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        try:
            from common.rag.retrieve import Retriever
            _RETRIEVER = Retriever()
        except (FileNotFoundError, ImportError) as e:
            print(f"[LLMFeatureDesigner] RAG index not available: {e}")
            _RETRIEVER = False
    return _RETRIEVER if _RETRIEVER is not False else None


# ── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class DatasetProfile:
    """Phase 1 输出: 数据集结构化描述."""
    material_system: str
    composition_unit: str
    available_columns: list[str]
    element_cols: list[str]
    process_cols: list[str]
    column_ranges: dict          # {col: {"min","max","mean","std","q25","q50","q75"}}
    target_cols: list[str]
    target_ranges: dict
    n_samples: int
    has_process_params: bool


@dataclass
class FeatureFormula:
    """LLM 提出的单个特征."""
    name: str
    formula: str
    rationale: str
    physical_domain: str        # precipitation | solid_solution | impurity | grain_size | phase_stability | thermal
    expected_target: str


@dataclass
class ValidationResult:
    """Phase 3 输出: 单个特征的验证结果."""
    feature: FeatureFormula
    passed: bool
    reason: str
    series: Optional[pd.Series] = None


# ── AST 安全校验 ─────────────────────────────────────────────────────────────

# 允许的 NumPy 函数白名单
_ALLOWED_NP_FUNCTIONS: set[str] = {
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "log", "log1p", "log2", "log10", "exp", "expm1", "sqrt",
    "abs", "clip", "maximum", "minimum", "power", "square",
    "where", "sign", "nan_to_num",
}

# 禁止的 Python 关键字/内置
_FORBIDDEN_NAMES: set[str] = {
    "__import__", "exec", "eval", "open", "compile",
    "getattr", "setattr", "delattr", "globals", "locals",
    "__builtins__", "__class__", "__subclasses__",
    "breakpoint", "input", "help", "license", "credits",
}


def validate_formula_safety(formula: str, available_columns: set[str]) -> tuple[bool, str]:
    """用 AST 白名单校验公式安全性。

    Returns (is_safe, error_message).
    """
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        node_type = type(node)

        # ── 字面量 ──
        if node_type in (ast.Constant, ast.Num):
            continue

        # ── 表达式容器 ──
        if node_type in (ast.Expression, ast.Load):
            continue

        # ── 二元运算: +, -, *, /, **, % ──
        if node_type in (ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod):
            continue

        # ── 一元运算: -x, +x ──
        if node_type in (ast.UnaryOp, ast.USub, ast.UAdd):
            continue

        # ── 比较运算: >, <, ==, !=, >=, <= ──
        if node_type in (ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq):
            continue

        # ── 布尔运算: and, or, not (np.where 内可能需要) ──
        if node_type in (ast.BoolOp, ast.And, ast.Or, ast.Not):
            continue

        # ── 三元: x if cond else y ──
        if node_type is ast.IfExp:
            continue

        # ── 切片/下标: arr[i], arr[i:j] ──
        if node_type in (ast.Subscript, ast.Slice):
            continue

        # ── 列表/元组: [x, y], (x, y) ──
        if node_type in (ast.List, ast.Tuple):
            continue

        # ── 变量名 (列名) ──
        if node_type is ast.Name:
            name = node.id
            if name in _FORBIDDEN_NAMES:
                return False, f"Forbidden name: '{name}'"
            if name not in available_columns and name != "np":
                return False, f"Unknown column: '{name}'"
            continue

        # ── 属性访问 (np.xxx) ──
        if node_type is ast.Attribute:
            # 只允许 np.<function>
            if not isinstance(node.value, ast.Name):
                return False, "Only 'np.func' attribute access allowed"
            if node.value.id != "np":
                return False, f"Only 'np.*' allowed, got: {node.value.id}.{node.attr}"
            if node.attr not in _ALLOWED_NP_FUNCTIONS:
                return False, f"Disallowed numpy function: np.{node.attr}"
            continue

        # ── 函数调用 ──
        if node_type is ast.Call:
            func = node.func
            if isinstance(func, ast.Attribute):
                # np.func(args)
                if not isinstance(func.value, ast.Name) or func.value.id != "np":
                    return False, "Only 'np.func()' calls allowed"
                if func.attr not in _ALLOWED_NP_FUNCTIONS:
                    return False, f"Disallowed numpy function: np.{func.attr}"
            else:
                return False, "Only 'np.func()' calls allowed"
            continue

        # ── 其他一切: 拒绝 ──
        return False, f"Unsupported syntax: {type(node).__name__}"

    # ── 额外安全检查: 字符串匹配禁止项 ──
    formula_lower = formula.lower()
    for forbidden in ["__", "exec", "eval", "open(", "compile(", "getattr", "setattr"]:
        if forbidden in formula_lower:
            return False, f"Forbidden pattern detected: '{forbidden}'"

    return True, "OK"


def _frame_fingerprint(df: pd.DataFrame, columns: list[str]) -> str:
    """Hash the exact ordered values visible to feature design."""
    view = df.loc[:, columns].copy()
    for col in view.columns:
        if not pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].astype("string").fillna("<NA>")
    row_hashes = pd.util.hash_pandas_object(
        view,
        index=True,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(columns, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _prompt_template_fingerprint() -> str:
    payload = "\n".join(
        [
            SYSTEM_PROMPT_FEATURE_DESIGN,
            ROUND_FEEDBACK_TEMPLATE,
            json.dumps(FEW_SHOT_FEATURES, ensure_ascii=False, sort_keys=True),
            json.dumps(FALLBACK_KNOWLEDGE, ensure_ascii=False, sort_keys=True),
            FALLBACK_GENERIC,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── 主类 ─────────────────────────────────────────────────────────────────────

class LLMFeatureDesigner:
    """LLM 驱动的合金特征自动构建器。"""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        top_k_rag: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.model = get_llm_model() if model is None else model
        self.temperature = temperature
        self.top_k_rag = top_k_rag
        self.random_state = random_state
        self.verbose = verbose

        self.key_pool = AsyncUSTCKeyPool(
            require_ustc_api_keys(1),
            per_key_concurrency=4,
        )
        self._sem = None  # Created lazily in async context (avoid no-event-loop error)

    # ── 公共入口 ──────────────────────────────────────────────────────────

    async def design_features(
        self,
        df: pd.DataFrame,
        target_cols: list[str],
        element_cols: list[str],
        process_cols: Optional[list[str]] = None,
        n_rounds: int = 3,
        max_features_per_round: int = 12,
        use_rag: bool = False,
        cache_path: Optional[str] = None,
        material_system: Optional[str] = None,
        composition_unit: str = "wt.%",
        utility_filter: bool = False,
    ) -> dict:
        """运行完整的 4 阶段特征设计流水线。

        Parameters
        ----------
        df: 包含成分/工艺/目标列的 DataFrame。
        target_cols: 目标列名 (如 ["Strength"] 或 ["Strength", "TC"])。
        element_cols: 元素列名 (如 ["Cu", "Mg", "Mn", ...])。
        process_cols: 工艺列名 (可选, 如 ["SATemp", "AgeingTemp", "AgeingTime"])。
        n_rounds: LLM 生成轮数 (推荐 2-3)。
        max_features_per_round: 每轮最大候选特征数。
        use_rag: 是否使用 RAG 检索冶金学上下文。
        cache_path: 可选缓存路径 (.joblib), 避免重复 LLM 调用。
        material_system: 已注册的数据集材料体系；提供时优先于自动推断。
        composition_unit: 元素成分列的显式基准，只允许 ``wt.%`` 或 ``at.%``。
        utility_filter: 是否在本模块内使用目标监督过滤。正式主协议保持
            ``False``，避免与外层选择器重复使用同一批标签。

        Returns
        -------
        dict:
            "llm_features": pd.DataFrame — 通过验证（以及可选效用过滤）的特征
            "all_candidates": pd.DataFrame — 所有生成的特征 (过滤前)
            "metadata": dict — 完整追溯 (profile, rounds, validation, filtering)
        """
        process_cols = process_cols or []
        if composition_unit not in {"wt.%", "at.%"}:
            raise ValueError(
                "composition_unit must be exactly 'wt.%' or 'at.%', got "
                f"{composition_unit!r}"
            )
        # Filter to columns actually present
        element_cols = [c for c in element_cols if c in df.columns]
        process_cols = [c for c in process_cols if c in df.columns]
        target_cols = [c for c in target_cols if c in df.columns]

        if not target_cols:
            raise ValueError(f"No target columns found in DataFrame. Looked for: {target_cols}")
        if not element_cols:
            raise ValueError(f"No element columns found in DataFrame. Looked for: {element_cols}")

        # ── Phase 1: Dataset Introspection ──
        self._log("Phase 1: Dataset introspection...")
        profile = self._introspect_dataset(
            df,
            target_cols,
            element_cols,
            process_cols,
            material_system=material_system,
            composition_unit=composition_unit,
        )
        self._log(f"  Material system: {profile.material_system}")
        self._log(f"  Elements: {profile.element_cols}, Targets: {profile.target_cols}")

        # RAG availability is part of the scientific cache identity.
        should_use_rag = use_rag and profile.material_system.startswith("Al-Cu-Mg")
        if use_rag and not should_use_rag:
            self._log(
                f"  RAG disabled: only has papers for Al alloys, "
                f"not {profile.material_system}"
            )
        visible_columns = [
            *profile.element_cols,
            *profile.process_cols,
            *profile.target_cols,
        ]
        cache_identity = {
            "cache_version": FEATURE_DESIGN_CACHE_VERSION,
            "formula_validator_version": FEATURE_FORMULA_VALIDATOR_VERSION,
            "model": self.model,
            "api_base_url": self.key_pool.base_url,
            "temperature": float(self.temperature),
            "n_rounds": int(n_rounds),
            "max_features_per_round": int(max_features_per_round),
            "use_rag_requested": bool(use_rag),
            "use_rag_effective": bool(should_use_rag),
            "material_system": profile.material_system,
            "composition_unit": profile.composition_unit,
            "input_columns": [*profile.element_cols, *profile.process_cols],
            "target_cols": profile.target_cols,
            "visible_data_sha256": _frame_fingerprint(df, visible_columns),
            "prompt_templates_sha256": _prompt_template_fingerprint(),
            "utility_filter": bool(utility_filter),
        }

        if cache_path:
            cache_file = Path(cache_path)
            if cache_file.exists():
                import pickle
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                cached_identity = cached.get("metadata", {}).get(
                    "cache_identity"
                )
                if cached_identity == cache_identity:
                    self._log(f"Loading cached LLM features from {cache_path}")
                    return cached
                self._log(
                    "Ignoring stale LLM feature cache: full scientific identity "
                    "does not match the current request"
                )

        # ── Per-target processing ──
        all_results = {}
        metadata = {
            "cache_identity": cache_identity,
            "profile": {
                "material_system": profile.material_system,
                "composition_unit": profile.composition_unit,
                "material_system_source": (
                    "registered" if str(material_system or "").strip()
                    else "composition_inference"
                ),
                "n_samples": profile.n_samples,
                "element_cols": profile.element_cols,
                "process_cols": profile.process_cols,
                "target_cols": profile.target_cols,
                "column_ranges": profile.column_ranges,
                "target_ranges": profile.target_ranges,
            },
            "per_target": {},
        }

        for target in target_cols:
            self._log(f"\n── Target: {target} ──")

            # Build base features for this target (elements + process)
            base_cols = profile.element_cols + profile.process_cols
            df_base = df[base_cols].copy()

            # ── Phase 2: LLM Feature Design ──
            self._log("Phase 2: LLM feature design...")
            all_formulas, round_records = await self._run_design_rounds(
                profile=profile,
                target=target,
                n_rounds=n_rounds,
                max_features_per_round=max_features_per_round,
                use_rag=should_use_rag,
                df=df,
                existing_df=df_base,
            )
            n_valid = len(all_formulas)
            self._log(f"  Total valid features across {n_rounds} rounds: {n_valid}")

            if n_valid == 0:
                self._log("  [WARN] No valid LLM features generated, returning empty result")
                all_results[target] = {
                    "llm_features": pd.DataFrame(index=df.index),
                    "all_candidates": pd.DataFrame(),
                    "selected_cols": [],
                }
                metadata["per_target"][target] = {
                    "rounds": round_records,
                    "n_total_valid": 0,
                    "selected_cols": [],
                }
                continue

            # ── Phase 3: Compute all valid features ──
            self._log("Phase 3: Computing valid features...")
            df_all_candidates = self._compute_feature_df(df, all_formulas, base_cols=base_cols)

            # The publication path performs one shared selection step outside
            # this module.  Keeping all validated candidates here prevents
            # target reuse and makes a zero-feature result explicit.
            if utility_filter:
                self._log("Phase 4: Optional utility filtering...")
                y_series = pd.to_numeric(df[target], errors="coerce")
                selected_cols = self._filter_by_utility(
                    df_base=df_base,
                    df_llm=df_all_candidates,
                    y=y_series,
                    target_name=target,
                )
            else:
                self._log(
                    "Phase 4: Utility filtering disabled; returning validated "
                    "candidates for the shared selector"
                )
                selected_cols = list(df_all_candidates.columns)
            self._log(f"  Selected {len(selected_cols)}/{len(df_all_candidates.columns)} features")

            df_selected = df_all_candidates[selected_cols] if selected_cols else pd.DataFrame(index=df.index)

            all_results[target] = {
                "llm_features": df_selected,
                "all_candidates": df_all_candidates,
                "selected_cols": selected_cols,
            }
            metadata["per_target"][target] = {
                "rounds": round_records,
                "all_formulas": [
                    {"name": f.name, "formula": f.formula, "rationale": f.rationale,
                     "domain": f.physical_domain}
                    for f in all_formulas
                ],
                "n_total_valid": n_valid,
                "selected_cols": selected_cols,
                "utility_filter_applied": bool(utility_filter),
            }

        # ── Assemble result ──
        result = {
            "llm_features": {
                t: all_results[t]["llm_features"] for t in target_cols
            },
            "all_candidates": {
                t: all_results[t]["all_candidates"] for t in target_cols
            },
            "metadata": metadata,
        }

        if cache_path:
            import pickle
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._log(f"Cached result to {cache_path}")

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 1: Dataset Introspection
    # ═══════════════════════════════════════════════════════════════════════════

    def _introspect_dataset(
        self,
        df: pd.DataFrame,
        target_cols: list[str],
        element_cols: list[str],
        process_cols: list[str],
        material_system: Optional[str] = None,
        composition_unit: str = "wt.%",
    ) -> DatasetProfile:
        """分析数据集并生成结构化 profile (纯数据驱动, 无 LLM 调用)."""
        numeric = df.select_dtypes(include=[np.number])
        column_ranges = {}
        for col in numeric.columns:
            col_data = numeric[col].dropna()
            if len(col_data) > 0:
                column_ranges[col] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "q25": float(col_data.quantile(0.25)),
                    "q50": float(col_data.quantile(0.50)),
                    "q75": float(col_data.quantile(0.75)),
                }

        target_ranges = {t: column_ranges[t] for t in target_cols if t in column_ranges}
        resolved_material_system = str(material_system or "").strip()
        if not resolved_material_system:
            resolved_material_system = self._detect_material_system(
                element_cols,
                df=df,
            )

        return DatasetProfile(
            material_system=resolved_material_system,
            composition_unit=composition_unit,
            # Target names and values inform the task description, but they are
            # never legal variables in a proposed feature formula.
            available_columns=list(dict.fromkeys([*element_cols, *process_cols])),
            element_cols=element_cols,
            process_cols=process_cols,
            column_ranges=column_ranges,
            target_cols=target_cols,
            target_ranges=target_ranges,
            n_samples=len(df),
            has_process_params=len(process_cols) > 0,
        )

    @staticmethod
    def _detect_material_system(
        element_cols: list[str],
        df: Optional[pd.DataFrame] = None,
    ) -> str:
        """从元素列推断合金体系.

        优先根据组成数据中的平均元素占比识别基体；只有没有明显基体或
        没有数据时才使用列名启发式。这样 Mg 基数据即使同时包含痕量
        Al/Cu 列，也不会被误判为 2xxx 铝合金。
        """
        elems = set(e.lower() for e in element_cols)

        # ── 数据驱动的基体识别 ──
        # 比例尺度 (sum≈1) 与百分数尺度 (sum≈100) 都适用，因为比较的是
        # 各元素非负均值占全部元素均值之和的份额。
        if df is not None:
            available = [col for col in element_cols if col in df.columns]
            if available:
                numeric = df[available].apply(pd.to_numeric, errors="coerce")
                means = numeric.clip(lower=0).mean(axis=0, skipna=True).fillna(0.0)
                total = float(means.sum())
                row_sum_median = float(
                    numeric.clip(lower=0).sum(axis=1, min_count=1).median()
                )
                # A largest-solute share is not a base-metal signal when the
                # listed columns sum to only a few percent and the matrix element
                # is implicit (balance). Accept only near-complete fraction- or
                # percent-scale composition vectors for base detection.
                complete_composition = (
                    0.50 <= row_sum_median <= 1.50
                    or 50.0 <= row_sum_median <= 110.0
                )
                if total > 0 and complete_composition:
                    base_col = str(means.idxmax())
                    base = base_col.lower()
                    base_share = float(means.loc[base_col] / total)
                    if base_share >= 0.50:
                        if base == "mg":
                            return "Mg alloy (Mg-based)"
                        if base == "ti":
                            return "Ti alloy (Ti-based)"
                        if base == "fe":
                            return "Steel (Fe-based)"
                        if base == "cu":
                            return "Cu alloy (Cu-based)"
                        if base == "ni":
                            return "Ni-based superalloy"
                        if base == "al":
                            if "zn" in elems and "mg" in elems:
                                return "Al-Zn-Mg (7xxx series)"
                            if "cu" in elems and "mg" in elems:
                                return "Al-Cu-Mg (2xxx series)"
                            if "si" in elems and "mg" in elems:
                                return "Al-Si-Mg (6xxx series)"
                            if "mg" in elems:
                                return "Al-Mg (5xxx series)"
                            return "Al alloy (Al-based)"

        # ── 无明显基体时的列名回退 ──
        ordered = [str(element).lower() for element in element_cols]
        first = ordered[0] if ordered else ""
        if first == "ti":
            return "Ti alloy (generic)"
        if first == "mg":
            return "Mg alloy (generic)"
        if first == "fe":
            return "Steel (Fe-based)"
        if first == "ni" and "cr" in elems:
            return "Ni-based superalloy"

        if "al" in elems:
            if "zn" in elems and "mg" in elems:
                return "Al-Zn-Mg (7xxx series)"
            if "cu" in elems and "mg" in elems:
                return "Al-Cu-Mg (2xxx series)"
            if "si" in elems and "mg" in elems:
                return "Al-Si-Mg (6xxx series)"
            if "mg" in elems:
                return "Al-Mg (5xxx series)"
            return "Al alloy (generic)"

        # ── 隐式 Al 合金检测: Cu+Mg 共存, 无钢/铜/镁合金强特征 ──
        if "cu" in elems and "mg" in elems:
            # 钢的特征: C 或 高浓度 Fe+Cr+Ni 组合
            steel_indicators = {"c", "mo", "w", "v", "co", "nb"}
            steel_strong = bool(elems & steel_indicators)
            # Ni 基特征: Ni + Cr + Co
            ni_based = "ni" in elems and "cr" in elems and ("co" in elems or "mo" in elems or "w" in elems)
            if not steel_strong and not ni_based:
                return "Al-Cu-Mg (2xxx series, Al as balance)"

        # ── 钢 ──
        if "fe" in elems and ("cr" in elems or "ni" in elems or "c" in elems):
            return "Steel (Fe-based)"

        # ── 铜合金 ──
        if "cu" in elems and "zn" in elems:
            return "Cu-Zn (Brass)"
        if "cu" in elems:
            return "Cu alloy (generic)"

        # ── 镁合金 ──
        if "mg" in elems:
            return "Mg alloy (generic)"

        # ── 钛合金 ──
        if "ti" in elems:
            return "Ti alloy (generic)"

        # ── 镍基 ──
        if "ni" in elems and "cr" in elems:
            return "Ni-based superalloy"

        return f"Unknown ({'|'.join(element_cols[:4])}...)"

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 2: LLM Feature Design
    # ═══════════════════════════════════════════════════════════════════════════

    async def _run_design_rounds(
        self,
        profile: DatasetProfile,
        target: str,
        n_rounds: int,
        max_features_per_round: int,
        use_rag: bool,
        df: pd.DataFrame,
        existing_df: pd.DataFrame,
    ) -> tuple[list[FeatureFormula], list[dict]]:
        """运行 LLM 特征生成的多轮迭代."""
        all_formulas: list[FeatureFormula] = []
        all_names: set[str] = set()
        round_records: list[dict] = []

        rag_context = ""
        if use_rag:
            rag_context = await self._retrieve_rag_context(profile, target)

        for round_idx in range(1, n_rounds + 1):
            self._log(f"  Round {round_idx}/{n_rounds}...")

            # Build feedback from previous round
            feedback = ""
            if round_records:
                feedback = self._build_round_feedback(round_records[-1], target, round_idx, all_names)

            # Call LLM
            formulas, llm_call_record = await self._call_llm_for_features(
                profile=profile,
                target=target,
                round_num=round_idx,
                feedback=feedback,
                rag_context=rag_context,
                max_features=max_features_per_round,
                previous_names=sorted(all_names),
            )

            # Phase 3 validation within the loop
            available_cols = set(profile.available_columns)
            results = self._compute_and_validate_batch(
                df,
                formulas,
                existing_df,
                available_cols,
            )

            n_passed = sum(1 for r in results if r.passed)
            n_failed = len(results) - n_passed

            round_record = {
                "round": round_idx,
                "n_proposed": len(formulas),
                "n_passed": n_passed,
                "n_failed": n_failed,
                "llm_call": llm_call_record,
                "features": [
                    {
                        "name": r.feature.name,
                        "formula": r.feature.formula,
                        "rationale": r.feature.rationale,
                        "passed": r.passed,
                        "error": "" if r.passed else r.reason,
                    }
                    for r in results
                ],
            }
            round_records.append(round_record)

            for r in results:
                if r.passed and r.feature.name not in all_names:
                    all_formulas.append(r.feature)
                    all_names.add(r.feature.name)

            self._log(f"    {n_passed} passed, {n_failed} failed")

            # Early stop if last round produced nothing useful
            if n_passed == 0 and round_idx >= 2:
                self._log("    No valid features in this round, stopping early")
                break

        return all_formulas, round_records

    async def _call_llm_for_features(
        self,
        profile: DatasetProfile,
        target: str,
        round_num: int,
        feedback: str,
        rag_context: str,
        max_features: int,
        previous_names: list[str],
    ) -> tuple[list[FeatureFormula], dict]:
        """调用 LLM 生成特征公式."""
        column_descriptions = self._format_column_descriptions(profile, target)
        target_description = self._format_target_description(profile, target)

        # Build few-shot section
        few_shot_text = self._format_few_shot(profile, target)

        system_prompt = SYSTEM_PROMPT_FEATURE_DESIGN.format(
            material_system=profile.material_system,
            target_name=target,
            n_samples=profile.n_samples,
            column_descriptions=column_descriptions,
            target_description=target_description,
            rag_context=rag_context,
            few_shot_section=few_shot_text,
            max_features=max_features,
        )

        # User message with round-specific instructions
        user_content = f"## Design Round {round_num}\n\n"
        if feedback:
            user_content += f"{feedback}\n\n"
        if previous_names:
            user_content += (
                f"### Already Designed Features (avoid these names)\n"
                f"{json.dumps(previous_names, indent=2, ensure_ascii=False)}\n\n"
            )
        user_content += f"Please propose {max_features} new features for predicting {target}."

        messages = [
            {"role": "system", "content": system_prompt},
            *FEW_SHOT_FEATURES,
            {"role": "user", "content": user_content},
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._sem is None:
                    self._sem = asyncio.Semaphore(10)
                async with self._sem:
                    await self.key_pool.open_session()
                    try:
                        resp, key_alias = await self.key_pool.create_chat_completion(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            response_format={"type": "json_object"},
                            timeout=120,
                        )
                    finally:
                        await self.key_pool.close_session()
                raw = extract_message_text(resp.choices[0].message)
                if not raw:
                    raise ValueError("empty LLM response")
                self._log(f"    Feature-design response served by {key_alias}")
                return self._parse_feature_response(
                    raw,
                    set(profile.available_columns),
                ), {
                    "model": self.model,
                    "temperature": float(self.temperature),
                    "key_alias": key_alias,
                    "messages": messages,
                    "raw_response": raw,
                    "raw_response_sha256": hashlib.sha256(
                        raw.encode("utf-8")
                    ).hexdigest(),
                }

            except Exception as e:
                self._log(f"    LLM call attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))
                else:
                    self._log(f"    All {max_retries} attempts failed, returning empty list")
                    return [], {
                        "model": self.model,
                        "temperature": float(self.temperature),
                        "messages": messages,
                        "error": str(e),
                    }

        return [], {
            "model": self.model,
            "temperature": float(self.temperature),
            "messages": messages,
            "error": "unreachable retry exit",
        }

    def _parse_feature_response(self, raw: str, available_columns: set[str]) -> list[FeatureFormula]:
        """从 LLM JSON 响应中解析特征列表."""
        fallback: list[FeatureFormula] = []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                decoder = json.JSONDecoder()
                data, _ = decoder.raw_decode(raw)
            except json.JSONDecodeError:
                self._log(f"    Failed to parse LLM JSON response: {raw[:200]}...")
                return fallback

        features_raw = data.get("features", [])
        if not isinstance(features_raw, list):
            return fallback

        formulas = []
        for item in features_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            formula = str(item.get("formula", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            domain = str(item.get("physical_domain", "unknown")).strip()
            exp_target = str(item.get("expected_target", "unknown")).strip()

            if not name or not formula:
                continue

            # Sanitize name
            name = "LLM_" + name if not name.startswith("LLM_") else name
            name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
            name = name[:80]  # Truncate long names

            formulas.append(FeatureFormula(
                name=name,
                formula=formula,
                rationale=rationale,
                physical_domain=domain,
                expected_target=exp_target,
            ))

        return formulas

    # ── RAG ────────────────────────────────────────────────────────────────

    async def _retrieve_rag_context(self, profile: DatasetProfile, target: str) -> str:
        """检索冶金学上下文并以格式化字符串返回."""
        retriever = _get_retriever()
        if retriever is None:
            return self._get_fallback_knowledge(profile.material_system)

        try:
            # Query 1: Feature design context
            query1 = (
                f"{profile.material_system} {target} "
                f"{profile.composition_unit} composition "
                f"{' '.join(profile.element_cols[:6])} "
                f"precipitation strengthening solid solution composition design "
                f"phase formation grain refinement"
            )
            hits1 = await retriever.asearch(query1, top_k=self.top_k_rag)

            # Query 2: Target-specific mechanism
            if target.lower() in ("strength", "uts", "yield"):
                query2 = (
                    f"{profile.material_system} {profile.composition_unit} composition "
                    "precipitation hardening strengthening "
                    f"mechanisms phase composition effect aging"
                )
            else:
                query2 = (
                    f"{profile.material_system} {target} "
                    f"{profile.composition_unit} composition "
                    f"thermal conductivity electron scattering solute impurity effect"
                )
            hits2 = await retriever.asearch(query2, top_k=max(3, self.top_k_rag // 2))

            # Deduplicate and limit
            all_hits = self._deduplicate_hits(hits1 + hits2)
            all_hits = all_hits[:self.top_k_rag]

            return self._format_snippets(all_hits)

        except Exception as e:
            self._log(f"  RAG retrieval failed: {e}, using fallback knowledge")
            return self._get_fallback_knowledge(profile.material_system)

    @staticmethod
    def _get_fallback_knowledge(material_system: str) -> str:
        """当 RAG 不可用时, 使用预设冶金学知识."""
        for key, knowledge in FALLBACK_KNOWLEDGE.items():
            if key.lower() in material_system.lower():
                return knowledge
        return FALLBACK_GENERIC

    @staticmethod
    def _deduplicate_hits(hits: list[dict]) -> list[dict]:
        """按 paper_title 去重, 保留最高分的."""
        seen = {}
        for h in hits:
            title = h.get("paper_title", "") or h.get("source_file", "")
            if title not in seen or h.get("score", 0) > seen[title].get("score", 0):
                seen[title] = h
        return sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)

    @staticmethod
    def _format_snippets(hits: list[dict], max_chars_per_hit: int = 500) -> str:
        """格式化 RAG 命中的文献片段 (复刻 reviewer.py 的模式)."""
        if not hits:
            return "(no relevant literature snippets)"
        lines = []
        for i, h in enumerate(hits, 1):
            title = (h.get("paper_title") or h.get("source_file") or "").strip()
            section = h.get("section", "")
            score = h.get("score", 0.0)
            text = h.get("text", "").replace("\n", " ").strip()
            if len(text) > max_chars_per_hit:
                text = text[:max_chars_per_hit] + "..."
            lines.append(
                f"[{i}] ({section}, sim={score:.3f}) {title}\n    {text}"
            )
        return "\n\n".join(lines)

    # ── Prompt 格式化 ──────────────────────────────────────────────────────

    def _format_column_descriptions(self, profile: DatasetProfile, target: str) -> str:
        """格式化列描述表 (注入 LLM prompt)."""
        lines = []
        # Elements first
        unit_label = (
            "atomic percent" if profile.composition_unit == "at.%"
            else "weight percent"
        )
        lines.append(
            f"### Element Composition Columns ({profile.composition_unit}; "
            f"{unit_label})"
        )
        lines.append(
            f"  Interpret every elemental value as {unit_label}; do not "
            "reinterpret or convert it to the other composition basis."
        )
        for col in profile.element_cols:
            if col in profile.column_ranges:
                r = profile.column_ranges[col]
                lines.append(f"  {col}: [{r['min']:.2f}, {r['max']:.2f}] mean={r['mean']:.2f} std={r['std']:.2f}")
            else:
                lines.append(f"  {col}: present")

        # Process params
        if profile.process_cols:
            lines.append("### Process/Input Columns")
            for col in profile.process_cols:
                if col in profile.column_ranges:
                    r = profile.column_ranges[col]
                    lines.append(f"  {col}: [{r['min']:.1f}, {r['max']:.1f}] mean={r['mean']:.1f}")
                else:
                    lines.append(f"  {col}: present")

        # Target
        if target in profile.target_ranges:
            r = profile.target_ranges[target]
            lines.append(f"### Target Column ({target})")
            lines.append(f"  {target}: [{r['min']:.1f}, {r['max']:.1f}] mean={r['mean']:.1f}")

        return "\n".join(lines)

    @staticmethod
    def _format_target_description(profile: DatasetProfile, target: str) -> str:
        """Format the target-property description injected into the prompt."""
        desc = f"Target label: {target}"
        if target in profile.target_ranges:
            r = profile.target_ranges[target]
            desc += f", observed range [{r['min']:.1f}, {r['max']:.1f}], mean {r['mean']:.1f}"
        desc += (
            "\nInfer the property semantics and controlling mechanisms from this "
            "target label, the available columns, value ranges, and RAG evidence. "
            "Do not assume a fixed alloy family or target mechanism unless it is "
            "supported by the provided context."
        )
        return desc

    @staticmethod
    def _format_few_shot(profile: DatasetProfile, target: str) -> str:
        """Return system-agnostic few-shot guidance."""
        return (
            "No alloy-family-specific examples are provided. Design features from "
            "the current dataset columns, target label, validation feedback, and "
            "retrieved evidence only."
        )

    @staticmethod
    def _build_round_feedback(
        prev_round: dict,
        target: str,
        current_round: int,
        existing_names: set[str],
    ) -> str:
        """构建上一轮的反馈文本."""
        passed = [f for f in prev_round["features"] if f["passed"]]
        failed = [f for f in prev_round["features"] if not f["passed"]]

        passed_summary = "\n".join(
            f"  [OK] {f['name']}: {f['formula']} - {f['rationale'][:60]}"
            for f in passed
        ) if passed else "  (none)"

        failed_summary = "\n".join(
            f"  [FAIL] {f['name']}: {f['formula']} - error: {f['error'][:80]}"
            for f in failed
        ) if failed else "  (none)"

        existing_list = ", ".join(sorted(existing_names)) if existing_names else "(none)"

        return ROUND_FEEDBACK_TEMPLATE.format(
            prev_round=prev_round["round"],
            n_passed=len(passed),
            n_failed=len(failed),
            passed_summary=passed_summary,
            failed_summary=failed_summary,
            current_round=current_round,
            target_name=target,
            existing_names=existing_list,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 3: Formula Validation & Computation
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_and_validate_batch(
        self,
        df: pd.DataFrame,
        formulas: list[FeatureFormula],
        existing_df: pd.DataFrame,
        available_cols: set[str],
    ) -> list[ValidationResult]:
        """批量计算并验证特征公式."""
        # Prepare safe namespace
        ns = {"np": np, "__builtins__": {}}
        for col in available_cols:
            if col not in df.columns:
                continue
            ns[col] = pd.to_numeric(df[col], errors="coerce").values

        results = []
        for feat in formulas:
            # 1) Safety check
            safe, error = validate_formula_safety(feat.formula, available_cols)
            if not safe:
                results.append(ValidationResult(feat, False, error))
                continue

            # 2) Compute
            try:
                computed = eval(feat.formula, {"__builtins__": {}}, ns)
                if isinstance(computed, (int, float)):
                    computed = np.full(len(df), float(computed))
                series = pd.Series(np.asarray(computed, dtype=np.float64).ravel(), index=df.index)
            except Exception:
                results.append(ValidationResult(
                    feat, False, f"Eval error: {traceback.format_exc(limit=0).split(chr(10))[-2].strip()}"
                ))
                continue

            # 3) Every candidate must be finite for every pool row.  Silent
            # formula-specific imputation would make the representation depend
            # on undocumented behavior and can break GPR downstream.
            finite_mask = np.isfinite(series.to_numpy(dtype=np.float64))
            if not bool(finite_mask.all()):
                missing_fraction = float(1.0 - finite_mask.mean())
                results.append(
                    ValidationResult(
                        feat,
                        False,
                        f"Non-finite output fraction={missing_fraction:.6f}",
                    )
                )
                continue

            # 4) Zero variance check
            if series.nunique() <= 1 or series.std() < 1e-12:
                results.append(ValidationResult(feat, False, "Zero or near-zero variance"))
                continue

            # 5) Collinearity check with existing features
            if existing_df is not None and len(existing_df.columns) > 0:
                for ec in existing_df.columns:
                    if ec in df.columns:
                        existing_vals = pd.to_numeric(df[ec], errors="coerce").fillna(0)
                        corr = abs(series.fillna(0).corr(existing_vals))
                        if corr > 0.98:
                            results.append(ValidationResult(
                                feat, False, f"Collinear (r={corr:.3f}) with existing column '{ec}'"
                            ))
                            break
                else:
                    results.append(ValidationResult(feat, True, "OK", series))
            else:
                results.append(ValidationResult(feat, True, "OK", series))

        return results

    def _compute_feature_df(
        self,
        df: pd.DataFrame,
        formulas: list[FeatureFormula],
        base_cols: list[str],
    ) -> pd.DataFrame:
        """计算所有通过验证的特征并组装为 DataFrame."""
        available_cols = set(base_cols)
        ns = {"np": np, "__builtins__": {}}
        for col in base_cols:
            if col not in df.columns:
                continue
            ns[col] = pd.to_numeric(df[col], errors="coerce").values

        result = pd.DataFrame(index=df.index)
        for feat in formulas:
            safe, _ = validate_formula_safety(feat.formula, available_cols)
            if not safe:
                continue
            try:
                computed = eval(feat.formula, {"__builtins__": {}}, ns)
                if isinstance(computed, (int, float)):
                    computed = np.full(len(df), float(computed))
                series = pd.Series(np.asarray(computed, dtype=np.float64).ravel(), index=df.index)
                if (
                    not np.isfinite(series.to_numpy(dtype=np.float64)).all()
                    or series.std() < 1e-12
                ):
                    continue
                result[feat.name] = series
            except Exception:
                continue

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 4: Utility Filtering
    # ═══════════════════════════════════════════════════════════════════════════

    def _filter_by_utility(
        self,
        df_base: pd.DataFrame,
        df_llm: pd.DataFrame,
        y: pd.Series,
        target_name: str,
        n_folds: int = 5,
        top_k: int = 25,
        freq_threshold: float = 0.50,
    ) -> list[str]:
        """Optionally filter LLM candidates with null-calibrated OOF importance.

        This is not used by the publication default.  It remains available for
        explicit ablations and deliberately returns no feature when the model
        cannot establish positive out-of-fold utility.
        """
        llm_cols = list(df_llm.columns)
        if not llm_cols:
            return []

        # ── Combine features ──
        x = pd.concat([
            df_base.reset_index(drop=True),
            df_llm.reset_index(drop=True),
        ], axis=1)

        # ── Drop rows with NaN target ──
        valid_mask = y.notna()
        x = x.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)

        if len(y) < 12 or y.nunique(dropna=True) < 3:
            self._log(
                f"  Utility filter invalid at n={len(y)}; selecting zero "
                "LLM features"
            )
            return []

        # ── Encode categorical columns, then impute ──
        from sklearn.preprocessing import OrdinalEncoder
        obj_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
        if obj_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            x[obj_cols] = enc.fit_transform(x[obj_cols].astype(str))
        imp = SimpleImputer(strategy="median")
        x_imp = pd.DataFrame(imp.fit_transform(x), columns=x.columns, index=x.index)
        x_imp = x_imp.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ── Importance-guided selection (from LightGBM) ──
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            self._log("  LightGBM not available; selecting zero LLM features")
            return []

        n_splits = min(int(n_folds), max(2, len(y) // 6))
        derived_cap = min(
            int(top_k),
            max(1, len(y) // 2),
            len(llm_cols),
        )
        try:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            freq = pd.Series(0.0, index=llm_cols)
            fold_importances: list[pd.Series] = []
            rng = np.random.RandomState(self.random_state)

            for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(x_imp)):
                x_tr, x_va = x_imp.iloc[tr_idx], x_imp.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                min_child = max(2, min(10, len(tr_idx) // 4))
                num_leaves = max(3, min(15, len(tr_idx) // min_child + 1))

                fold_model = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=num_leaves,
                    min_child_samples=min_child,
                    random_state=self.random_state + fold_idx,
                    verbosity=-1,
                )
                fold_model.fit(x_tr, y_tr)
                observed = permutation_importance(
                    fold_model,
                    x_va,
                    y_va,
                    n_repeats=10,
                    random_state=self.random_state + fold_idx,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1,
                )
                observed_mean = pd.Series(
                    observed.importances_mean,
                    index=x_imp.columns,
                )

                null_model = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=num_leaves,
                    min_child_samples=min_child,
                    random_state=self.random_state + 1000 + fold_idx,
                    verbosity=-1,
                )
                null_model.fit(x_tr, rng.permutation(y_tr.to_numpy()))
                null_result = permutation_importance(
                    null_model,
                    x_va,
                    rng.permutation(y_va.to_numpy()),
                    n_repeats=10,
                    random_state=self.random_state + 1000 + fold_idx,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1,
                )
                null_llm = pd.Series(
                    null_result.importances_mean,
                    index=x_imp.columns,
                ).loc[llm_cols]
                null_threshold = max(
                    0.0,
                    float(np.quantile(null_llm.to_numpy(dtype=float), 0.95)),
                )
                eligible = observed_mean.loc[llm_cols]
                eligible = eligible[
                    np.isfinite(eligible)
                    & (eligible > 0.0)
                    & (eligible > null_threshold)
                ]
                top_in_fold = eligible.sort_values(
                    ascending=False,
                ).head(derived_cap)
                freq.loc[top_in_fold.index] += 1.0
                fold_importances.append(observed_mean.loc[llm_cols])

            freq = freq / n_splits
            mean_importance = pd.concat(
                fold_importances,
                axis=1,
            ).mean(axis=1)
            final_selected = [
                col
                for col in llm_cols
                if freq.get(col, 0.0) >= float(freq_threshold)
                and np.isfinite(mean_importance.get(col, np.nan))
                and mean_importance.get(col, 0.0) > 0.0
            ]
            return sorted(
                final_selected,
                key=lambda col: (
                    -float(freq[col]),
                    -float(mean_importance[col]),
                    col,
                ),
            )

        except Exception as e:
            self._log(
                f"  Null-calibrated utility filtering failed: {e}; "
                "selecting zero LLM features"
            )
            return []

    # ── 日志 ────────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[LLMFeatDesign] {msg}", flush=True)
