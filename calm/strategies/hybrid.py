"""Strategy Hybrid: soft heterogeneous ensemble UCB + LLM knowledge.

GBDT, RFR, GPR, and NN are combined with periodically updated OOF weights.
Predictive variance contains both each model's uncertainty and cross-model
disagreement, then its UCB is additively combined with the LLM score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from calm.experiment import DYNAMIC_RANK_FUSION_VERSION

from .base import AcquisitionStrategy

PROJECT = Path(__file__).resolve().parent.parent.parent

MC_SAMPLES = 30
CV_FOLDS = 5
GBDT_N_ESTIMATORS = 300
RFR_N_ESTIMATORS = 200
SIGMA_CALIBRATION_QUANTILE = 0.95
SIGMA_CALIBRATION_Z = 1.96
SIGMA_SCALE_MIN = 0.10
SIGMA_SCALE_MAX = 10.0


class HybridAcquisition(AcquisitionStrategy):
    """Soft heterogeneous ensemble UCB with additive LLM knowledge."""

    name = "Hybrid"
    uses_llm_knowledge = True
    supports_dynamic_rescoring = True
    needs_al_feedback_context = True
    uses_soft_ensemble = True

    SURROGATE_POOL = ["gbdt", "rfr", "gpr", "nn"]

    def __init__(
        self,
        alpha: float = 1.96,
        lam_llm: float = 0.6,
        ensemble_min_train_samples: int = 40,
        ensemble_weight_update_samples: int = 25,
        ensemble_weight_floor: float = 0.05,
        ensemble_weight_temperature: float = 0.35,
        ensemble_top_weight_strength: float = 2.0,
        gbdt_bootstrap_models: int = 5,
        extreme_rank_power: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.lam_llm = lam_llm
        extreme_rank_power = float(extreme_rank_power)
        if not np.isfinite(extreme_rank_power) or extreme_rank_power < 0.0:
            raise ValueError("extreme_rank_power must be finite and non-negative")
        self.extreme_rank_power = extreme_rank_power
        self._last_llm_rho: float = 0.0
        self._last_llm_extreme_rho: float = 0.0
        self._last_lambda_eff: float = 0.0

        self.ensemble_min_train_samples = max(10, int(ensemble_min_train_samples))
        self.ensemble_weight_update_samples = max(
            1, int(ensemble_weight_update_samples)
        )
        self.ensemble_weight_floor = min(
            0.20, max(0.0, float(ensemble_weight_floor))
        )
        self.ensemble_weight_temperature = max(
            1e-6, float(ensemble_weight_temperature)
        )
        self.ensemble_top_weight_strength = max(
            0.0, float(ensemble_top_weight_strength)
        )
        self.gbdt_bootstrap_models = max(2, int(gbdt_bootstrap_models))

        # Fixed-surrogate compatibility state used only by NN-UCB.
        self._best_surrogate: str | None = None
        self._cv_scores: dict[str, float] = {}
        self._oof_weighted_mae: dict[str, float] = {}
        self._surrogate_sigma_scales: dict[str, float] = {
            name: 1.0 for name in self.SURROGATE_POOL
        }
        self._oof_uncertainty_diagnostics: dict[str, dict[str, float]] = {}
        self._surrogate_weights: dict[str, float] = {
            name: 1.0 / len(self.SURROGATE_POOL)
            for name in self.SURROGATE_POOL
        }
        self._last_weight_update_train_size: int = 0
        self._last_ensemble_disagreement_mean: float = 0.0
        self._last_ensemble_within_sigma_mean: float = 0.0

        # NN 专用
        self._nn_model: nn.Module | None = None
        self._nn_feature_cols: list[str] | None = None
        self._nn_imputer: SimpleImputer | None = None
        self._nn_scaler: StandardScaler | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._device: torch.device | None = None

        # 树模型专用
        self._gbdt_model: LGBMRegressor | None = None
        self._gbdt_models: list[LGBMRegressor] = []
        self._rfr_model: RandomForestRegressor | None = None
        self._gpr_model: GaussianProcessRegressor | None = None
        self._tree_imputer: SimpleImputer | None = None
        self._gpr_scaler: StandardScaler | None = None

        # Dynamic LLM rescoring is configured by runner.py and triggered inside
        # Hybrid during select_next(), so the strategy owns the knowledge update.
        self._dynamic_scorer = None
        self._rescoring_interval: int = 0
        self._dynamic_top_m_frac: float = 0.25
        self._dynamic_top_m_min_mult: int = 5
        self._dynamic_top_m_cap: int = 200
        self._dynamic_random_frac: float = 0.10
        self._al_iteration: int = 0
        self._pool_indices: np.ndarray | None = None
        self._train_idx: np.ndarray | None = None
        self._y_all: np.ndarray | None = None
        self._all_scores: dict[str, np.ndarray] | None = None
        self._score_overrides: dict[str, dict[int, float]] = {
            "s_phys": {},
        }
        self._dynamic_score_cache: dict[tuple[str, int], float] = {}
        self._last_dynamic_subset_local = np.array([], dtype=int)

    def configure_dynamic_rescoring(
        self,
        scorer,
        interval: int,
        *,
        top_m_frac: float = 0.25,
        top_m_min_mult: int = 5,
        top_m_cap: int = 200,
        random_frac: float = 0.10,
    ) -> None:
        """Attach a DynamicLLMScorer for periodic pool rescoring."""
        if not self.supports_dynamic_rescoring:
            raise RuntimeError(f"{self.name} does not support dynamic LLM rescoring")
        self._dynamic_scorer = scorer
        self._rescoring_interval = max(0, int(interval))
        self._dynamic_top_m_frac = max(0.0, float(top_m_frac))
        self._dynamic_top_m_min_mult = max(1, int(top_m_min_mult))
        self._dynamic_top_m_cap = max(1, int(top_m_cap))
        self._dynamic_random_frac = min(1.0, max(0.0, float(random_frac)))

    def reset_run_state(self) -> None:
        """Clear per-run AL feedback so independent seeds do not share LLM updates."""
        if self._dynamic_scorer is not None and hasattr(
            self._dynamic_scorer, "reset_run_state"
        ):
            self._dynamic_scorer.reset_run_state()
        self._score_overrides = {
            "s_phys": {},
        }
        self._dynamic_score_cache = {}
        self._al_iteration = 0
        self._pool_indices = None
        self._train_idx = None
        self._y_all = None
        self._all_scores = None
        self._last_dynamic_subset_local = np.array([], dtype=int)
        self._last_llm_rho = 0.0
        self._last_llm_extreme_rho = 0.0
        self._last_lambda_eff = 0.0
        self._cv_scores = {}
        self._oof_weighted_mae = {}
        self._surrogate_sigma_scales = {
            name: 1.0 for name in self.SURROGATE_POOL
        }
        self._oof_uncertainty_diagnostics = {}
        self._surrogate_weights = {
            name: 1.0 / len(self.SURROGATE_POOL)
            for name in self.SURROGATE_POOL
        }
        self._last_weight_update_train_size = 0
        self._last_ensemble_disagreement_mean = 0.0
        self._last_ensemble_within_sigma_mean = 0.0

    def export_run_state(self) -> dict:
        """Persist dynamic LLM calibration state between interrupted iterations."""
        return {
            "score_overrides": {
                key: {str(index): float(value) for index, value in values.items()}
                for key, values in self._score_overrides.items()
            },
            "dynamic_score_cache": [
                [fingerprint, int(index), float(value)]
                for (fingerprint, index), value in self._dynamic_score_cache.items()
            ],
            "last_llm_rho": float(self._last_llm_rho),
            "last_llm_extreme_rho": float(self._last_llm_extreme_rho),
            "last_lambda_eff": float(self._last_lambda_eff),
            "surrogate_weights": {
                key: float(value) for key, value in self._surrogate_weights.items()
            },
            "cv_scores": {
                key: float(value) for key, value in self._cv_scores.items()
            },
            "oof_weighted_mae": {
                key: float(value) for key, value in self._oof_weighted_mae.items()
            },
            "surrogate_sigma_scales": {
                key: float(value)
                for key, value in self._surrogate_sigma_scales.items()
            },
            "oof_uncertainty_diagnostics": self._oof_uncertainty_diagnostics,
            "last_weight_update_train_size": int(
                self._last_weight_update_train_size
            ),
            "dynamic_scorer_state": (
                self._dynamic_scorer.export_run_state()
                if self._dynamic_scorer is not None
                and hasattr(self._dynamic_scorer, "export_run_state")
                else {}
            ),
        }

    def import_run_state(self, state: dict) -> None:
        """Restore dynamic scores; fitted surrogate objects are rebuilt each iteration."""
        self._score_overrides = {
            str(key): {int(index): float(value) for index, value in values.items()}
            for key, values in state.get("score_overrides", {"s_phys": {}}).items()
        }
        self._score_overrides.setdefault("s_phys", {})
        self._dynamic_score_cache = {
            (str(fingerprint), int(index)): float(value)
            for fingerprint, index, value in state.get("dynamic_score_cache", [])
        }
        self._last_llm_rho = float(state.get("last_llm_rho", 0.0))
        self._last_llm_extreme_rho = float(
            state.get("last_llm_extreme_rho", 0.0)
        )
        self._last_lambda_eff = float(state.get("last_lambda_eff", 0.0))
        restored_weights = state.get("surrogate_weights", {})
        if restored_weights:
            self._surrogate_weights = {
                name: float(restored_weights.get(name, 0.0))
                for name in self.SURROGATE_POOL
            }
        self._cv_scores = {
            str(key): float(value)
            for key, value in state.get("cv_scores", {}).items()
        }
        self._oof_weighted_mae = {
            str(key): float(value)
            for key, value in state.get("oof_weighted_mae", {}).items()
        }
        restored_scales = state.get("surrogate_sigma_scales", {})
        self._surrogate_sigma_scales = {
            name: float(restored_scales.get(name, 1.0))
            for name in self.SURROGATE_POOL
        }
        self._oof_uncertainty_diagnostics = {
            str(name): {
                str(key): float(value)
                for key, value in diagnostics.items()
            }
            for name, diagnostics in state.get(
                "oof_uncertainty_diagnostics", {}
            ).items()
        }
        self._last_weight_update_train_size = int(
            state.get("last_weight_update_train_size", 0)
        )
        if self._dynamic_scorer is not None and hasattr(
            self._dynamic_scorer, "import_run_state"
        ):
            self._dynamic_scorer.import_run_state(
                state.get("dynamic_scorer_state", {})
            )

    def set_al_context(
        self,
        *,
        iteration: int,
        pool_indices: np.ndarray,
        train_idx: np.ndarray,
        y_all: np.ndarray,
        all_scores: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Receive current AL state before candidate selection."""
        self._al_iteration = int(iteration)
        self._pool_indices = np.asarray(pool_indices)
        self._train_idx = np.asarray(train_idx)
        self._y_all = np.asarray(y_all)
        self._all_scores = all_scores

    # ═══════════════════════════════════════════════════════════════════
    # fit: 训练所有候选代理模型 → 定期 OOF 评估 → 软权重组合
    # ═══════════════════════════════════════════════════════════════════

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._train_gbdt(X, y)
        self._train_rfr(X, y)
        self._train_gpr(X, y)
        self._train_nn(X, y)

        self._update_surrogate_weights(X, y)
        rounded_weights = {
            key: round(value, 3)
            for key, value in self._surrogate_weights.items()
        }
        rounded_losses = {
            key: round(value, 3)
            for key, value in self._oof_weighted_mae.items()
        }
        rounded_sigma_scales = {
            key: round(value, 3)
            for key, value in self._surrogate_sigma_scales.items()
        }
        print(
            f"[{self.name}] Soft ensemble weights: {rounded_weights} "
            f"(OOF weighted MAE={rounded_losses}, "
            f"sigma scales={rounded_sigma_scales})"
        )

    # ── GBDT ──────────────────────────────────────────────────────────

    def _train_gbdt(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._tree_imputer = SimpleImputer(strategy="median")
        X_imp = self._tree_imputer.fit_transform(X.values)
        n = len(y)
        self._gbdt_models = []
        for seed in range(self.gbdt_bootstrap_models):
            rng = np.random.RandomState(10_000 + seed)
            bootstrap = rng.choice(n, size=n, replace=True)
            model = LGBMRegressor(
                n_estimators=GBDT_N_ESTIMATORS, learning_rate=0.05,
                num_leaves=min(31, max(2, n // 3)),
                min_child_samples=min(20, max(1, n // 8)),
                random_state=42 + seed, n_jobs=-1, verbosity=-1,
            )
            model.fit(X_imp[bootstrap], y.ravel()[bootstrap])
            self._gbdt_models.append(model)
        self._gbdt_model = self._gbdt_models[0]

    # ── RFR ───────────────────────────────────────────────────────────

    def _train_rfr(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._rfr_model = RandomForestRegressor(
            n_estimators=RFR_N_ESTIMATORS, random_state=0, n_jobs=-1,
        )
        self._rfr_model.fit(X.values, y.ravel())

    # ── GPR ───────────────────────────────────────────────────────────

    def _train_gpr(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._gpr_scaler = StandardScaler()
        X_scaled = self._gpr_scaler.fit_transform(X.values)
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self._gpr_model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=0,
        )
        try:
            self._gpr_model.fit(X_scaled, y.ravel())
        except Exception:
            self._gpr_model = None  # 样本太少时 GP 可能失败

    # ── NN ────────────────────────────────────────────────────────────

    @staticmethod
    def _nn_train_config():
        """One NN protocol shared by production fits and every OOF fold."""
        from common.nn_ensemble.model import TrainConfig

        return TrainConfig(
            target="Strength", epochs=100, lr=3e-3, weight_decay=1e-3,
            mlp_hidden_dims=(64, 32, 16), dropout=0.20, input_noise=0.0,
            seed=42, early_stop_patience=20, warmup_epochs=5,
        )

    @staticmethod
    def _enable_mc_dropout(model: nn.Module) -> None:
        """Enable dropout sampling without re-enabling training-only input noise."""
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _train_nn(self, X: pd.DataFrame, y: np.ndarray) -> None:
        from common.nn_ensemble.model import (
            SimpleMLPRegressor,
            set_global_seed,
        )

        set_global_seed(42)
        self._nn_feature_cols = list(X.columns)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._nn_imputer = SimpleImputer(strategy="median")
        self._nn_scaler = StandardScaler()
        X_proc = self._nn_imputer.fit_transform(X[self._nn_feature_cols].values)
        X_proc = self._nn_scaler.fit_transform(X_proc)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32)

        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) if y.std() > 0 else 1.0
        y_norm = (y - self._y_mean) / self._y_std

        cfg = self._nn_train_config()

        model = SimpleMLPRegressor(
            input_dim=X_tensor.shape[1],
            hidden_dims=cfg.mlp_hidden_dims,
            dropout=cfg.dropout,
        ).to(self._device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs))

        y_t = torch.tensor(y_norm, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_tensor, y_t),
                            batch_size=cfg.batch_size, shuffle=True)

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        model.train()
        for epoch in range(cfg.epochs):
            if epoch < cfg.warmup_epochs:
                lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
                for pg in opt.param_groups:
                    pg["lr"] = lr
            else:
                scheduler.step()

            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                pred = model(xb)
                loss = nn.functional.huber_loss(pred, yb, delta=cfg.huber_delta)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                opt.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._nn_model = model

    # ═══════════════════════════════════════════════════════════════════
    # OOF 评估: 高值加权误差 → 平滑代理权重
    # ═══════════════════════════════════════════════════════════════════

    def _available_surrogates(self) -> dict[str, object]:
        return {
            "gbdt": self._gbdt_model,
            "rfr": self._rfr_model,
            "gpr": self._gpr_model,
            "nn": self._nn_model,
        }

    def _set_uniform_available_weights(self) -> None:
        available = [
            name for name, model in self._available_surrogates().items()
            if model is not None
        ]
        if not available:
            raise RuntimeError(f"{self.name}: no fitted surrogate is available")
        self._surrogate_weights = {
            name: (1.0 / len(available) if name in available else 0.0)
            for name in self.SURROGATE_POOL
        }

    def _update_surrogate_weights(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Update soft weights from complete OOF predictions at stable intervals."""
        n = len(y)
        available_models = self._available_surrogates()
        available = [name for name, model in available_models.items() if model is not None]
        if not available:
            raise RuntimeError(f"{self.name}: no fitted surrogate is available")

        if n < self.ensemble_min_train_samples:
            self._set_uniform_available_weights()
            self._cv_scores = {}
            self._oof_weighted_mae = {}
            self._surrogate_sigma_scales = {
                name: 1.0 for name in self.SURROGATE_POOL
            }
            self._oof_uncertainty_diagnostics = {}
            return

        due = (
            self._last_weight_update_train_size < self.ensemble_min_train_samples
            or n - self._last_weight_update_train_size
            >= self.ensemble_weight_update_samples
        )
        if not due:
            total = sum(
                self._surrogate_weights.get(name, 0.0) for name in available
            )
            if total <= 0:
                self._set_uniform_available_weights()
            else:
                self._surrogate_weights = {
                    name: (
                        self._surrogate_weights.get(name, 0.0) / total
                        if name in available else 0.0
                    )
                    for name in self.SURROGATE_POOL
                }
            return

        n_folds = min(CV_FOLDS, max(3, n // 10))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_mu = {
            name: np.full(n, np.nan, dtype=np.float64)
            for name in available
        }
        oof_sigma = {
            name: np.full(n, np.nan, dtype=np.float64)
            for name in available
        }
        for train_i, val_i in kf.split(X):
            X_train, y_train = X.iloc[train_i], y[train_i]
            X_val = X.iloc[val_i]
            for name in available:
                try:
                    mu, sigma = self._surrogate_predict(
                        name,
                        available_models[name],
                        X_train,
                        y_train,
                        X_val,
                    )
                    oof_mu[name][val_i] = np.asarray(mu, dtype=np.float64)
                    oof_sigma[name][val_i] = np.asarray(
                        sigma, dtype=np.float64
                    )
                except Exception as exc:
                    print(f"[{self.name}] OOF fold failed for {name}: {exc}")

        y_float = np.asarray(y, dtype=np.float64)
        y_scale = float(np.std(y_float))
        if y_scale <= 1e-12:
            y_scale = 1.0
        target_percentile = self._rank_percentile(y_float)
        sample_weight = (
            1.0
            + self.ensemble_top_weight_strength * np.square(target_percentile)
        )

        losses: dict[str, float] = {}
        scores: dict[str, float] = {}
        sigma_scales = {
            name: 1.0 for name in self.SURROGATE_POOL
        }
        uncertainty_diagnostics: dict[str, dict[str, float]] = {}
        for name in available:
            prediction = oof_mu[name]
            raw_sigma = oof_sigma[name]
            finite = np.isfinite(prediction) & np.isfinite(raw_sigma)
            if finite.sum() != n:
                losses[name] = 1e6
                scores[name] = -1.0
                continue
            absolute_error = np.abs(prediction - y_float)
            losses[name] = float(
                np.average(absolute_error, weights=sample_weight) / y_scale
            )
            if np.std(prediction) > 1e-12 and np.std(y_float) > 1e-12:
                rho, _ = spearmanr(prediction, y_float)
            else:
                rho = 0.0
            scores[name] = float(rho) if rho is not None and np.isfinite(rho) else 0.0

            sigma_floor = max(y_scale * 1e-6, 1e-9)
            raw_sigma = np.maximum(raw_sigma, sigma_floor)
            ratio = absolute_error / raw_sigma
            scale = float(
                np.quantile(ratio, SIGMA_CALIBRATION_QUANTILE)
                / SIGMA_CALIBRATION_Z
            )
            scale = float(np.clip(scale, SIGMA_SCALE_MIN, SIGMA_SCALE_MAX))
            calibrated_sigma = raw_sigma * scale
            sigma_scales[name] = scale
            uncertainty_diagnostics[name] = {
                "n_oof": float(n),
                "scale": scale,
                "raw_95_coverage": float(
                    np.mean(absolute_error <= SIGMA_CALIBRATION_Z * raw_sigma)
                ),
                "calibrated_95_coverage": float(
                    np.mean(
                        absolute_error
                        <= SIGMA_CALIBRATION_Z * calibrated_sigma
                    )
                ),
                "raw_sigma_mean": float(np.mean(raw_sigma)),
                "calibrated_sigma_mean": float(np.mean(calibrated_sigma)),
            }

        finite_names = [name for name in available if np.isfinite(losses[name])]
        if not finite_names:
            self._set_uniform_available_weights()
            self._cv_scores = scores
            self._oof_weighted_mae = losses
            self._surrogate_sigma_scales = sigma_scales
            self._oof_uncertainty_diagnostics = uncertainty_diagnostics
            self._last_weight_update_train_size = n
            return

        loss_values = np.asarray([losses[name] for name in finite_names])
        logits = -(loss_values - float(loss_values.min())) / self.ensemble_weight_temperature
        logits -= float(logits.max())
        soft = np.exp(logits)
        soft /= float(soft.sum())
        floor = min(self.ensemble_weight_floor, 0.99 / len(finite_names))
        soft = floor + (1.0 - floor * len(finite_names)) * soft
        weighted = dict(zip(finite_names, soft.tolist()))
        self._surrogate_weights = {
            name: float(weighted.get(name, 0.0))
            for name in self.SURROGATE_POOL
        }
        self._cv_scores = scores
        self._oof_weighted_mae = losses
        self._surrogate_sigma_scales = sigma_scales
        self._oof_uncertainty_diagnostics = uncertainty_diagnostics
        self._last_weight_update_train_size = n

    def _surrogate_predict(
        self,
        sname: str,
        model,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_pool: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对单个代理模型做快速训练+预测, 返回 (μ, σ)."""
        if sname == "gbdt":
            imp = SimpleImputer(strategy="median")
            Xt = imp.fit_transform(X_train.values)
            Xp = imp.transform(X_pool.values)
            n = len(y_train)
            predictions = []
            for seed in range(self.gbdt_bootstrap_models):
                bootstrap_rng = np.random.RandomState(10_000 + seed)
                bootstrap = bootstrap_rng.choice(n, size=n, replace=True)
                fitted = LGBMRegressor(
                    n_estimators=GBDT_N_ESTIMATORS,
                    learning_rate=0.05,
                    num_leaves=min(31, max(2, n // 3)),
                    min_child_samples=min(20, max(1, n // 8)),
                    random_state=42 + seed,
                    n_jobs=-1,
                    verbosity=-1,
                )
                fitted.fit(Xt[bootstrap], y_train.ravel()[bootstrap])
                predictions.append(fitted.predict(Xp))
            predictions = np.asarray(predictions, dtype=np.float64)
            return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)

        elif sname == "rfr":
            m = RandomForestRegressor(
                n_estimators=RFR_N_ESTIMATORS,
                random_state=0,
                n_jobs=-1,
            )
            m.fit(X_train.values, y_train.ravel())
            preds = np.array([t.predict(X_pool.values) for t in m.estimators_])
            mu = preds.mean(axis=0)
            sigma = preds.std(axis=0, ddof=1)
            return mu, sigma

        elif sname == "gpr":
            sc = StandardScaler()
            Xt = sc.fit_transform(X_train.values)
            Xp = sc.transform(X_pool.values)
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            m = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
            m.fit(Xt, y_train.ravel())
            mu, sigma = m.predict(Xp, return_std=True)
            return mu.ravel(), sigma

        elif sname == "nn":
            # NN CV: 简化版, 重用主训练逻辑
            return self._nn_cv_predict(X_train, y_train, X_pool)

        raise ValueError(f"Unknown surrogate: {sname}")

    def _nn_cv_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_pool: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NN CV fold 快速训练+预测."""
        from common.nn_ensemble.model import (
            SimpleMLPRegressor,
            set_global_seed,
        )
        set_global_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_cols = list(X_train.columns)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_proc = imputer.fit_transform(X_train[feature_cols].values)
        X_proc = scaler.fit_transform(X_proc)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32)

        ym, ys = float(y_train.mean()), float(y_train.std()) if y_train.std() > 0 else 1.0
        y_norm = (y_train - ym) / ys

        cfg = self._nn_train_config()
        model = SimpleMLPRegressor(
            input_dim=X_tensor.shape[1],
            hidden_dims=cfg.mlp_hidden_dims,
            dropout=cfg.dropout,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs)
        )
        y_t = torch.tensor(y_norm, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tensor, y_t),
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        patience = 0
        model.train()
        for epoch in range(cfg.epochs):
            if epoch < cfg.warmup_epochs:
                lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
                for param_group in opt.param_groups:
                    param_group["lr"] = lr
            else:
                scheduler.step()

            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = nn.functional.huber_loss(pred, yb, delta=cfg.huber_delta)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip_norm
                )
                opt.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break

        # MC prediction on pool
        if best_state is not None:
            model.load_state_dict(best_state)
        Xp_proc = imputer.transform(X_pool[feature_cols].values)
        Xp_proc = scaler.transform(Xp_proc)
        Xp_tensor = torch.tensor(Xp_proc, dtype=torch.float32).to(device)

        self._enable_mc_dropout(model)
        mc = []
        with torch.no_grad():
            for _ in range(MC_SAMPLES):
                p = model(Xp_tensor).cpu().numpy() * ys + ym
                mc.append(p)
        mc = np.array(mc)
        return mc.mean(axis=0), mc.std(axis=0, ddof=1)

    # ═══════════════════════════════════════════════════════════════════
    # select_next
    # ═══════════════════════════════════════════════════════════════════

    def select_next(
        self,
        pool_X: pd.DataFrame,
        pool_scores: dict[str, np.ndarray] | None,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        if self.uses_soft_ensemble:
            mu, sigma = self._ensemble_predict(pool_X)
        else:
            if self._best_surrogate is None:
                raise RuntimeError(f"{self.name} requires fit() before select_next()")
            mu, sigma = self._best_predict(pool_X)
        ucb = mu + self.alpha * sigma
        ucb_rank = self._rank_percentile(ucb)

        if not self.uses_llm_knowledge:
            # Adaptive-UCB and NN-UCB use the exact same finite-value and tie
            # handling as Hybrid's UCB component, while ignoring knowledge.
            acq = ucb_rank
        elif pool_scores is not None and "s_phys" in pool_scores:
            phys = np.asarray(pool_scores["s_phys"], dtype=np.float64)
            llm_score = phys
            tie_break = np.asarray(
                pool_scores.get("_tie_break_rank", np.arange(len(llm_score))),
                dtype=np.float64,
            )
            if tie_break.shape != llm_score.shape or not np.all(
                np.isfinite(tie_break)
            ):
                raise ValueError(
                    f"{self.name} received an invalid deterministic tie order"
                )

            # The immutable target-blind full-pool score vector is the common
            # knowledge coordinate used by both EWRC and rank-slot fusion.
            base_llm_rank = self._rank_percentile(
                llm_score,
                tie_break=tie_break,
            )
            lambda_eff = self._effective_lambda()
            static_acq = (
                (1.0 - lambda_eff) * ucb_rank
                + lambda_eff * base_llm_rank
            )
            pool_scores = self._maybe_dynamic_rescore(
                pool_scores=pool_scores,
                mu=mu,
                sigma=sigma,
                ucb_rank=ucb_rank,
                static_acq=static_acq,
                batch_size=n,
                rng=rng,
            )

            dynamic_score = np.asarray(pool_scores["s_phys"], dtype=np.float64)
            llm_rank = self._rerank_dynamic_subset_with_static_slots(
                static_llm_rank=base_llm_rank,
                dynamic_score=dynamic_score,
                subset_local=self._last_dynamic_subset_local,
            )

            acq = (1.0 - lambda_eff) * ucb_rank + lambda_eff * llm_rank
        else:
            raise RuntimeError(
                f"{self.name} requires pool_scores['s_phys']; use Adaptive-UCB for "
                "the matched data-only acquisition"
            )

        if self.uses_llm_knowledge:
            return np.lexsort((-tie_break, -acq))[:n]
        return np.argsort(-acq, kind="mergesort")[:n]

    def get_iteration_diagnostics(self) -> dict:
        """Return JSON-safe acquisition diagnostics for convergence auditing."""
        return {
            "surrogate_mode": (
                "soft_ensemble" if self.uses_soft_ensemble else "fixed"
            ),
            "surrogate_weights": {
                key: float(value)
                for key, value in self._surrogate_weights.items()
            },
            "oof_weighted_mae": {
                key: (
                    float(value) if np.isfinite(value) else None
                )
                for key, value in self._oof_weighted_mae.items()
            },
            "oof_spearman": {
                key: float(value) for key, value in self._cv_scores.items()
            },
            "surrogate_sigma_scales": {
                key: float(value)
                for key, value in self._surrogate_sigma_scales.items()
            },
            "oof_uncertainty_diagnostics": self._oof_uncertainty_diagnostics,
            "weight_fit_train_size": int(self._last_weight_update_train_size),
            "ensemble_disagreement_mean": float(
                self._last_ensemble_disagreement_mean
            ),
            "ensemble_within_sigma_mean": float(
                self._last_ensemble_within_sigma_mean
            ),
            "llm_rho": float(self._last_llm_rho),
            "llm_rho_global": float(self._last_llm_rho),
            "llm_rho_extreme": float(self._last_llm_extreme_rho),
            "ewrc_gamma": float(self.extreme_rank_power),
            "lambda_eff": float(self._last_lambda_eff),
            "dynamic_rank_fusion": DYNAMIC_RANK_FUSION_VERSION,
            "knowledge_base": "static_full_pool",
            "dynamic_rescore_subset_size": int(
                len(self._last_dynamic_subset_local)
            ),
        }

    @staticmethod
    def _rank_percentile(
        values: np.ndarray,
        tie_break: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert any score scale to rank percentiles; higher values stay better."""
        v = np.asarray(values, dtype=np.float64)
        if len(v) <= 1:
            return np.zeros_like(v)
        finite = np.isfinite(v)
        if not np.any(finite):
            return np.zeros_like(v)
        filled = v.copy()
        finite_min = float(np.nanmin(filled[finite]))
        filled[~finite] = np.nextafter(finite_min, -np.inf)
        if tie_break is None:
            ranks = rankdata(filled, method="average")
        else:
            tie = np.asarray(tie_break, dtype=np.float64)
            if tie.shape != filled.shape or not np.all(np.isfinite(tie)):
                raise ValueError("tie_break must be finite and match values")
            if len(np.unique(tie)) != len(tie):
                raise ValueError("tie_break must define a unique total order")
            order = np.lexsort((tie, filled))
            ranks = np.empty(len(filled), dtype=np.float64)
            ranks[order] = np.arange(1, len(filled) + 1, dtype=np.float64)
        return (ranks - 1.0) / max(len(v) - 1, 1)

    @staticmethod
    def _rerank_dynamic_subset_with_static_slots(
        *,
        static_llm_rank: np.ndarray,
        dynamic_score: np.ndarray,
        subset_local: np.ndarray,
    ) -> np.ndarray:
        """Use dynamic scores only to permute the subset's base rank slots.

        Static and feedback-conditioned LLM scores are not assumed to share an
        absolute calibration. Candidates outside the dynamically rescored
        subset therefore keep their static full-pool knowledge ranks. Inside
        the subset, dynamic scores determine which candidate receives each of
        the static rank slots already owned by that subset.

        Dynamic ties preserve the static order.  No calibration parameter is
        introduced, and the multiset of knowledge ranks is unchanged.
        """
        static_rank = np.asarray(static_llm_rank, dtype=np.float64)
        dynamic = np.asarray(dynamic_score, dtype=np.float64)
        subset = np.asarray(subset_local, dtype=int)

        if static_rank.shape != dynamic.shape:
            raise ValueError("static_llm_rank and dynamic_score must have the same shape")
        if static_rank.ndim != 1:
            raise ValueError("rank-slot permutation expects one-dimensional scores")

        result = static_rank.copy()
        if len(subset) == 0:
            return result
        if np.any(subset < 0) or np.any(subset >= len(static_rank)):
            raise IndexError("dynamic subset contains an out-of-range pool index")
        if len(np.unique(subset)) != len(subset):
            raise ValueError("dynamic subset indices must be unique")
        if not np.all(np.isfinite(static_rank[subset])):
            raise ValueError("dynamic subset has non-finite static rank slots")
        if not np.all(np.isfinite(dynamic[subset])):
            raise ValueError("dynamic subset has non-finite dynamic scores")

        subset_static_rank = static_rank[subset]
        subset_dynamic = dynamic[subset]
        # np.lexsort uses the final key as primary: dynamic score descending,
        # then static rank descending, then local index ascending.  The second
        # key makes a dynamic tie leave the original static order unchanged.
        dynamic_order = np.lexsort(
            (subset, -subset_static_rank, -subset_dynamic)
        )
        rank_slots = np.sort(subset_static_rank)[::-1]
        result[subset[dynamic_order]] = rank_slots
        return result

    @staticmethod
    def _extreme_weighted_rank_correlation(
        llm_scores: np.ndarray,
        targets: np.ndarray,
        gamma: float = 2.0,
    ) -> float:
        """Return an objective-best-tail-weighted Spearman correlation.

        Both inputs are converted to average ranks. The runner supplies an
        oriented utility whose larger values are always better; its percentile
        ``q_i = rank(utility_i) / n`` receives weight
        ``q_i ** gamma``.  ``gamma=0`` therefore recovers ordinary Spearman
        exactly, while larger values progressively emphasize the upper tail.
        """
        gamma = float(gamma)
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative")

        scores = np.asarray(llm_scores, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if scores.shape != y.shape:
            raise ValueError("llm_scores and targets must have the same shape")

        finite = np.isfinite(scores) & np.isfinite(y)
        scores = scores[finite]
        y = y[finite]
        n = len(y)
        if n < 2:
            return 0.0

        score_ranks = rankdata(scores, method="average").astype(np.float64)
        target_ranks = rankdata(y, method="average").astype(np.float64)
        weights = np.power(target_ranks / float(n), gamma)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            return 0.0

        score_mean = float(np.sum(weights * score_ranks) / weight_sum)
        target_mean = float(np.sum(weights * target_ranks) / weight_sum)
        score_centered = score_ranks - score_mean
        target_centered = target_ranks - target_mean
        covariance = float(
            np.sum(weights * score_centered * target_centered) / weight_sum
        )
        score_variance = float(
            np.sum(weights * np.square(score_centered)) / weight_sum
        )
        target_variance = float(
            np.sum(weights * np.square(target_centered)) / weight_sum
        )
        denominator = float(np.sqrt(score_variance * target_variance))
        if not np.isfinite(denominator) or denominator <= 1e-12:
            return 0.0

        rho = covariance / denominator
        if not np.isfinite(rho):
            return 0.0
        return float(np.clip(rho, -1.0, 1.0))

    def _effective_lambda(self) -> float:
        """Scale LLM influence by extreme-weighted rank agreement."""
        self._last_llm_rho = 0.0
        self._last_llm_extreme_rho = 0.0
        self._last_lambda_eff = 0.0
        if (
            self._all_scores is None
            or self._train_idx is None
            or self._y_all is None
            or "s_phys" not in self._all_scores
        ):
            return 0.0

        train_idx = np.asarray(self._train_idx, dtype=int)
        if len(train_idx) < 5:
            return 0.0

        phys = np.asarray(self._all_scores["s_phys"], dtype=np.float64)[train_idx]
        for pos, global_idx in enumerate(train_idx):
            idx = int(global_idx)
            if idx in self._score_overrides["s_phys"]:
                phys[pos] = self._score_overrides["s_phys"][idx]
        llm_train = phys
        y_train = np.asarray(self._y_all, dtype=np.float64)[train_idx]

        finite = np.isfinite(llm_train) & np.isfinite(y_train)
        if np.count_nonzero(finite) < 5:
            return 0.0
        llm_valid = llm_train[finite]
        y_valid = y_train[finite]
        if np.std(llm_valid) <= 1e-12 or np.std(y_valid) <= 1e-12:
            return 0.0

        rho_global, _ = spearmanr(llm_valid, y_valid)
        if rho_global is None or not np.isfinite(rho_global):
            rho_global = 0.0
        rho_extreme = self._extreme_weighted_rank_correlation(
            llm_valid,
            y_valid,
            gamma=self.extreme_rank_power,
        )

        lambda_eff = min(
            max(self.lam_llm * max(float(rho_extreme), 0.0), 0.0),
            self.lam_llm,
        )
        self._last_llm_rho = float(rho_global)
        self._last_llm_extreme_rho = float(rho_extreme)
        self._last_lambda_eff = float(lambda_eff)
        return float(lambda_eff)

    @staticmethod
    def _normalize_score(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        v_min, v_max = float(np.nanmin(v)), float(np.nanmax(v))
        if v_max > v_min:
            return (v - v_min) / (v_max - v_min)
        return np.zeros_like(v)

    def _maybe_dynamic_rescore(
        self,
        pool_scores: dict[str, np.ndarray] | None,
        mu: np.ndarray,
        sigma: np.ndarray,
        ucb_rank: np.ndarray,
        static_acq: np.ndarray,
        batch_size: int,
        rng: np.random.RandomState,
    ) -> dict[str, np.ndarray] | None:
        """Periodically rescore a diversified top-M subset under AL feedback."""
        self._last_dynamic_subset_local = np.array([], dtype=int)
        if (
            self._dynamic_scorer is None
            or self._rescoring_interval <= 0
            or self._al_iteration <= 0
            or self._al_iteration % self._rescoring_interval != 0
        ):
            return pool_scores

        if self._pool_indices is None or self._train_idx is None or self._y_all is None:
            return pool_scores

        subset_local = self._select_dynamic_rescore_subset(
            mu=mu,
            sigma=sigma,
            ucb_rank=ucb_rank,
            static_acq=static_acq,
            batch_size=batch_size,
            rng=rng,
        )
        if len(subset_local) == 0:
            return pool_scores
        self._last_dynamic_subset_local = np.asarray(subset_local, dtype=int).copy()
        subset_global = self._pool_indices[subset_local]

        progress_report = None
        feedback_fingerprint = f"iter-{self._al_iteration}"
        if hasattr(self._dynamic_scorer, "prepare_candidate_memories"):
            self._dynamic_scorer.prepare_candidate_memories(
                train_idx=self._train_idx,
                query_indices=subset_global,
                iteration=self._al_iteration,
            )
        if hasattr(self._dynamic_scorer, "build_feedback"):
            progress_report, feedback_fingerprint = self._dynamic_scorer.build_feedback(
                self._train_idx,
                self._y_all,
                current_scores=self._all_scores,
                score_overrides=self._score_overrides,
                query_indices=subset_global,
            )

        updated = {
            k: np.asarray(v, dtype=np.float64).copy()
            for k, v in pool_scores.items()
        }
        missing_positions: list[int] = []
        cached_positions: list[int] = []
        overrides = self._score_overrides.setdefault("s_phys", {})
        if "s_phys" in updated:
            for pos, global_idx in enumerate(subset_global):
                cache_key = (feedback_fingerprint, int(global_idx))
                if cache_key in self._dynamic_score_cache:
                    value = float(self._dynamic_score_cache[cache_key])
                    updated["s_phys"][subset_local[pos]] = value
                    overrides[int(global_idx)] = value
                    cached_positions.append(pos)
                else:
                    missing_positions.append(pos)
        else:
            missing_positions = list(range(len(subset_global)))

        if cached_positions and hasattr(self._dynamic_scorer, "record_cache_reuse"):
            cached_positions_arr = np.asarray(cached_positions, dtype=int)
            cached_global = subset_global[cached_positions_arr]
            cached_local = subset_local[cached_positions_arr]
            self._dynamic_scorer.record_cache_reuse(
                pool_indices=cached_global,
                scores=updated["s_phys"][cached_local],
                feedback_fingerprint=feedback_fingerprint,
                iteration=self._al_iteration,
                progress_report=progress_report or "",
            )

        if len(missing_positions) == 0:
            if hasattr(self._dynamic_scorer, "record_prequery_ledger"):
                self._dynamic_scorer.record_prequery_ledger(
                    pool_indices=subset_global,
                    scores=updated["s_phys"][subset_local],
                    feedback_fingerprint=feedback_fingerprint,
                    iteration=self._al_iteration,
                    progress_report=progress_report or "",
                )
            print(
                f"[{self.name}] Dynamic LLM rescoring reused "
                f"{len(subset_local)}/{len(self._pool_indices)} cached candidates "
                f"(feedback={feedback_fingerprint})",
                flush=True,
            )
            return updated

        missing_positions_arr = np.asarray(missing_positions, dtype=int)
        missing_local = subset_local[missing_positions_arr]
        missing_global = subset_global[missing_positions_arr]
        print(
            f"[{self.name}] Dynamic LLM rescoring at iteration {self._al_iteration} "
            f"for {len(missing_global)}/{len(subset_local)} uncached pool candidates "
            f"(feedback={feedback_fingerprint})",
            flush=True,
        )
        # Strict policy: every selected candidate must receive a successful
        # dynamic score. Transient API failures are retried inside LLMReviewer;
        # permanent request/configuration errors stop the run and preserve the
        # checkpoint. Never silently substitute the static score here.
        rescored = self._dynamic_scorer.rescore(
            pool_indices=missing_global,
            train_idx=self._train_idx,
            y_train=self._y_all,
            current_scores=self._all_scores,
            score_overrides=self._score_overrides,
            progress_report=progress_report,
            feedback_fingerprint=feedback_fingerprint,
            iteration=self._al_iteration,
        )
        for key in ("s_phys",):
            if key in updated and key in rescored:
                values = np.asarray(rescored[key], dtype=np.float64)
                if len(values) != len(missing_local) or not np.all(np.isfinite(values)):
                    raise RuntimeError(
                        f"Dynamic LLM rescoring returned invalid {key}: "
                        f"expected {len(missing_local)} finite values, got {values}"
                    )
                updated[key][missing_local] = values
                overrides = self._score_overrides.setdefault(key, {})
                for global_idx, value in zip(missing_global, values):
                    overrides[int(global_idx)] = float(value)
                    if key == "s_phys":
                        self._dynamic_score_cache[
                            (feedback_fingerprint, int(global_idx))
                        ] = float(value)
        if hasattr(self._dynamic_scorer, "record_prequery_ledger"):
            self._dynamic_scorer.record_prequery_ledger(
                pool_indices=subset_global,
                scores=updated["s_phys"][subset_local],
                feedback_fingerprint=feedback_fingerprint,
                iteration=self._al_iteration,
                progress_report=progress_report or "",
            )
        return updated

    def _select_dynamic_rescore_subset(
        self,
        *,
        mu: np.ndarray,
        sigma: np.ndarray,
        ucb_rank: np.ndarray,
        static_acq: np.ndarray,
        batch_size: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Build a multi-criterion subset so low initial LLM scores do not lock candidates out."""
        n_pool = len(static_acq)
        if n_pool == 0:
            return np.array([], dtype=int)

        min_m = max(batch_size, self._dynamic_top_m_min_mult * batch_size)
        frac_m = int(np.ceil(self._dynamic_top_m_frac * n_pool))
        target_m = min(n_pool, self._dynamic_top_m_cap, max(min_m, frac_m))

        quotas = {
            "ucb": int(np.ceil(0.40 * target_m)),
            "mu": int(np.ceil(0.25 * target_m)),
            "sigma": int(np.ceil(0.15 * target_m)),
            "static": int(np.ceil(0.10 * target_m)),
        }
        random_quota = max(0, target_m - sum(quotas.values()))
        random_quota = max(random_quota, int(np.ceil(self._dynamic_random_frac * target_m)))

        selected: list[int] = []
        selected_set: set[int] = set()

        def _add_top(values: np.ndarray, k: int) -> None:
            if k <= 0:
                return
            ranked = np.argsort(-np.asarray(values, dtype=np.float64))[: min(k, n_pool)]
            for idx in ranked:
                i = int(idx)
                if i not in selected_set:
                    selected.append(i)
                    selected_set.add(i)
                if len(selected) >= target_m:
                    return

        _add_top(ucb_rank, quotas["ucb"])
        _add_top(mu, quotas["mu"])
        _add_top(sigma, quotas["sigma"])
        _add_top(static_acq, quotas["static"])

        if len(selected) < target_m and random_quota > 0:
            remaining = np.array(
                [i for i in range(n_pool) if i not in selected_set],
                dtype=int,
            )
            if len(remaining) > 0:
                take = min(random_quota, target_m - len(selected), len(remaining))
                for idx in rng.choice(remaining, size=take, replace=False):
                    i = int(idx)
                    selected.append(i)
                    selected_set.add(i)

        if len(selected) < target_m:
            _add_top(ucb_rank, n_pool)

        return np.array(selected[:target_m], dtype=int)

    def _predict_fitted_surrogate(
        self,
        sname: str,
        pool_X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with one already-fitted surrogate."""
        if sname == "gbdt":
            if not self._gbdt_models or self._tree_imputer is None:
                raise RuntimeError("GBDT model not trained")
            X_imp = self._tree_imputer.transform(pool_X.values)
            predictions = np.asarray([
                model.predict(X_imp) for model in self._gbdt_models
            ])
            return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)

        elif sname == "rfr":
            if self._rfr_model is None:
                raise RuntimeError("RFR model not trained")
            preds = np.array([t.predict(pool_X.values) for t in self._rfr_model.estimators_])
            return preds.mean(axis=0), preds.std(axis=0, ddof=1)

        elif sname == "gpr":
            if self._gpr_model is None or self._gpr_scaler is None:
                raise RuntimeError("GPR model not trained")
            X_scaled = self._gpr_scaler.transform(pool_X.values)
            mu, sigma = self._gpr_model.predict(X_scaled, return_std=True)
            return mu.ravel(), sigma

        elif sname == "nn":
            return self._nn_predict(pool_X)

        raise RuntimeError(f"Unknown surrogate: {sname}")

    def _ensemble_predict(
        self,
        pool_X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine predictive distributions and include model disagreement."""
        predictions: list[tuple[str, float, np.ndarray, np.ndarray]] = []
        for name in self.SURROGATE_POOL:
            weight = float(self._surrogate_weights.get(name, 0.0))
            if weight <= 0:
                continue
            try:
                mu, sigma = self._predict_fitted_surrogate(name, pool_X)
            except RuntimeError:
                continue
            mu = np.asarray(mu, dtype=np.float64).ravel()
            sigma = np.asarray(sigma, dtype=np.float64).ravel()
            if len(mu) != len(pool_X) or len(sigma) != len(pool_X):
                raise RuntimeError(
                    f"{self.name}/{name}: invalid prediction lengths "
                    f"mu={len(mu)}, sigma={len(sigma)}, pool={len(pool_X)}"
                )
            if not np.all(np.isfinite(mu)):
                continue
            sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 0.0), 0.0)
            sigma *= float(self._surrogate_sigma_scales.get(name, 1.0))
            predictions.append((name, weight, mu, sigma))

        if not predictions:
            raise RuntimeError(f"{self.name}: no finite ensemble prediction")
        weights = np.asarray([item[1] for item in predictions], dtype=np.float64)
        weights /= float(weights.sum())
        mus = np.stack([item[2] for item in predictions], axis=0)
        sigmas = np.stack([item[3] for item in predictions], axis=0)
        mean = np.sum(weights[:, None] * mus, axis=0)
        within_variance = np.sum(weights[:, None] * np.square(sigmas), axis=0)
        disagreement_variance = np.sum(
            weights[:, None] * np.square(mus - mean[None, :]),
            axis=0,
        )
        variance = np.maximum(within_variance + disagreement_variance, 0.0)
        self._last_ensemble_disagreement_mean = float(
            np.mean(np.sqrt(np.maximum(disagreement_variance, 0.0)))
        )
        self._last_ensemble_within_sigma_mean = float(
            np.mean(np.sqrt(np.maximum(within_variance, 0.0)))
        )
        return mean, np.sqrt(variance)

    def _best_predict(self, pool_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict with a fixed surrogate (used by NN-UCB)."""
        if self._best_surrogate is None:
            raise RuntimeError(f"{self.name}: fixed surrogate is not configured")
        mu, sigma = self._predict_fitted_surrogate(self._best_surrogate, pool_X)
        sigma = np.asarray(sigma, dtype=np.float64) * float(
            self._surrogate_sigma_scales.get(self._best_surrogate, 1.0)
        )
        return mu, sigma

    def _nn_predict(self, pool_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """MC Dropout 预测."""
        if self._nn_model is None:
            raise RuntimeError("NN model not trained")

        if (
            self._nn_feature_cols is None
            or self._nn_imputer is None
            or self._nn_scaler is None
            or self._device is None
        ):
            raise RuntimeError("NN preprocessing state is unavailable")
        X_proc = self._nn_imputer.transform(
            pool_X[self._nn_feature_cols].values
        )
        X_proc = self._nn_scaler.transform(X_proc)
        X_tensor = torch.tensor(X_proc, dtype=torch.float32).to(self._device)

        self._enable_mc_dropout(self._nn_model)
        mc_preds = []
        with torch.no_grad():
            for _ in range(MC_SAMPLES):
                p = self._nn_model(X_tensor).cpu().numpy()
                mc_preds.append(p * self._y_std + self._y_mean)
        self._nn_model.eval()

        mc_preds = np.array(mc_preds)
        return mc_preds.mean(axis=0), mc_preds.std(axis=0, ddof=1)
