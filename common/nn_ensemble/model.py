"""Neural regressors used by CALM and legacy feature-group experiments.

The formal CALM protocol uses :class:`SimpleMLPRegressor`, a standardized
64-32-16 GELU MLP with MC Dropout.  The older group-gated architecture remains
available only for historical/offline compatibility and is not part of the
formal raw-feature benchmark.

三组特征 (base / llm / magpie) 各自 MLP 投影为 token, 通过 CLS token + Pre-Norm
多头 Cross-Group Attention 学习组间交互, Gated 残差 + Gate Pooling 融合。

分组语义:
    base    — 经验事实: 元素 wt% + 工艺参数 ("我加了什么")
    magpie  — 物理先验: 元素的原子本性 (周期表普适常数)
    llm     — 知识蒸馏: AI 冶金学家自主推理的体系自适应特征

训练期: GaussianNoise + Feature Dropout + MixUp
优化:   AdamW + Warmup + CosineAnnealing, SWA, Huber Loss

用法:
    python common/nn_ensemble/model.py --target Strength
    python common/nn_ensemble/model.py --target TC
"""

from __future__ import annotations

import argparse
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 数据 / 特征分组
# ---------------------------------------------------------------------------

TARGET_COLUMNS = ["Strength", "TC"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dabase"

# 元素 + 工艺类基础列(命中即视为 base 组)
BASE_HINTS = {
    "Cu", "Mg", "Mn", "Si", "Fe", "Cr", "Zn", "V", "Ti", "Zr", "Ni", "Bi", "Pb", "Sc", "Al",
    "SATemp", "AgeingTemp", "AgeingTime",
}
# ── 手工特征匹配关键字 (仅用于向后兼容, LLM 特征不会被误判) ──
HANDCRAFTED_HINTS = (
    "ratio", "sum", "PSI", "RII", "CDTI", "_x_", "Delta_T", "TempTime", "Log_",
    "solute", "purity", "transition", "impurity", "conductivity", "scattering",
    "entropy", "mismatch", "plus",
)


def split_feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    """将特征列拆分为三组: base / llm / magpie.

    - base:    元素列 + 工艺列 (如 Cu, Mg, AgeingTemp)
    - llm:     LLM_ 前缀的 AI 冶金学家设计的特征
    - magpie:  MagpieData 前缀的通用原子性质描述符
    """
    base = [c for c in feature_cols if c in BASE_HINTS]
    llm = [c for c in feature_cols if c.startswith("LLM_")]
    magpie = [c for c in feature_cols if c.startswith("MagpieData ")]
    used = set(base) | set(llm) | set(magpie)

    # 剩余特征: 手工特征 (向后兼容) 或未分类
    handcrafted = [
        c for c in feature_cols
        if c not in used and any(k in c for k in HANDCRAFTED_HINTS)
    ]
    used |= set(handcrafted)
    rest = [c for c in feature_cols if c not in used]
    handcrafted.extend(rest)

    return {
        "base": base,
        "llm": llm,
        "magpie": magpie,
        "handcrafted": handcrafted,  # 向后兼容, 通常为空
    }


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class GaussianNoise(nn.Module):
    """训练期对输入施加高斯噪声, 推断期为恒等映射。"""

    def __init__(self, sigma: float = 0.05):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x


class SimpleMLPRegressor(nn.Module):
    """Small standardized tabular MLP used by NN-UCB and Hybrid.

    Input imputation and standardization are fitted outside the module on the
    current labelled set. Dropout is deliberately kept in every hidden block
    so inference can obtain predictive dispersion through MC sampling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.20,
    ):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(value < 1 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: List[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.hidden_dims = hidden_dims
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GroupEncoder(nn.Module):
    """单组特征 -> emb_dim token 的 Pre-Norm 双层 MLP。"""

    def __init__(self, in_dim: int, emb_dim: int, dropout: float):
        super().__init__()
        hidden = max(emb_dim * 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossGroupBlock(nn.Module):
    """Pre-Norm Transformer Block: MHA + FFN, 残差缩放 + DropPath。"""

    def __init__(self, emb_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            emb_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        hidden = emb_dim * 2
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )
        self.drop = nn.Dropout(dropout)
        # 残差门控, 初始化偏向恒等映射
        self.gate_attn = nn.Parameter(torch.full((emb_dim,), 0.5))
        self.gate_ff = nn.Parameter(torch.full((emb_dim,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out) * torch.sigmoid(self.gate_attn)
        h = self.norm2(x)
        x = x + self.drop(self.ff(h)) * torch.sigmoid(self.gate_ff)
        return x


class GroupGatedFusionRegressor(nn.Module):
    """三组嵌入 + CLS Token + 跨组注意力 + 门控融合的小样本回归网络。

    分组语义:
      base   — 经验事实 (元素 wt% + 工艺参数)
      llm    — 知识蒸馏 (AI 冶金学家自主推理特征)
      magpie — 物理先验 (元素原子本性)
    """

    def __init__(
        self,
        group_dims: Dict[str, int],
        emb_dim: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.30,
        input_noise: float = 0.05,
        feature_dropout: float = 0.10,
    ):
        super().__init__()
        # llm 组优先级高于 handcrafted (向后兼容),
        # 空组自动跳过 (group_dims[g] > 0 条件).
        self.group_order = [
            g for g in ["base", "llm", "magpie", "handcrafted"]
            if group_dims.get(g, 0) > 0
        ]
        if not self.group_order:
            raise ValueError("group_dims 中没有任何非空分组。")

        self.input_noise = GaussianNoise(input_noise)
        self.feature_dropout = nn.Dropout(feature_dropout)

        self.encoders = nn.ModuleDict({
            g: GroupEncoder(group_dims[g], emb_dim, dropout) for g in self.group_order
        })
        # 每组一个 learnable position/type embedding
        self.group_pos = nn.Parameter(torch.zeros(len(self.group_order), emb_dim))
        nn.init.trunc_normal_(self.group_pos, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            CrossGroupBlock(emb_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        # 门控池化: 学习每个 token 的权重
        self.pool_gate = nn.Linear(emb_dim, 1)

        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x_groups: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for i, g in enumerate(self.group_order):
            x = x_groups[g]
            x = self.input_noise(x)
            x = self.feature_dropout(x)
            tok = self.encoders[g](x) + self.group_pos[i]
            tokens.append(tok.unsqueeze(1))
        cls = self.cls_token.expand(tokens[0].size(0), -1, -1)
        h = torch.cat([cls] + tokens, dim=1)  # [B, 1+G, D]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)

        # CLS + 门控池化的组合表示
        gate = torch.softmax(self.pool_gate(h).squeeze(-1), dim=1)  # [B, 1+G]
        pooled = (h * gate.unsqueeze(-1)).sum(dim=1)
        feat = 0.5 * h[:, 0] + 0.5 * pooled
        return self.head(feat).squeeze(-1)


# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    target: str
    n_splits: int = 5
    seed: int = 42
    epochs: int = 400
    batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-3
    mlp_hidden_dims: Tuple[int, ...] = (64, 32, 16)
    emb_dim: int = 48
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.30
    input_noise: float = 0.06
    feature_dropout: float = 0.10
    warmup_epochs: int = 20
    early_stop_patience: int = 80
    grad_clip_norm: float = 1.0
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5
    swa_start_frac: float = 0.6
    swa_lr_ratio: float = 0.25  # SWA 阶段 lr = lr * swa_lr_ratio
    huber_delta: float = 1.0


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor_dict(
    arr_dict: Dict[str, np.ndarray], idx: np.ndarray, device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        k: torch.tensor(v[idx], dtype=torch.float32, device=device)
        for k, v in arr_dict.items()
        if v.shape[1] > 0
    }


def create_group_arrays(df: pd.DataFrame, groups: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for g, cols in groups.items():
        if not cols:
            arrays[g] = np.zeros((len(df), 0), dtype=np.float32)
            continue
        x = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
        x = x.replace([np.inf, -np.inf], np.nan)
        arrays[g] = x.to_numpy(dtype=np.float32)
    return arrays


def fold_preprocess(
    train_arr: np.ndarray, val_arr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """每折独立的 median imputer + StandardScaler, 避免数据泄漏。"""
    if train_arr.shape[1] == 0:
        return train_arr, val_arr
    imputer = SimpleImputer(strategy="median")
    train_imp = imputer.fit_transform(train_arr)
    val_imp = imputer.transform(val_arr)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imp).astype(np.float32)
    val_scaled = scaler.transform(val_imp).astype(np.float32)
    return train_scaled, val_scaled


def cosine_warmup_lr(epoch: int, total_epochs: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def mixup_groups(
    x_groups: Dict[str, torch.Tensor],
    y: torch.Tensor,
    alpha: float,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """对所有分组用同一组 (lam, perm) 做 MixUp。"""
    if alpha <= 0:
        return x_groups, y
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # 偏向更接近主样本, 稳定一些
    batch_size = y.size(0)
    perm = torch.randperm(batch_size, device=y.device)
    mixed_groups = {k: lam * v + (1.0 - lam) * v[perm] for k, v in x_groups.items()}
    mixed_y = lam * y + (1.0 - lam) * y[perm]
    return mixed_groups, mixed_y


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def train_one_fold(
    model: nn.Module,
    x_train: Dict[str, np.ndarray],
    x_val: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.HuberLoss(delta=cfg.huber_delta)

    train_size = len(y_train)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm_np = (y_val - y_mean) / y_std

    swa_start_epoch = int(cfg.epochs * cfg.swa_start_frac)
    swa_state: Optional[Dict[str, torch.Tensor]] = None
    swa_count = 0

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience = 0
    min_lr = max(cfg.lr * 0.01, 1e-5)
    swa_lr = cfg.lr * cfg.swa_lr_ratio

    for epoch in range(cfg.epochs):
        if epoch < swa_start_epoch:
            lr = cosine_warmup_lr(epoch, swa_start_epoch, cfg.warmup_epochs, cfg.lr, min_lr)
        else:
            # SWA 阶段使用相对低且固定的 lr, 便于权重平均
            lr = swa_lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        indices = np.random.permutation(train_size)
        for start in range(0, train_size, cfg.batch_size):
            batch_idx = indices[start:start + cfg.batch_size]
            if len(batch_idx) < 2:
                continue
            xb = to_tensor_dict(x_train, batch_idx, device)
            yb = torch.tensor(y_train_norm[batch_idx], dtype=torch.float32, device=device)

            if cfg.mixup_alpha > 0 and random.random() < cfg.mixup_prob:
                xb, yb = mixup_groups(xb, yb, cfg.mixup_alpha)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()

        # 进入 SWA 阶段后, 每个 epoch 收一次权重均值
        if epoch >= swa_start_epoch:
            with torch.no_grad():
                cur_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if swa_state is None:
                    swa_state = cur_state
                    swa_count = 1
                else:
                    swa_count += 1
                    for k, v in cur_state.items():
                        if v.dtype.is_floating_point:
                            swa_state[k] = swa_state[k] + (v - swa_state[k]) / swa_count
                        else:
                            swa_state[k] = v

        # 验证
        model.eval()
        with torch.no_grad():
            x_val_t = to_tensor_dict(x_val, np.arange(len(y_val)), device)
            y_val_t = torch.tensor(y_val_norm_np, dtype=torch.float32, device=device)
            val_pred = model(x_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience and epoch >= swa_start_epoch:
                break

    # 评估 SWA 与 best_val 两个权重, 取验证集更优者
    def _eval_state(state: Dict[str, torch.Tensor]) -> Tuple[float, np.ndarray]:
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            x_val_t = to_tensor_dict(x_val, np.arange(len(y_val)), device)
            pred_norm = model(x_val_t).detach().cpu().numpy()
        mae = float(np.mean(np.abs(pred_norm - y_val_norm_np)))
        return mae, pred_norm * y_std + y_mean

    candidates = {}
    if best_state is not None:
        candidates["best"] = _eval_state(best_state)
    if swa_state is not None:
        # swa_state 在 device 上, eval 时 load_state_dict 没问题
        candidates["swa"] = _eval_state(swa_state)

    if not candidates:
        # 退化情况, 直接用当前 model
        model.eval()
        with torch.no_grad():
            x_val_t = to_tensor_dict(x_val, np.arange(len(y_val)), device)
            pred_norm = model(x_val_t).detach().cpu().numpy()
        return pred_norm * y_std + y_mean

    tag = min(candidates, key=lambda k: candidates[k][0])
    return candidates[tag][1]


# ---------------------------------------------------------------------------
# OOF 训练入口
# ---------------------------------------------------------------------------

def run_oof_training(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_all = pd.to_numeric(df[cfg.target], errors="coerce")
    valid_mask = y_all.notna().to_numpy()
    work_df = df.loc[valid_mask].reset_index(drop=True)
    y = pd.to_numeric(work_df[cfg.target], errors="coerce").to_numpy(dtype=np.float32)

    feature_cols = [c for c in work_df.columns if c not in TARGET_COLUMNS and c != "Alloy"]
    groups = split_feature_groups(feature_cols)
    group_arrays = create_group_arrays(work_df, groups)
    group_dims = {k: v.shape[1] for k, v in group_arrays.items()}
    print(f"[INFO] 样本数: {len(work_df)}, 目标: {cfg.target}")
    print(f"[INFO] 分组维度: {group_dims}")
    for g, cols in groups.items():
        print(f"       - {g} ({len(cols)}): {cols}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    seed = cfg.seed
    set_global_seed(seed)
    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(work_df), dtype=np.float32)
    print(f"[INFO] seed = {seed}")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(work_df), start=1):
        x_train_fold, x_val_fold = {}, {}
        for g in group_arrays:
            tr_scaled, va_scaled = fold_preprocess(
                group_arrays[g][tr_idx], group_arrays[g][va_idx]
            )
            x_train_fold[g] = tr_scaled
            x_val_fold[g] = va_scaled

        model = GroupGatedFusionRegressor(
            group_dims=group_dims,
            emb_dim=cfg.emb_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            input_noise=cfg.input_noise,
            feature_dropout=cfg.feature_dropout,
        ).to(device)

        va_pred = train_one_fold(
            model, x_train_fold, x_val_fold, y[tr_idx], y[va_idx], cfg, device
        )
        oof_pred[va_idx] = va_pred
        fold_mae = mean_absolute_error(y[va_idx], va_pred)
        fold_r2 = r2_score(y[va_idx], va_pred)
        print(f"[FOLD {fold}] MAE={fold_mae:.4f}  R2={fold_r2:.4f}")

    oof_mae = mean_absolute_error(y, oof_pred)
    oof_rmse = math.sqrt(mean_squared_error(y, oof_pred))
    oof_r2 = r2_score(y, oof_pred)
    print(f"[OOF] MAE={oof_mae:.4f}  RMSE={oof_rmse:.4f}  R2={oof_r2:.4f}")

    overall = {
        "mae": float(oof_mae),
        "rmse": float(oof_rmse),
        "r2": float(oof_r2),
        "n_samples": int(len(work_df)),
        "seed": int(seed),
        "n_splits": int(cfg.n_splits),
    }
    print(
        "[OOF-SUMMARY] "
        f"MAE={overall['mae']:.4f}  RMSE={overall['rmse']:.4f}  R2={overall['r2']:.4f}"
    )

    result = work_df[[cfg.target]].copy()
    result[f"{cfg.target}_oof_pred_nn"] = oof_pred
    return result, overall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_csv_for(target: str) -> Path:
    return DEFAULT_DATA_DIR / f"2xxx_mn_features_selected_{target.lower()}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Strength", choices=TARGET_COLUMNS)
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="特征CSV路径, 默认按 target 自动选择 2xxx_mn_features_selected_{target}.csv",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument("--emb_dim", type=int, default=48)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path) if args.csv_path else default_csv_for(args.target)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到特征CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[LOAD] {csv_path}  shape={df.shape}")

    cfg = TrainConfig(
        target=args.target,
        n_splits=args.n_splits,
        epochs=args.epochs,
        seed=args.seed,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        mixup_alpha=args.mixup_alpha,
    )
    out, metrics = run_oof_training(df, cfg)

    out_path = DEFAULT_DATA_DIR / f"oof_nn_{args.target.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(f"[SAVE] OOF结果 -> {out_path}")

    metrics_path = DEFAULT_DATA_DIR / f"oof_nn_{args.target.lower()}_metrics.json"
    import json
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] 指标JSON -> {metrics_path}")


if __name__ == "__main__":
    main()
