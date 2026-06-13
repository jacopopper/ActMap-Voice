from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ARCH_CHOICES = ("vit", "vit2d", "resnet", "convnext")
LABEL_TO_ID = {
    "ANSWER": 0,
    "VERIFY": 1,
    "ESCALATE": 2,
}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
SPATIAL_H = 32
SPATIAL_W = 128
EXPECTED_SHAPE = (12, SPATIAL_H, SPATIAL_W)


class ProbeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        augment: bool = False,
        noise_std: float = 0.03,
    ):
        self.rows = rows
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        heatmap = row["actmap"].float()
        if tuple(heatmap.shape) != EXPECTED_SHAPE:
            raise ValueError(f"bad actmap shape for row id={row.get('id')}: {tuple(heatmap.shape)}")
        label = torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long)
        if self.augment and self.noise_std > 0:
            heatmap = heatmap + torch.randn_like(heatmap) * self.noise_std
        return heatmap, label


def make_splits(
    rows: list[dict[str, Any]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    if not 0.0 < val_frac < 1.0 or not 0.0 < test_frac < 1.0:
        raise ValueError("val_frac and test_frac must be between 0 and 1.")
    if val_frac + test_frac >= 0.8:
        raise ValueError("val_frac + test_frac is too large.")

    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {label: [] for label in LABEL_TO_ID}
    for index, row in enumerate(rows):
        by_label[row["label"]].append(index)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label, indices in by_label.items():
        if len(indices) < 3:
            raise ValueError(f"need at least 3 rows for label {label}")
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_frac))
        n_val = max(1, round(len(indices) * val_frac))
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test : n_test + n_val])
        train_idx.extend(indices[n_test + n_val :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def save_split_indices(
    output_dir: str | Path,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> None:
    split_path = Path(output_dir) / "split_indices.json"
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, handle, indent=2)


def load_split_indices(path: str | Path) -> tuple[list[int], list[int], list[int]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload["train"]), list(payload["val"]), list(payload["test"])


class _PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_h: int, patch_w: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * mask


class _ViTBlockWithDropPath(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attn_drop: float,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = _DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = _DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.drop_path1(self.attn(x_norm, x_norm, x_norm, need_weights=False)[0])
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ActMapViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 3,
        patch_h: int = 4,
        patch_w: int = 16,
        embed_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 6,
        mlp_ratio: float = 3.0,
        dropout: float = 0.3,
        attn_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        spatial_h: int = SPATIAL_H,
        spatial_w: int = SPATIAL_W,
    ):
        super().__init__()
        assert spatial_h % patch_h == 0 and spatial_w % patch_w == 0
        num_patches = (spatial_h // patch_h) * (spatial_w // patch_w)

        self.patch_embed = _PatchEmbed(in_channels, patch_h, patch_w, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout * 0.5)

        mlp_dim = int(embed_dim * mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [
                _ViTBlockWithDropPath(embed_dim, num_heads, mlp_dim, dropout, attn_drop, dpr[i])
                for i in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(
            self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.size(0), -1)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class ActMapViT2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 3,
        patch_h: int = 4,
        patch_w: int = 16,
        embed_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 6,
        mlp_ratio: float = 3.0,
        dropout: float = 0.3,
        attn_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        spatial_h: int = SPATIAL_H,
        spatial_w: int = SPATIAL_W,
    ):
        super().__init__()
        assert spatial_h % patch_h == 0 and spatial_w % patch_w == 0
        self.grid_h = spatial_h // patch_h
        self.grid_w = spatial_w // patch_w

        self.patch_embed = _PatchEmbed(in_channels, patch_h, patch_w, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.row_pos = nn.Parameter(torch.zeros(1, self.grid_h, 1, embed_dim))
        self.col_pos = nn.Parameter(torch.zeros(1, 1, self.grid_w, embed_dim))
        self.pos_drop = nn.Dropout(dropout * 0.5)

        mlp_dim = int(embed_dim * mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [
                _ViTBlockWithDropPath(embed_dim, num_heads, mlp_dim, dropout, attn_drop, dpr[i])
                for i in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(
            self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.size(0), -1)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        nn.init.trunc_normal_(self.row_pos, std=0.02)
        nn.init.trunc_normal_(self.col_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.patch_embed(x)
        pos = (self.row_pos + self.col_pos).reshape(1, self.grid_h * self.grid_w, -1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls + self.cls_pos, x + pos], dim=1)
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.shortcut(x))


class ActMapResNet(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            _ResidualBlock(32, 32),
            _ResidualBlock(32, 64, stride=2),
            _ResidualBlock(64, 64),
            _ResidualBlock(64, 128, stride=2),
            _ResidualBlock(128, 128),
            _ResidualBlock(128, 192, stride=2),
            _ResidualBlock(192, 192),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


class _ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.pwconv = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        return residual + self.gamma * x


class ActMapConvNeXt(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.stages = nn.Sequential(
            _ConvNeXtBlock(48, dropout * 0.5),
            _ConvNeXtBlock(48, dropout * 0.5),
            nn.Conv2d(48, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.GELU(),
            _ConvNeXtBlock(96, dropout * 0.5),
            _ConvNeXtBlock(96, dropout * 0.5),
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.GELU(),
            _ConvNeXtBlock(192, dropout * 0.5),
            _ConvNeXtBlock(192, dropout * 0.5),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


def build_model(
    arch: str,
    *,
    in_channels: int,
    num_classes: int = 3,
    patch_h: int = 4,
    patch_w: int = 16,
    embed_dim: int = 192,
    num_heads: int = 6,
    num_layers: int = 6,
    mlp_ratio: float = 3.0,
    dropout: float = 0.3,
    attn_drop: float = 0.1,
    drop_path_rate: float = 0.05,
    spatial_h: int = SPATIAL_H,
    spatial_w: int = SPATIAL_W,
) -> nn.Module:
    if arch == "vit":
        return ActMapViT(
            in_channels=in_channels,
            num_classes=num_classes,
            patch_h=patch_h,
            patch_w=patch_w,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )
    if arch == "vit2d":
        return ActMapViT2D(
            in_channels=in_channels,
            num_classes=num_classes,
            patch_h=patch_h,
            patch_w=patch_w,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
        )
    if arch == "resnet":
        return ActMapResNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    if arch == "convnext":
        return ActMapConvNeXt(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown architecture: {arch}")


def make_model_config(
    arch: str,
    *,
    in_channels: int,
    num_classes: int,
    patch_h: int,
    patch_w: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    mlp_ratio: float,
    dropout: float,
    attn_drop: float,
    drop_path_rate: float,
    spatial_h: int = SPATIAL_H,
    spatial_w: int = SPATIAL_W,
) -> dict[str, Any]:
    return {
        "arch": arch,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_ratio": mlp_ratio,
        "dropout": dropout,
        "attn_drop": attn_drop,
        "drop_path_rate": drop_path_rate,
        "spatial_h": spatial_h,
        "spatial_w": spatial_w,
    }


def model_label(model_config: dict[str, Any]) -> str:
    arch = model_config["arch"]
    if arch in {"vit", "vit2d"}:
        return (
            f"{arch} {model_config['in_channels']}ch "
            f"{model_config['spatial_h'] // model_config['patch_h']}x"
            f"{model_config['spatial_w'] // model_config['patch_w']} patches "
            f"{model_config['num_layers']}Lx{model_config['num_heads']}Hx"
            f"{model_config['embed_dim']}D"
        )
    return f"{arch} {model_config['in_channels']}ch"


def load_rows(data_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"Expected {data_path} to contain a list, got {type(rows).__name__}")
    for index, row in enumerate(rows):
        if row.get("label") not in LABEL_TO_ID:
            raise ValueError(f"row {index} has unknown label {row.get('label')!r}")
        actmap = row.get("actmap")
        if actmap is None or tuple(actmap.shape) != EXPECTED_SHAPE:
            raise ValueError(f"row {index} has bad actmap shape")
        if not torch.isfinite(actmap.float()).all().item():
            raise ValueError(f"row {index} has non-finite actmap values")
    return rows


def save_checkpoint(path: Path, model: nn.Module, model_config: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "expected_shape": EXPECTED_SHAPE,
        },
        path,
    )


def load_checkpoint_payload(path: str | Path, device: torch.device) -> Any:
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def checkpoint_model_config(payload: Any, checkpoint_path: str | Path) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_config" in payload:
        return payload["model_config"]
    config_path = Path(checkpoint_path).with_name("model_config.json")
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        return saved["model_config"]
    raise ValueError(f"No model config found in {checkpoint_path}")


def select_device(require_cuda: bool = False, requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        beta = torch.distributions.Beta(alpha, alpha)
        lam = float(beta.sample().item())
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def _confusion(labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros(len(LABEL_TO_ID), len(LABEL_TO_ID), dtype=torch.long)
    for truth, pred in zip(labels.cpu(), preds.cpu()):
        matrix[int(truth), int(pred)] += 1
    return matrix


def _metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, Any]:
    preds = logits.argmax(dim=1)
    confusion = _confusion(labels, preds)
    accuracy = (preds == labels).float().mean().item()
    per_class = {}
    f1s = []
    for label_id, label_name in ID_TO_LABEL.items():
        tp = confusion[label_id, label_id].item()
        fp = confusion[:, label_id].sum().item() - tp
        fn = confusion[label_id, :].sum().item() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "confusion": confusion.tolist(),
        "per_class": per_class,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    mixup_alpha: float = 0.0,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.train(train)
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.set_grad_enabled(train):
        for heatmaps, labels_batch in loader:
            heatmaps = heatmaps.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            labels_orig = labels_batch.clone()

            if train and mixup_alpha > 0:
                heatmaps, labels_a, labels_b, lam = mixup_batch(heatmaps, labels_batch, mixup_alpha)
                logits = model(heatmaps)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(heatmaps)
                loss = criterion(logits, labels_batch)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels_batch)
            all_logits.append(logits.detach().cpu().float())
            all_labels.append(labels_orig.detach().cpu())

    labels = torch.cat(all_labels)
    logits = torch.cat(all_logits)
    return total_loss / max(len(labels), 1), logits, labels


def _cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.01 + 0.5 * (1.0 - 0.01) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _class_weights(rows: list[dict[str, Any]], indices: list[int]) -> torch.Tensor:
    counts = torch.zeros(len(LABEL_TO_ID), dtype=torch.float32)
    for index in indices:
        counts[LABEL_TO_ID[rows[index]["label"]]] += 1
    total = counts.sum()
    return total / (len(LABEL_TO_ID) * counts.clamp_min(1.0))


def train(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    arch: str = "vit2d",
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    dropout: float = 0.3,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    patience: int = 10,
    noise_std: float = 0.03,
    warmup_epochs: int = 4,
    mixup_alpha: float = 0.1,
    patch_h: int = 4,
    patch_w: int = 8,
    embed_dim: int = 192,
    num_heads: int = 6,
    num_layers: int = 6,
    mlp_ratio: float = 3.0,
    attn_drop: float = 0.1,
    drop_path_rate: float = 0.05,
    split_file: str | Path | None = None,
    require_cuda: bool = False,
    device_name: str = "auto",
    num_workers: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_path)
    in_channels = rows[0]["actmap"].shape[0]
    label_counts = {label: sum(row["label"] == label for row in rows) for label in LABEL_TO_ID}
    print(f"Loaded {len(rows)} rows | in_channels={in_channels} | labels={label_counts}")

    if split_file is None:
        train_idx, val_idx, test_idx = make_splits(rows, val_frac, test_frac, seed)
    else:
        train_idx, val_idx, test_idx = load_split_indices(split_file)
        print(f"Using split file: {split_file}")
    save_split_indices(out, train_idx, val_idx, test_idx)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        counts = {label: sum(rows[i]["label"] == label for i in idx) for label in LABEL_TO_ID}
        print(f"  {name}: {counts}")

    train_ds = ProbeDataset([rows[i] for i in train_idx], augment=True, noise_std=noise_std)
    val_ds = ProbeDataset([rows[i] for i in val_idx], augment=False)
    test_ds = ProbeDataset([rows[i] for i in test_idx], augment=False)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = select_device(require_cuda=require_cuda, requested=device_name)
    print(f"Device: {device}")

    model_config = make_model_config(
        arch,
        in_channels=in_channels,
        num_classes=len(LABEL_TO_ID),
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_drop=attn_drop,
        drop_path_rate=drop_path_rate,
    )
    model = build_model(
        arch,
        in_channels=in_channels,
        num_classes=len(LABEL_TO_ID),
        patch_h=patch_h,
        patch_w=patch_w,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_drop=attn_drop,
        drop_path_rate=drop_path_rate,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_label(model_config)} | params={n_params:,}")
    with (out / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"model_config": model_config}, handle, indent=2)

    class_weights = _class_weights(rows, train_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _cosine_schedule(optimizer, warmup_epochs, epochs)

    best_val_acc = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    best_path = out / "best_model.pt"

    header = f"{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>9} {'ValAcc':>8} {'ValF1':>8}"
    print(f"\n{header}\n{'-' * len(header)}")

    for epoch in range(1, epochs + 1):
        started_at = time.time()
        train_loss, train_logits, train_labels = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            mixup_alpha=mixup_alpha,
        )
        val_loss, val_logits, val_labels = _run_epoch(
            model,
            val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            train=False,
        )
        scheduler.step()

        train_metrics = _metrics(train_labels, train_logits)
        val_metrics = _metrics(val_labels, val_logits)
        elapsed = time.time() - started_at
        print(
            f"{epoch:>5} {train_loss:>10.4f} {val_loss:>9.4f} "
            f"{val_metrics['accuracy']:>8.4f} {val_metrics['macro_f1']:>8.4f} [{elapsed:.1f}s]"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(best_path, model, model_config)

        if epoch - best_epoch >= patience:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(best val accuracy {best_val_acc:.4f} at epoch {best_epoch})"
            )
            break

    with (out / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    payload = load_checkpoint_payload(best_path, device)
    model.load_state_dict(checkpoint_state_dict(payload))

    test_loss, test_logits, test_labels = _run_epoch(
        model,
        test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
    )
    test_metrics = _metrics(test_labels, test_logits)
    results = {
        "arch": arch,
        "model_config": model_config,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_confusion": test_metrics["confusion"],
        "test_per_class": test_metrics["per_class"],
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "label_counts": label_counts,
        "hyperparams": {
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "batch_size": batch_size,
            "noise_std": noise_std,
            "warmup_epochs": warmup_epochs,
            "mixup_alpha": mixup_alpha,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_ratio": mlp_ratio,
            "attn_drop": attn_drop,
            "drop_path_rate": drop_path_rate,
            "seed": seed,
        },
    }

    with (out / "test_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\n=== TEST RESULTS (n={len(test_labels)}) ===")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Confusion rows=true cols=pred: {test_metrics['confusion']}")
    print(f"\nSaved checkpoint -> {best_path}")
    print(f"Saved results    -> {out / 'test_results.json'}")
    return results


def _cmd_train(args: argparse.Namespace) -> None:
    train(
        data_path=args.data,
        output_dir=args.output,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        patience=args.patience,
        noise_std=args.noise_std,
        warmup_epochs=args.warmup_epochs,
        mixup_alpha=args.mixup_alpha,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        attn_drop=args.attn_drop,
        drop_path_rate=args.drop_path_rate,
        split_file=args.split_file,
        require_cuda=args.require_cuda,
        device_name=args.device,
        num_workers=args.num_workers,
    )


def _cmd_sweep(args: argparse.Namespace) -> None:
    arches = args.arches.split(",")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    results = []
    for arch in arches:
        for seed in seeds:
            output = Path(args.output) / f"{arch}_seed{seed}"
            print(f"\n### sweep arch={arch} seed={seed} output={output}")
            result = train(
                data_path=args.data,
                output_dir=output,
                arch=arch,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                seed=seed,
                patience=args.patience,
                noise_std=args.noise_std,
                warmup_epochs=args.warmup_epochs,
                mixup_alpha=args.mixup_alpha,
                patch_h=args.patch_h,
                patch_w=args.patch_w,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                mlp_ratio=args.mlp_ratio,
                attn_drop=args.attn_drop,
                drop_path_rate=args.drop_path_rate,
                split_file=args.split_file,
                require_cuda=args.require_cuda,
                device_name=args.device,
                num_workers=args.num_workers,
            )
            results.append(
                {
                    "arch": arch,
                    "seed": seed,
                    "output": str(output),
                    "test_accuracy": result["test_accuracy"],
                    "test_macro_f1": result["test_macro_f1"],
                    "best_val_accuracy": result["best_val_accuracy"],
                }
            )

    results = sorted(results, key=lambda row: row["test_accuracy"], reverse=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with (Path(args.output) / "sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("\n=== SWEEP SUMMARY ===")
    for row in results:
        print(
            f"{row['arch']} seed={row['seed']} "
            f"test_acc={row['test_accuracy']:.4f} test_f1={row['test_macro_f1']:.4f} "
            f"output={row['output']}"
        )


def _cmd_test(args: argparse.Namespace) -> None:
    device = select_device(require_cuda=args.require_cuda, requested=args.device)
    rows = load_rows(args.data)

    if args.split_file is not None:
        train_idx, val_idx, test_idx = load_split_indices(args.split_file)
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        rows = [rows[i] for i in split_map[args.split]]
        print(f"Using split '{args.split}' from {args.split_file}: {len(rows)} rows")

    payload = load_checkpoint_payload(args.checkpoint, device)
    model_config = checkpoint_model_config(payload, args.checkpoint)
    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint_state_dict(payload))
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Model: {model_label(model_config)}")

    loader = DataLoader(
        ProbeDataset(rows, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    criterion = nn.CrossEntropyLoss()
    loss, logits, labels = _run_epoch(
        model,
        loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
    )
    metrics = _metrics(labels, logits)
    print(f"\n=== INFERENCE RESULTS (n={len(labels)}) ===")
    print(f"  Loss:     {loss:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Confusion rows=true cols=pred: {metrics['confusion']}")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default="data/actmap_voice_qwen3_full_48tok.pt")
    parser.add_argument("--output", default="runs/actmap_vit2d")
    parser.add_argument("--arch", choices=ARCH_CHOICES, default="vit2d")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--mixup-alpha", type=float, default=0.1)
    parser.add_argument("--patch-h", type=int, default=4)
    parser.add_argument("--patch-w", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=3.0)
    parser.add_argument("--attn-drop", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.05)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT/CV router for ActMap Voice classification")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    add_train_args(p_train)

    p_sweep = sub.add_parser("sweep")
    add_train_args(p_sweep)
    p_sweep.add_argument("--arches", default="vit2d,resnet,convnext")
    p_sweep.add_argument("--seeds", default="42")

    p_test = sub.add_parser("test")
    p_test.add_argument("--data", default="data/actmap_voice_qwen3_full_48tok.pt")
    p_test.add_argument("--checkpoint", required=True)
    p_test.add_argument("--batch-size", type=int, default=128)
    p_test.add_argument("--split-file", default=None)
    p_test.add_argument("--split", choices=["train", "val", "test"], default="test")
    p_test.add_argument("--require-cuda", action="store_true")
    p_test.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p_test.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "sweep":
        _cmd_sweep(args)
    elif args.command == "test":
        _cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
