#!/usr/bin/env python3
"""
src/training/train.py

Training script for FusionClassifier (image + text).

Outputs in --save-dir:
- best.pt
- metrics.csv
- loss_curve.png
- accuracy_curve.png

Optionally runs evaluation at the end by calling src/evaluation/evaluate.py logic:
  python src/training/train.py --eval
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------

def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item())


# -------------------------
# Dynamic import for data-ingestion (dash folder)
# -------------------------

def import_multimodal_ingestion(src_dir: Path):
    module_path = src_dir / "data-ingestion" / "multimodal_ingestion.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Expected file not found: {module_path}")

    spec = importlib.util.spec_from_file_location("multimodal_ingestion", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------------------------
# Config
# -------------------------

@dataclass
class TrainConfig:
    train_path: str
    val_path: str
    test_path: str
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    use_scheduler: bool
    gamma: float
    save_dir: str
    seed: int

    text_model_name: str
    image_encoder_name: str
    text_encoder_name: str
    dropout: float
    max_length: int
    local_files_only: bool

    freeze_encoders: bool
    run_eval: bool


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()

    p.add_argument("--train-path", default="/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Train")
    p.add_argument("--val-path", default="/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Val")
    p.add_argument("--test-path", default="/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_Test")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    p.add_argument("--use-scheduler", action="store_true")
    p.add_argument("--gamma", type=float, default=0.95)

    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--text-model-name", default="distilbert-base-uncased")
    p.add_argument("--image-encoder-name", default="convnext_tiny")
    p.add_argument("--text-encoder-name", default="distilbert")
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--local-files-only", action="store_true")

    p.add_argument("--freeze-encoders", action="store_true")
    p.add_argument("--eval", action="store_true", help="Run evaluation pack after training")

    a = p.parse_args()
    return TrainConfig(
        train_path=a.train_path,
        val_path=a.val_path,
        test_path=a.test_path,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        use_scheduler=a.use_scheduler,
        gamma=a.gamma,
        save_dir=a.save_dir,
        seed=a.seed,
        text_model_name=a.text_model_name,
        image_encoder_name=a.image_encoder_name,
        text_encoder_name=a.text_encoder_name,
        dropout=a.dropout,
        max_length=a.max_length,
        local_files_only=a.local_files_only,
        freeze_encoders=a.freeze_encoders,
        run_eval=a.eval,
    )


# -------------------------
# Train/Eval loops (loss+acc only)
# -------------------------

def train_one_epoch(model, loader, device, criterion, optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += accuracy_from_logits(logits, labels)
        n += 1

    return total_loss / max(n, 1), total_acc / max(n, 1)


@torch.no_grad()
def evaluate_simple(model, loader, device, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += float(loss.item())
        total_acc += accuracy_from_logits(logits, labels)
        n += 1

    return total_loss / max(n, 1), total_acc / max(n, 1)


def save_checkpoint(path: Path, model: nn.Module, cfg: TrainConfig, class_to_idx: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "class_to_idx": class_to_idx,
        },
        path,
    )


def save_metrics_csv(path: Path, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_curves(save_dir: Path, history: List[Dict[str, float]]) -> None:
    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_loss"]) for r in history]
    val_loss = [float(r["val_loss"]) for r in history]
    train_acc = [float(r["train_acc"]) for r in history]
    val_acc = [float(r["val_acc"]) for r in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_acc, label="Train acc")
    plt.plot(epochs, val_acc, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_curve.png", dpi=200)
    plt.close()


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = get_device()
    print(f"Device: {device}")

    # Paths
    src_dir = Path(__file__).resolve().parents[1]   # .../src
    repo_root = src_dir.parent                     # repo root

    # Make src importable for models.*
    if str(src_dir) not in os.sys.path:
        os.sys.path.insert(0, str(src_dir))

    # Imports
    from models.classifiers import FusionClassifier  # noqa: E402
    ingestion_mod = import_multimodal_ingestion(src_dir)

    train_loader, val_loader, test_loader, class_to_idx = ingestion_mod.make_loaders(
        train_path=cfg.train_path,
        val_path=cfg.val_path,
        test_path=cfg.test_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        tokenizer_name=cfg.text_model_name,
        max_length=cfg.max_length,
        local_files_only=cfg.local_files_only,
    )

    # Build model
    model = FusionClassifier(
        text_model_name=cfg.text_model_name,
        num_classes=len(class_to_idx),
        dropout=cfg.dropout,
        image_encoder_name=cfg.image_encoder_name,
        text_encoder_name=cfg.text_encoder_name,
    ).to(device)

    if cfg.freeze_encoders:
        model.set_encoder_trainable(False)
        for p in model.head.parameters():
            p.requires_grad = True
        print("Encoders frozen: training fusion head only.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=cfg.gamma) if cfg.use_scheduler else None

    save_dir = repo_root / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    best_val_acc = -1.0

    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate_simple(model, val_loader, device, criterion)

        if scheduler is not None:
            scheduler.step()

        history.append({
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_path, model, cfg, class_to_idx)
            print(f"  Saved best checkpoint -> {best_path} (val acc={best_val_acc:.4f})")

    # Save training artifacts
    save_metrics_csv(save_dir / "metrics.csv", history)
    plot_curves(save_dir, history)
    print(f"Saved training metrics/curves to: {save_dir}")

    # Optional evaluation pack (delegated to src/evaluation/evaluate.py)
    if cfg.run_eval:
        from evaluation.evaluate import run_evaluation_pack  # noqa: E402

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        eval_out = save_dir / "eval_best"
        run_evaluation_pack(
            checkpoint_path=best_path,
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            output_dir=eval_out,
        )
        print(f"Saved evaluation pack to: {eval_out}")


if __name__ == "__main__":
    main()