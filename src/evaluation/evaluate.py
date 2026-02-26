#!/usr/bin/env python3
"""
src/evaluation/evaluate.py

Evaluation-only module: loads a checkpoint and saves a full classification
evaluation "pack" (figures + reports) for multiclass classification.

Outputs in output_dir:
- classification_report.txt
- per_class_metrics.csv
- confusion_matrix_counts.png
- confusion_matrix_normalized.png
- roc_curves_ovr.png
- pr_curves_ovr.png
- calibration_curve.png
- confidence_hist.png
- accuracy_vs_confidence.png
- top_confident_wrong.csv
- misclassified_grid.png (optional; requires dataset to return 'path' and 'sentence')
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], Optional[List[str]]]:
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    paths: List[str] = []
    sents: List[str] = []
    have_path = True
    have_sent = True

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(pixel_values, input_ids, attention_mask)
        probs = softmax_probs(logits)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.append(probs.detach().cpu().numpy())

        if have_path:
            if "path" in batch:
                paths.extend([str(p) for p in batch["path"]])
            else:
                have_path = False
        if have_sent:
            if "sentence" in batch:
                sents.extend([str(t) for t in batch["sentence"]])
            else:
                have_sent = False

    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    y_prob_np = np.concatenate(y_prob, axis=0) if y_prob else np.zeros((0, 0))

    return y_true_np, y_pred_np, y_prob_np, (paths if have_path else None), (sents if have_sent else None)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(save_path: Path, cm: np.ndarray, class_names: List[str], normalize: bool) -> None:
    mat = cm.astype(float)
    title = "Confusion Matrix (Test)"

    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums
        title += " - Normalized"

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(mat)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = f"{mat[i, j]*100:.1f}%" if normalize else str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_reports(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (output_dir / "classification_report.txt").write_text(report)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    with (output_dir / "per_class_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for i, name in enumerate(class_names):
            w.writerow([name, float(precision[i]), float(recall[i]), float(f1[i]), int(support[i])])


def plot_roc_pr(output_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> None:
    num_classes = len(class_names)
    if y_prob.size == 0 or y_prob.shape[1] != num_classes:
        return

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # ROC
    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest) - Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_ovr.png", dpi=200)
    plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, c], y_prob[:, c])
        ap = average_precision_score(y_true_bin[:, c], y_prob[:, c])
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (One-vs-Rest) - Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves_ovr.png", dpi=200)
    plt.close()


def plot_calibration(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    if y_prob.size == 0:
        return

    conf = y_prob.max(axis=1)
    correct = (y_pred == y_true).astype(float)

    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)

    bin_conf = np.zeros(n_bins, dtype=float)
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_count = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        m = bin_ids == b
        bin_count[b] = int(m.sum())
        if bin_count[b] > 0:
            bin_conf[b] = float(conf[m].mean())
            bin_acc[b] = float(correct[m].mean())

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(bin_conf, bin_acc, marker="o")
    plt.xlabel("Confidence (mean max prob)")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve (Max-Probability Binning) - Test")
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(conf, bins=20)
    plt.xlabel("Predicted confidence (max prob)")
    plt.ylabel("Count")
    plt.title("Confidence Histogram - Test")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(n_bins)
    plt.bar(x, bin_acc)
    plt.xticks(x, [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Confidence bin")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Confidence Bin - Test")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_confidence.png", dpi=200)
    plt.close()


def save_top_confident_wrong(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    paths: Optional[List[str]],
    sentences: Optional[List[str]],
    top_k: int = 50,
) -> None:
    conf = y_prob.max(axis=1) if y_prob.size else np.zeros_like(y_true, dtype=float)
    wrong = np.where(y_true != y_pred)[0]
    sorted_wrong = wrong[np.argsort(-conf[wrong])] if wrong.size else np.array([], dtype=int)
    sorted_wrong = sorted_wrong[:top_k]

    out = output_dir / "top_confident_wrong.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "confidence", "true", "pred", "path", "sentence"])
        for r, i in enumerate(sorted_wrong, start=1):
            w.writerow([
                r,
                float(conf[i]),
                class_names[int(y_true[i])],
                class_names[int(y_pred[i])],
                (paths[i] if paths else ""),
                (sentences[i] if sentences else ""),
            ])


def save_misclassified_grid(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    paths: Optional[List[str]],
    sentences: Optional[List[str]],
    max_items: int = 25,
) -> None:
    if paths is None:
        return

    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        return

    conf = y_prob.max(axis=1) if y_prob.size else np.zeros_like(y_true, dtype=float)
    pick = wrong[np.argsort(-conf[wrong])][:max_items]

    import PIL.Image

    n = int(min(max_items, pick.size))
    cols = 5
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for k in range(n):
        i = int(pick[k])
        img = PIL.Image.open(paths[i]).convert("RGB")

        ax = plt.subplot(rows, cols, k + 1)
        ax.imshow(img)
        ax.axis("off")

        t = class_names[int(y_true[i])]
        p = class_names[int(y_pred[i])]
        c = float(conf[i])
        sent = sentences[i] if (sentences is not None and i < len(sentences)) else ""
        title = f"T:{t}  P:{p} ({c:.2f})"
        if sent:
            title += "\n" + sent[:40]
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "misclassified_grid.png", dpi=200)
    plt.close()


def run_evaluation_pack(
    *,
    checkpoint_path: Path,
    model,
    test_loader,
    device: torch.device,
    class_names: List[str],
    output_dir: Path,
) -> None:
    """
    Main entry point: loads checkpoint into the provided model instance,
    runs test inference, and writes all evaluation artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    y_true, y_pred, y_prob, paths, sentences = collect_predictions(model, test_loader, device)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    plot_confusion_matrix(output_dir / "confusion_matrix_counts.png", cm, class_names, normalize=False)
    plot_confusion_matrix(output_dir / "confusion_matrix_normalized.png", cm, class_names, normalize=True)

    save_reports(output_dir, y_true, y_pred, class_names)
    plot_roc_pr(output_dir, y_true, y_prob, class_names)
    plot_calibration(output_dir, y_true, y_pred, y_prob)

    save_top_confident_wrong(output_dir, y_true, y_pred, y_prob, class_names, paths, sentences)
    save_misclassified_grid(output_dir, y_true, y_pred, y_prob, class_names, paths, sentences)