"""
Create presentation-ready visualizations for DINOv2 stage-1 metrics.

Examples:
    python visualize_dinov2_metrics.py
    python visualize_dinov2_metrics.py --input_dir output_stage1
    python visualize_dinov2_metrics.py --input_dir output_stage1 --output_dir output_stage1/figures
"""

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DINOv2 metrics for presentations")
    parser.add_argument("--input_dir", type=str, default="output_stage1",
                        help="Directory containing training_log.csv, test_predictions.csv, and optionally test_results.txt")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save generated figures; defaults to <input_dir>/figures")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    return parser.parse_args()


def ensure_output_dir(input_dir: Path, output_dir_arg: str | None) -> Path:
    output_dir = Path(output_dir_arg) if output_dir_arg else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_training_log(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "val_loss": float(row["val_loss"]),
                "val_acc": float(row["val_acc"]),
                "lr": float(row["lr"]),
            })
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_predictions(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image": row["image"],
                "ground_truth": row["ground_truth"],
                "prediction": row["prediction"],
                "confidence": float(row["confidence"]),
                "correct": str(row["correct"]).lower() == "true",
            })
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def parse_test_results(path: Path):
    metrics = {}
    if not path.exists():
        return metrics

    text = path.read_text()

    acc_match = re.search(r"Test accuracy:\s*([0-9.]+)\s*\((\d+)/(\d+)\)", text)
    if acc_match:
        metrics["accuracy"] = float(acc_match.group(1))
        metrics["correct"] = int(acc_match.group(2))
        metrics["total"] = int(acc_match.group(3))

    time_match = re.search(r"Avg inference time:\s*([0-9.]+)\s*ms/image", text)
    if time_match:
        metrics["avg_inference_ms"] = float(time_match.group(1))

    return metrics


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def compute_confusion_matrix(y_true, y_pred, class_names):
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            cm[label_to_idx[true_label], label_to_idx[pred_label]] += 1
    return cm


def compute_classification_report(y_true, y_pred, class_names):
    cm = compute_confusion_matrix(y_true, y_pred, class_names)
    support = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    total = cm.sum()
    correct = np.trace(cm)

    report = {}
    precisions, recalls, f1s = [], [], []

    for idx, class_name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = pred_totals[idx] - tp
        fn = support[idx] - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(support[idx]),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    weighted_precision = sum(report[c]["precision"] * report[c]["support"] for c in class_names) / total if total > 0 else 0.0
    weighted_recall = sum(report[c]["recall"] * report[c]["support"] for c in class_names) / total if total > 0 else 0.0
    weighted_f1 = sum(report[c]["f1-score"] * report[c]["support"] for c in class_names) / total if total > 0 else 0.0

    report["accuracy"] = correct / total if total > 0 else 0.0
    report["macro avg"] = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1-score": float(np.mean(f1s)) if f1s else 0.0,
        "support": int(total),
    }
    report["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1,
        "support": int(total),
    }
    return report


def save_training_curves(training_rows, output_dir: Path, dpi: int):
    epochs = [r["epoch"] for r in training_rows]
    train_loss = [r["train_loss"] for r in training_rows]
    val_loss = [r["val_loss"] for r in training_rows]
    train_acc = [r["train_acc"] for r in training_rows]
    val_acc = [r["val_acc"] for r in training_rows]
    lrs = [r["lr"] for r in training_rows]

    best_epoch = int(np.argmax(val_acc))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].plot(epochs, train_loss, color="#0f766e", linewidth=2.5, label="Train Loss")
    axes[0].plot(epochs, val_loss, color="#dc2626", linewidth=2.5, label="Val Loss")
    axes[0].axvline(epochs[best_epoch], linestyle="--", color="#6b7280", alpha=0.8, linewidth=1.5)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, np.array(train_acc) * 100, color="#0369a1", linewidth=2.5, label="Train Accuracy")
    axes[1].plot(epochs, np.array(val_acc) * 100, color="#ea580c", linewidth=2.5, label="Val Accuracy")
    axes[1].scatter([epochs[best_epoch]], [val_acc[best_epoch] * 100], color="#111827", s=60, zorder=5)
    axes[1].annotate(
        f"Best {val_acc[best_epoch] * 100:.2f}%\nEpoch {epochs[best_epoch]}",
        (epochs[best_epoch], val_acc[best_epoch] * 100),
        xytext=(10, -28),
        textcoords="offset points",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(min(np.array(val_acc) * 100) - 1.0, 100.2)
    axes[1].legend(loc="lower right")

    axes[2].plot(epochs, lrs, color="#7c3aed", linewidth=2.5)
    axes[2].fill_between(epochs, lrs, color="#c4b5fd", alpha=0.35)
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")

    fig.suptitle("DINOv2 Stage-1 Training Summary", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(y_true, y_pred, class_names, output_dir: Path, dpi: int):
    cm = compute_confusion_matrix(y_true, y_pred, class_names)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xticks(np.arange(len(class_names)))
    axes[0].set_yticks(np.arange(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Ground Truth")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "#111827"
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cm_norm, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axes[1].set_title("Normalized Confusion Matrix")
    axes[1].set_xticks(np.arange(len(class_names)))
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Ground Truth")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "#111827"
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Test Set Class-wise Performance", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_class_metrics_figure(report_dict, class_names, output_dir: Path, dpi: int):
    precision = [report_dict[c]["precision"] * 100 for c in class_names]
    recall = [report_dict[c]["recall"] * 100 for c in class_names]
    f1 = [report_dict[c]["f1-score"] * 100 for c in class_names]
    support = [report_dict[c]["support"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3.0, 1.4])

    axes[0].bar(x - width, precision, width, label="Precision", color="#2563eb")
    axes[0].bar(x, recall, width, label="Recall", color="#059669")
    axes[0].bar(x + width, f1, width, label="F1-score", color="#ea580c")
    axes[0].set_title("Per-class Metrics")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=35, ha="right")
    axes[0].legend(ncols=3, loc="lower center")

    bars = axes[1].bar(x, support, color="#7c3aed", alpha=0.85)
    axes[1].set_title("Test Samples per Class")
    axes[1].set_ylabel("Support")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=35, ha="right")
    for bar, val in zip(bars, support):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(val),
                     ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "per_class_metrics.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_overview_dashboard(training_rows, predictions, report_dict, test_metrics, output_dir: Path, dpi: int):
    accuracy = sum(r["correct"] for r in predictions) / len(predictions)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]
    avg_conf = float(np.mean([r["confidence"] for r in predictions]))
    avg_conf_correct = float(np.mean([r["confidence"] for r in predictions if r["correct"]]))
    avg_conf_wrong = float(np.mean([r["confidence"] for r in predictions if not r["correct"]])) if any(
        not r["correct"] for r in predictions
    ) else 0.0
    best_val_acc = max(r["val_acc"] for r in training_rows)
    final_train_acc = training_rows[-1]["train_acc"]
    final_val_acc = training_rows[-1]["val_acc"]
    avg_inference_ms = test_metrics.get("avg_inference_ms")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 2.0], hspace=0.35, wspace=0.28)

    ax_cards = fig.add_subplot(gs[0, :])
    ax_cards.axis("off")

    cards = [
        ("Test Accuracy", f"{accuracy * 100:.2f}%"),
        ("Macro F1", f"{macro_f1 * 100:.2f}%"),
        ("Weighted F1", f"{weighted_f1 * 100:.2f}%"),
        ("Best Val Acc", f"{best_val_acc * 100:.2f}%"),
        ("Avg Confidence", f"{avg_conf * 100:.2f}%"),
        ("Avg Latency", f"{avg_inference_ms:.2f} ms" if avg_inference_ms is not None else "N/A"),
    ]
    card_colors = ["#dbeafe", "#dcfce7", "#ffedd5", "#ede9fe", "#fef3c7", "#fee2e2"]

    for idx, ((title, value), bg) in enumerate(zip(cards, card_colors)):
        x0 = 0.015 + idx * 0.162
        rect = plt.Rectangle((x0, 0.15), 0.145, 0.72, transform=ax_cards.transAxes,
                             facecolor=bg, edgecolor="#d1d5db", linewidth=1.2)
        ax_cards.add_patch(rect)
        ax_cards.text(x0 + 0.0725, 0.62, title, ha="center", va="center",
                      fontsize=12, fontweight="bold", color="#1f2937", transform=ax_cards.transAxes)
        ax_cards.text(x0 + 0.0725, 0.36, value, ha="center", va="center",
                      fontsize=19, fontweight="bold", color="#111827", transform=ax_cards.transAxes)

    ax_acc = fig.add_subplot(gs[1, 0])
    epochs = [r["epoch"] for r in training_rows]
    ax_acc.plot(epochs, np.array([r["train_acc"] for r in training_rows]) * 100, color="#0284c7", linewidth=2.5)
    ax_acc.plot(epochs, np.array([r["val_acc"] for r in training_rows]) * 100, color="#f97316", linewidth=2.5)
    ax_acc.set_title("Train vs Val Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend(["Train", "Val"], loc="lower right")

    ax_conf = fig.add_subplot(gs[1, 1])
    correct_conf = [r["confidence"] for r in predictions if r["correct"]]
    wrong_conf = [r["confidence"] for r in predictions if not r["correct"]]
    bins = np.linspace(0, 1, 16)
    ax_conf.hist(correct_conf, bins=bins, alpha=0.8, color="#10b981", label="Correct")
    if wrong_conf:
        ax_conf.hist(wrong_conf, bins=bins, alpha=0.8, color="#ef4444", label="Wrong")
    ax_conf.set_title("Prediction Confidence Distribution")
    ax_conf.set_xlabel("Confidence")
    ax_conf.set_ylabel("Count")
    ax_conf.legend()

    ax_gap = fig.add_subplot(gs[1, 2])
    categories = ["Final Train", "Final Val", "Best Val", "Correct Conf", "Wrong Conf"]
    values = [
        final_train_acc * 100,
        final_val_acc * 100,
        best_val_acc * 100,
        avg_conf_correct * 100,
        avg_conf_wrong * 100 if wrong_conf else 0.0,
    ]
    colors = ["#2563eb", "#f97316", "#7c3aed", "#059669", "#dc2626"]
    bars = ax_gap.bar(categories, values, color=colors, alpha=0.9)
    ax_gap.set_title("Snapshot")
    ax_gap.set_ylabel("Value (%)")
    ax_gap.set_ylim(0, 105)
    ax_gap.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax_gap.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=10)

    fig.suptitle("DINOv2 Stage-1 Presentation Dashboard", y=0.98)
    fig.savefig(output_dir / "metrics_dashboard.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_output_dir(input_dir, args.output_dir)

    training_log_path = input_dir / "training_log.csv"
    predictions_path = input_dir / "test_predictions.csv"
    test_results_path = input_dir / "test_results.txt"

    if not training_log_path.exists():
        raise FileNotFoundError(f"Missing file: {training_log_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing file: {predictions_path}")

    set_plot_style()

    training_rows = read_training_log(training_log_path)
    predictions = read_predictions(predictions_path)
    test_metrics = parse_test_results(test_results_path)

    class_names = sorted({row["ground_truth"] for row in predictions})
    y_true = [row["ground_truth"] for row in predictions]
    y_pred = [row["prediction"] for row in predictions]

    report_dict = compute_classification_report(y_true, y_pred, class_names)

    save_training_curves(training_rows, output_dir, args.dpi)
    save_confusion_matrix_figure(y_true, y_pred, class_names, output_dir, args.dpi)
    save_class_metrics_figure(report_dict, class_names, output_dir, args.dpi)
    save_overview_dashboard(training_rows, predictions, report_dict, test_metrics, output_dir, args.dpi)

    print("Saved figures:")
    for name in [
        "training_curves.png",
        "confusion_matrix.png",
        "per_class_metrics.png",
        "metrics_dashboard.png",
    ]:
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    main()
