"""
Visualization functions.
- Week 5: gradient norm, LR schedule, overlay comparison, overfitting gap, convergence speed.
- Legacy: plot_training_curves, plot_comparison_loss, plot_test_accuracy_bar, plot_confusion_matrix.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CLASS_NAMES, EXPERIMENT_NAMES, RESULTS_DIR
from src.presets import PRESET_COLORS


def _ensure_results_dir():
    """Creates the output directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_training_curves(history, experiment_name, save=True):
    """
    Plots train/validation loss and accuracy curves for a single experiment.
    """
    _ensure_results_dir()
    title = EXPERIMENT_NAMES[experiment_name]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_title(f"{title} — Loss Curve", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax2.set_title(f"{title} — Accuracy Curve", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(RESULTS_DIR, f"{experiment_name}_curves.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filepath}")

    plt.close(fig)


def plot_all_training_curves(results, save=True):
    """Plots training curves for all experiments side by side."""
    for name, (_, history) in results.items():
        plot_training_curves(history, name, save=save)


def plot_comparison_loss(results, save=True):
    """Compares validation loss curves for all experiments on a single chart."""
    _ensure_results_dir()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i, (name, (_, history)) in enumerate(results.items()):
        label = EXPERIMENT_NAMES[name]
        ax1.plot(history.history["val_loss"], label=label, linewidth=2, color=colors[i])
        ax2.plot(history.history["val_accuracy"], label=label, linewidth=2, color=colors[i])

    ax1.set_title("Validation Loss Comparison", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Accuracy Comparison", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(RESULTS_DIR, "comparison_curves.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filepath}")

    plt.close(fig)


def plot_test_accuracy_bar(metrics, save=True):
    """Shows test accuracy values for all experiments as a bar chart."""
    _ensure_results_dir()

    names = [EXPERIMENT_NAMES[k] for k in metrics]
    accuracies = [metrics[k][1] for k in metrics]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, accuracies, color=colors, edgecolor="black", linewidth=0.8)

    # Write value on top of each bar
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title("Test Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(RESULTS_DIR, "test_accuracy_comparison.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filepath}")

    plt.close(fig)


def plot_confusion_matrix(cm, experiment_name, save=True):
    """Plots confusion matrix as a heatmap."""
    _ensure_results_dir()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {EXPERIMENT_NAMES[experiment_name]}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    plt.tight_layout()

    if save:
        filepath = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filepath}")

    plt.close(fig)


# ──────────────────── Week 5: Preset Visualization Functions ──────────────────


def _get_color(preset_key: str, idx: int) -> str:
    fallback = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#c0392b", "#27ae60"]
    return PRESET_COLORS.get(preset_key, fallback[idx % len(fallback)])


def plot_preset_training_curves(result: dict) -> plt.Figure:
    """Loss/accuracy train+val overlay for a single preset (2 panels)."""
    h = result["history"].history
    preset = result["preset"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(h["loss"], label="Train", linewidth=2)
    ax1.plot(h["val_loss"], label="Val", linewidth=2, linestyle="--")
    ax1.set_title(f"{preset.name} — Loss", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(h["accuracy"], label="Train", linewidth=2)
    ax2.plot(h["val_accuracy"], label="Val", linewidth=2, linestyle="--")
    ax2.set_title(f"{preset.name} — Accuracy", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_lr_and_grad_norm(result: dict) -> plt.Figure:
    """LR schedule curve + gradient norm curve (2 panel)."""
    preset = result["preset"]
    lr_hist = result["lr_history"]
    grad_norms = result["gradient_norms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    if lr_hist:
        ax1.plot(lr_hist, linewidth=2, color="#3498db")
    ax1.set_title(f"{preset.name} — LR Schedule ({preset.lr_schedule})", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Learning Rate")
    ax1.grid(True, alpha=0.3)

    if grad_norms:
        ax2.plot(grad_norms, linewidth=2, color="#e74c3c")
    ax2.set_title(f"{preset.name} — Gradient L2 Norm", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Norm")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_overlay_comparison(results: dict) -> plt.Figure:
    """Overlay comparison of val_loss and val_accuracy for all selected presets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (key, r) in enumerate(results.items()):
        h = r["history"].history
        color = _get_color(key, idx)
        label = r["preset"].name
        ax1.plot(h["val_loss"], label=label, linewidth=2, color=color)
        ax2.plot(h["val_accuracy"], label=label, linewidth=2, color=color)

    ax1.set_title("Validation Loss Comparison", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Accuracy Comparison", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_test_accuracy_bar_presets(results: dict) -> plt.Figure:
    """Test accuracy bar chart for presets."""
    names = [r["preset"].name for r in results.values()]
    accs = [r["test_acc"] for r in results.values()]
    colors = [_get_color(k, i) for i, k in enumerate(results.keys())]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.4f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_overfitting_gap_bar(results: dict) -> plt.Figure:
    """Train_acc - val_acc overfitting gap bar chart for each preset."""
    names = [r["preset"].name for r in results.values()]
    gaps = [r["overfitting_gap"] for r in results.values()]
    colors = [_get_color(k, i) for i, k in enumerate(results.keys())]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))
    bars = ax.bar(names, gaps, color=colors, edgecolor="black", linewidth=0.8)
    for bar, gap in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002 if gap >= 0 else bar.get_height() - 0.012,
            f"{gap:.4f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Train Acc − Val Acc")
    ax.set_title("Overfitting Gap (higher = more overfit)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_convergence_speed_bar(results: dict) -> plt.Figure:
    """Epoch at which each preset reached its best val loss (convergence speed)."""
    names = [r["preset"].name for r in results.values()]
    epochs = [r["convergence_epoch"] for r in results.values()]
    colors = [_get_color(k, i) for i, k in enumerate(results.keys())]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))
    bars = ax.bar(names, epochs, color=colors, edgecolor="black", linewidth=0.8)
    for bar, ep in zip(bars, epochs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(ep), ha="center", fontsize=11, fontweight="bold"
        )
    ax.set_ylabel("Epoch")
    ax.set_title("Convergence Speed (lower = faster)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_confusion_matrix_preset(cm, preset_name: str) -> plt.Figure:
    """Preset ismiyle confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {preset_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()
    return fig
