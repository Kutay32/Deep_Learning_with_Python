"""
Main run script — Week 5 Preset support added.

Usage:
  python main.py                        # All Week 4 experiments (legacy behaviour)
  python main.py --preset adam_default  # Run a specific preset
  python main.py --compare              # Compare all presets
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import prepare_data
from src.evaluate import build_comparison_table, compute_confusion_matrix, evaluate_all, get_best_experiment
from src.presets import PRESETS
from src.train import run_all_experiments, train_with_preset
from src.visualize import (
    plot_all_training_curves,
    plot_comparison_loss,
    plot_confusion_matrix,
    plot_overlay_comparison,
    plot_test_accuracy_bar,
    plot_test_accuracy_bar_presets,
)


def run_legacy(x_train, y_train, x_test, y_test):
    """Week 4: regularization experiments."""
    print("\n[2/4] Running experiments...")
    results = run_all_experiments(x_train, y_train)

    print("\n[3/4] Evaluating on test set...")
    metrics = evaluate_all(results, x_test, y_test)

    print("\n[4/4] Generating plots...")
    plot_all_training_curves(results)
    plot_comparison_loss(results)
    plot_test_accuracy_bar(metrics)

    best_name = get_best_experiment(metrics)
    best_model = results[best_name][0]
    cm, report = compute_confusion_matrix(best_model, x_test, y_test)
    plot_confusion_matrix(cm, best_name)
    print(f"\nClassification Report ({best_name}):\n{report}")


def run_preset(preset_key, x_train, y_train, x_test, y_test):
    """Single preset training."""
    preset = PRESETS[preset_key]
    print(f"\n  Preset: {preset.name}")
    print(f"  {preset.description}\n")
    result = train_with_preset(preset, x_train, y_train, x_test, y_test)
    print(f"  Test Acc : {result['test_acc']:.4f}")
    print(f"  Test Loss: {result['test_loss']:.4f}")
    print(f"  Conv.Ep. : {result['convergence_epoch']}")
    print(f"  OvFit Gap: {result['overfitting_gap']:+.4f}")
    print(f"  Params   : {result['param_count']:,}")
    if result["diverged"]:
        print("  DIVERGED — NaN/Inf detected!")


def run_compare(x_train, y_train, x_test, y_test):
    """Train all presets sequentially and compare."""
    compare_results = {}
    for key, preset in PRESETS.items():
        print(f"  Training: {preset.name}…")
        result = train_with_preset(preset, x_train, y_train, x_test, y_test)
        compare_results[key] = result
        status = "DIVERGED" if result["diverged"] else f"acc={result['test_acc']:.4f}"
        print(f"    → {status}")

    print("\n  Comparison table:")
    df = build_comparison_table(compare_results)
    print(df.to_string())

    fig_ov = plot_overlay_comparison(compare_results)
    fig_ov.savefig("results/week5_overlay_comparison.png", dpi=150, bbox_inches="tight")

    fig_bar = plot_test_accuracy_bar_presets(compare_results)
    fig_bar.savefig("results/week5_test_accuracy_bar.png", dpi=150, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close("all")
    print("\n  Plots saved to 'results/' directory.")


def main():
    parser = argparse.ArgumentParser(description="Deep Learning with Python — CLI")
    parser.add_argument(
        "--preset", type=str, default=None, choices=list(PRESETS.keys()), help="Preset to run (e.g. adam_default)"
    )
    parser.add_argument("--compare", action="store_true", help="Compare all presets")
    args = parser.parse_args()

    if args.preset:
        title = f"PROJECT: Week 5 Preset — {PRESETS[args.preset].name}"
    elif args.compare:
        title = "PROJECT: Week 5 — All Preset Comparison"
    else:
        title = "PROJECT 3: Comparative Analysis of Regularization Techniques"

    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

    print("\n[1/N] Preparing dataset...")
    x_train, y_train, x_test, y_test = prepare_data()

    if args.preset:
        run_preset(args.preset, x_train, y_train, x_test, y_test)
    elif args.compare:
        run_compare(x_train, y_train, x_test, y_test)
    else:
        run_legacy(x_train, y_train, x_test, y_test)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
