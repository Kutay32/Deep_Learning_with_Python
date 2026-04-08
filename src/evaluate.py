"""
Model evaluation functions.
- build_comparison_table(): Comparison table from Week 5 preset results.
- Legacy evaluate_all() / compute_confusion_matrix() kept for backward compatibility.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.config import CLASS_NAMES, EXPERIMENT_NAMES


# ─────────────────────── Week 5: Preset Metric Tools ───────────────────────

def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Builds a comparison table from multiple preset results.
    results: {preset_key: train_with_preset() output dict}
    """
    rows = []
    for key, r in results.items():
        h = r["history"].history
        rows.append({
            "Preset": r["preset"].name,
            "Test Accuracy": round(r["test_acc"], 4),
            "Test Loss": round(r["test_loss"], 4),
            "Best Val Loss": round(float(np.min(h.get("val_loss", [0]))), 4),
            "Convergence Epoch": r["convergence_epoch"],
            "Training Time (s)": round(r["train_time"], 1),
            "Overfitting Gap": round(r["overfitting_gap"], 4),
            "Final Grad Norm": round(r["gradient_norms"][-1], 4) if r["gradient_norms"] else 0.0,
            "Param Count": r["param_count"],
        })
    return pd.DataFrame(rows).set_index("Preset")


def compute_confusion_matrix_preset(result: dict, x_test, y_test):
    """Produces confusion matrix and report from a preset training result."""
    model = result["model"]
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes, target_names=CLASS_NAMES, digits=4)
    return cm, report


# ─────────────────────── Legacy (Week 4 backward compatibility) ──────────────────────────

def evaluate_model(model, x_test, y_test, experiment_name):
    """Evaluates a model on the test set."""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[{EXPERIMENT_NAMES[experiment_name]}]  "
          f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc


def evaluate_all(results, x_test, y_test):
    """Computes test performance for all experiments."""
    print(f"\n{'='*60}")
    print("  TEST SET RESULTS")
    print(f"{'='*60}")
    metrics = {}
    for name, (model, _) in results.items():
        test_loss, test_acc = evaluate_model(model, x_test, y_test, name)
        metrics[name] = (test_loss, test_acc)
    return metrics


def get_best_experiment(metrics):
    """Returns the experiment with the highest test accuracy."""
    best_name = max(metrics, key=lambda k: metrics[k][1])
    best_loss, best_acc = metrics[best_name]
    print(f"\nBest model: {EXPERIMENT_NAMES[best_name]} (Accuracy: {best_acc:.4f})")
    return best_name


def compute_confusion_matrix(model, x_test, y_test):
    """Computes confusion matrix and classification report."""
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes, target_names=CLASS_NAMES, digits=4)
    return cm, report
