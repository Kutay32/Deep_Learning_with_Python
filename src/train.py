"""
Model training functions.
- train_with_preset(): Week 5 full preset-based pipeline.
- train_experiment() / run_all_experiments(): Kept for Week 4 backward compatibility.
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from src.config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE, EXPERIMENT_NAMES,
)
from src.model import MODEL_BUILDERS, build_model
from src.optimizers import create_optimizer
from src.schedulers import create_lr_schedule_callback
from src.callbacks import GradientNormCallback, LRHistoryCallback, NaNDetectorCallback


# ─────────────────────── Week 5: Preset-Based Training ────────────────────────

def train_with_preset(preset, x_train, y_train, x_test, y_test) -> dict:
    """
    Builds, trains and evaluates a model from the given preset.

    Returns:
        dict with keys: model, history, test_loss, test_acc, train_time,
                        gradient_norms, lr_history, convergence_epoch,
                        overfitting_gap, param_count, preset, diverged.
    """
    model = build_model(preset)
    optimizer = create_optimizer(preset)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = []

    lr_cb = create_lr_schedule_callback(preset)
    if lr_cb:
        callbacks.append(lr_cb)

    if preset.early_stopping:
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=preset.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ))

    x_sample = x_train[:256]
    y_sample = y_train[:256]
    grad_cb = GradientNormCallback(x_sample, y_sample)
    lr_hist_cb = LRHistoryCallback()
    nan_cb = NaNDetectorCallback()
    callbacks.extend([grad_cb, lr_hist_cb, nan_cb])

    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=preset.epochs,
        batch_size=preset.batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=0,
    )
    train_time = time.time() - t0

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    val_losses = history.history.get("val_loss", [0.0])
    convergence_epoch = int(np.argmin(val_losses)) + 1

    acc_hist = history.history.get("accuracy", [0.0])
    val_acc_hist = history.history.get("val_accuracy", [0.0])
    overfitting_gap = float(acc_hist[-1] - val_acc_hist[-1])

    return {
        "model": model,
        "history": history,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "train_time": train_time,
        "gradient_norms": grad_cb.gradient_norms,
        "lr_history": lr_hist_cb.lr_history,
        "convergence_epoch": convergence_epoch,
        "overfitting_gap": overfitting_gap,
        "param_count": model.count_params(),
        "preset": preset,
        "diverged": nan_cb.diverged,
    }


# ─────────────────────── Legacy Training (Week 4 backward compatibility) ──────────────────

def compile_model(model):
    """Compiles the model: Adam optimizer, crossentropy loss, accuracy metric."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_experiment(experiment_name, x_train, y_train):
    """
    Runs the specified experiment.

    Args:
        experiment_name: "baseline", "l2", "dropout" or "early_stopping"
        x_train: Training images
        y_train: Training labels

    Returns:
        model: Trained model
        history: Training history (loss, accuracy values)
    """
    print(f"\n{'='*60}")
    print(f"  Experiment: {EXPERIMENT_NAMES[experiment_name]}")
    print(f"{'='*60}")

    build_fn = MODEL_BUILDERS[experiment_name]
    model = build_fn()
    compile_model(model)

    callbacks = []
    if experiment_name == "early_stopping":
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )
        )

    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def run_all_experiments(x_train, y_train):
    """
    Runs all experiments sequentially.

    Returns:
        results: {experiment_name: (model, history)} dict
    """
    results = {}
    for name in MODEL_BUILDERS:
        model, history = train_experiment(name, x_train, y_train)
        results[name] = (model, history)
    return results
