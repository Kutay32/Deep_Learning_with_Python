"""
Model architectures.
- build_model(preset): Week 5 preset-based builder (supports init, norm, reg).
- Legacy builders (build_baseline_model etc.) kept for backward compatibility.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from src.config import INPUT_SHAPE, NUM_CLASSES, HIDDEN_UNITS, L2_LAMBDA, DROPOUT_RATE


# ─────────────────────── Week 5: Preset-Based Builder ───────────────────────

def build_model(preset) -> tf.keras.Model:
    """
    Builds an MLP model from preset parameters.
    Supports initialization, normalization, regularization and architecture.
    """
    initializer = _get_initializer(preset.initializer)
    layer_list = [layers.Flatten(input_shape=INPUT_SHAPE)]

    for units in preset.hidden_units:
        dense_kwargs = {
            "units": units,
            "activation": "relu" if preset.normalization == "none" else None,
            "kernel_initializer": initializer,
        }
        if preset.regularization == "l2":
            dense_kwargs["kernel_regularizer"] = regularizers.l2(preset.l2_lambda)

        layer_list.append(layers.Dense(**dense_kwargs))

        if preset.normalization == "batch":
            layer_list.append(layers.BatchNormalization())
        elif preset.normalization == "layer":
            layer_list.append(layers.LayerNormalization())

        if preset.normalization != "none":
            layer_list.append(layers.Activation("relu"))

        if preset.regularization == "dropout":
            layer_list.append(layers.Dropout(preset.dropout_rate))

    layer_list.append(
        layers.Dense(NUM_CLASSES, activation="softmax", kernel_initializer=initializer)
    )
    return models.Sequential(layer_list)


def _get_initializer(name: str) -> tf.keras.initializers.Initializer:
    return {
        "random_normal": tf.keras.initializers.RandomNormal(stddev=0.05),
        "glorot_uniform": tf.keras.initializers.GlorotUniform(),
        "he_normal": tf.keras.initializers.HeNormal(),
        "he_uniform": tf.keras.initializers.HeUniform(),
    }[name]


# ─────────────────────── Legacy Builders (Week 4 backward compatibility) ──────────────────

def build_baseline_model():
    """Baseline model without regularization (overfitting expected)."""
    model = models.Sequential([
        layers.Flatten(input_shape=INPUT_SHAPE),
        layers.Dense(HIDDEN_UNITS[0], activation="relu"),
        layers.Dense(HIDDEN_UNITS[1], activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model


def build_l2_model():
    """Model with L2 (weight decay) regularization."""
    model = models.Sequential([
        layers.Flatten(input_shape=INPUT_SHAPE),
        layers.Dense(
            HIDDEN_UNITS[0], activation="relu",
            kernel_regularizer=regularizers.l2(L2_LAMBDA),
        ),
        layers.Dense(
            HIDDEN_UNITS[1], activation="relu",
            kernel_regularizer=regularizers.l2(L2_LAMBDA),
        ),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model


def build_dropout_model():
    """Model with dropout regularization."""
    model = models.Sequential([
        layers.Flatten(input_shape=INPUT_SHAPE),
        layers.Dense(HIDDEN_UNITS[0], activation="relu"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(HIDDEN_UNITS[1], activation="relu"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model


def build_early_stopping_model():
    """
    Model to be used with Early Stopping.
    Architecture is identical to baseline; the difference is the training callback.
    """
    return build_baseline_model()


MODEL_BUILDERS = {
    "baseline": build_baseline_model,
    "l2": build_l2_model,
    "dropout": build_dropout_model,
    "early_stopping": build_early_stopping_model,
}
