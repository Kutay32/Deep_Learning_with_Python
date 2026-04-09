"""
Optimizer factory — Week 5: SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam.
References: Goodfellow & Bengio, Deep Learning, Chapter 8.
"""

import tensorflow as tf

from src.presets import TrainingPreset


def create_optimizer(preset: TrainingPreset) -> tf.keras.optimizers.Optimizer:
    """Creates an optimizer instance from the preset. Includes gradient clipping."""

    if preset.optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(
            learning_rate=preset.learning_rate,
            momentum=preset.momentum,
            nesterov=preset.nesterov,
        )
    elif preset.optimizer == "adam":
        opt = tf.keras.optimizers.Adam(
            learning_rate=preset.learning_rate,
            beta_1=preset.beta_1,
            beta_2=preset.beta_2,
        )
    elif preset.optimizer == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(
            learning_rate=preset.learning_rate,
            rho=preset.rho,
        )
    elif preset.optimizer == "adagrad":
        opt = tf.keras.optimizers.Adagrad(
            learning_rate=preset.learning_rate,
        )
    else:
        raise ValueError(f"Unknown optimizer: {preset.optimizer}")

    if preset.gradient_clip_norm is not None:
        opt.clipnorm = preset.gradient_clip_norm

    return opt
