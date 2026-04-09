"""
Learning Rate Schedule factory — Week 5.
Supports Constant, Step Decay and Cosine Annealing schedules.
"""

import math
from typing import Optional

import tensorflow as tf

from src.presets import TrainingPreset


def create_lr_schedule_callback(preset: TrainingPreset) -> Optional[tf.keras.callbacks.Callback]:
    """Returns a LR schedule callback based on the preset. Returns None for constant."""

    if preset.lr_schedule == "constant":
        return None

    elif preset.lr_schedule == "step_decay":
        decay_factor = preset.lr_decay_factor
        decay_epochs = preset.lr_decay_epochs

        def step_decay_fn(epoch, lr):
            if epoch > 0 and epoch % decay_epochs == 0:
                return lr * decay_factor
            return lr

        return tf.keras.callbacks.LearningRateScheduler(step_decay_fn, verbose=0)

    elif preset.lr_schedule == "cosine":
        initial_lr = preset.learning_rate
        total_epochs = preset.epochs

        def cosine_fn(epoch, lr):
            return initial_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))

        return tf.keras.callbacks.LearningRateScheduler(cosine_fn, verbose=0)

    else:
        raise ValueError(f"Unknown lr_schedule: {preset.lr_schedule}")
