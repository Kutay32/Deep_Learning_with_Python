"""
Custom Keras Callbacks — Week 5.
GradientNormCallback: Logs gradient L2 norm at the end of each epoch.
LRHistoryCallback: Records the effective learning rate at the end of each epoch.
NaNDetectorCallback: Stops training when NaN/Inf gradient or loss is detected.
"""

import tensorflow as tf
import numpy as np


class GradientNormCallback(tf.keras.callbacks.Callback):
    """Computes and logs gradient L2 norm at the end of each epoch."""

    def __init__(self, x_sample, y_sample):
        super().__init__()
        self.x_sample = tf.constant(x_sample, dtype=tf.float32)
        self.y_sample = tf.constant(y_sample, dtype=tf.int32)
        self.gradient_norms = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            with tf.GradientTape() as tape:
                y_pred = self.model(self.x_sample, training=False)
                loss = self.model.compiled_loss(self.y_sample, y_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
            valid_grads = [g for g in grads if g is not None]
            if valid_grads:
                total_norm = tf.sqrt(
                    tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in valid_grads])
                )
                norm_val = float(total_norm.numpy())
                self.gradient_norms.append(norm_val if np.isfinite(norm_val) else 0.0)
            else:
                self.gradient_norms.append(0.0)
        except Exception:
            self.gradient_norms.append(0.0)


class LRHistoryCallback(tf.keras.callbacks.Callback):
    """Records the effective learning rate at the end of each epoch."""

    def __init__(self):
        super().__init__()
        self.lr_history = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            self.lr_history.append(lr)
        except Exception:
            self.lr_history.append(0.0)


class NaNDetectorCallback(tf.keras.callbacks.Callback):
    """Stops training and sets a warning flag if loss becomes NaN/Inf."""

    def __init__(self):
        super().__init__()
        self.diverged = False

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", 0.0) if logs else 0.0
        if not np.isfinite(loss):
            self.diverged = True
            self.model.stop_training = True
