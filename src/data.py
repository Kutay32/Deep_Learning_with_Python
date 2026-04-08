"""
Fashion-MNIST dataset loading, subset selection and preprocessing.
"""

import numpy as np
import tensorflow as tf
from src.config import SUBSET_SIZE, RANDOM_SEED, NUM_CLASSES


def load_fashion_mnist():
    """Loads and normalizes the Fashion-MNIST dataset."""
    (x_train_full, y_train_full), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    # [0, 255] -> [0, 1] normalize
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (28, 28) -> (28, 28, 1)
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train_full, y_train_full), (x_test, y_test)


def create_subset(x_train_full, y_train_full, subset_size=SUBSET_SIZE):
    """
    Randomly selects a small subset from the training set.
    A small training set is used to reliably induce overfitting.
    """
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(x_train_full), size=subset_size, replace=False)
    x_subset = x_train_full[indices]
    y_subset = y_train_full[indices]
    return x_subset, y_subset


def prepare_data():
    """
    Runs all data preparation steps.

    Returns:
        x_train: Training subset images
        y_train: Training subset labels
        x_test:  Test set images
        y_test:  Test set labels
    """
    (x_train_full, y_train_full), (x_test, y_test) = load_fashion_mnist()
    x_train, y_train = create_subset(x_train_full, y_train_full)

    print(f"Training subset size : {x_train.shape[0]}")
    print(f"Test set size        : {x_test.shape[0]}")
    print(f"Image shape          : {x_train.shape[1:]}")
    print(f"Number of classes    : {NUM_CLASSES}")

    return x_train, y_train, x_test, y_test
