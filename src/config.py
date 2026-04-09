"""
Hyperparameters and project constants.
All experiments use the values defined in this file.
"""

# Dataset settings
SUBSET_SIZE = 1000  # Training subset size (kept small to ensure overfitting)
VALIDATION_SPLIT = 0.2  # Validation split ratio from training data
RANDOM_SEED = 42  # Fixed seed for reproducibility

# Model architecture
INPUT_SHAPE = (28, 28, 1)  # Fashion-MNIST image shape
NUM_CLASSES = 10  # Number of classes
HIDDEN_UNITS = [256, 128]  # Hidden layer neuron counts

# Training settings
EPOCHS = 100  # Maximum number of epochs
BATCH_SIZE = 32  # Mini-batch size
LEARNING_RATE = 1e-3  # Learning rate

# Regularization parameters
L2_LAMBDA = 0.01  # L2 regularization coefficient
DROPOUT_RATE = 0.5  # Dropout rate
EARLY_STOPPING_PATIENCE = 5  # Early stopping patience

# Class names (Fashion-MNIST)
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Experiment names
EXPERIMENT_NAMES = {
    "baseline": "Baseline (No Regularization)",
    "l2": "L2 Regularization",
    "dropout": "Dropout",
    "early_stopping": "Early Stopping",
}

# Output directory
RESULTS_DIR = "results"
