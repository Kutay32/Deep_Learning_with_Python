"""
Hyperparameter Tuning Presets — Week 5: Optimization for Training Deep Models.
Each preset represents a specific optimization strategy.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class TrainingPreset:
    """Defines all parameters of a training configuration."""

    # Meta
    name: str
    description: str
    category: Literal["basic", "adaptive", "advanced"]

    # Optimizer
    optimizer: Literal["sgd", "adam", "rmsprop", "adagrad"]
    learning_rate: float
    momentum: float = 0.0
    nesterov: bool = False
    beta_1: float = 0.9
    beta_2: float = 0.999
    rho: float = 0.9

    # Initialization
    initializer: Literal["random_normal", "glorot_uniform", "he_normal", "he_uniform"] = "glorot_uniform"

    # Normalization
    normalization: Literal["none", "batch", "layer"] = "none"

    # Gradient Clipping
    gradient_clip_norm: Optional[float] = None

    # LR Schedule
    lr_schedule: Literal["constant", "step_decay", "cosine"] = "constant"
    lr_decay_factor: float = 0.1
    lr_decay_epochs: int = 30

    # Regularization
    regularization: Literal["none", "l2", "dropout"] = "none"
    l2_lambda: float = 0.01
    dropout_rate: float = 0.5

    # Training
    epochs: int = 100
    batch_size: int = 32
    early_stopping: bool = False
    early_stopping_patience: int = 5

    # Architecture
    hidden_units: list = field(default_factory=lambda: [256, 128])


PRESETS: dict[str, TrainingPreset] = {
    "vanilla_sgd": TrainingPreset(
        name="Vanilla SGD",
        description=(
            "The simplest optimizer. Very sensitive to learning rate, converges slowly. "
            "Without momentum, zigzags in narrow valleys. "
            "Goodfellow Algorithm 8.1 — Random Normal init for symmetry breaking."
        ),
        category="basic",
        optimizer="sgd",
        learning_rate=0.01,
        initializer="random_normal",
    ),
    "sgd_momentum": TrainingPreset(
        name="SGD + Momentum",
        description=(
            "Reduces oscillation via velocity accumulation. With α=0.9, "
            "terminal velocity is 10× plain gradient descent. "
            "Goodfellow Algorithm 8.2 — accumulates gradients in consistent directions."
        ),
        category="basic",
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        initializer="he_normal",
        lr_schedule="step_decay",
    ),
    "sgd_nesterov": TrainingPreset(
        name="SGD + Nesterov Momentum",
        description=(
            "Look-ahead gradient: performs a 'ghost jump' with current velocity, "
            "computes future gradient, prevents overshoot. "
            "Goodfellow Algorithm 8.3 — O(1/k²) convergence in convex batch case."
        ),
        category="basic",
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True,
        initializer="he_normal",
        lr_schedule="step_decay",
    ),
    "adam_default": TrainingPreset(
        name="Adam Default",
        description=(
            "Industry standard. 1st moment (momentum) + 2nd moment (RMSProp) + "
            "bias correction. Least sensitive to hyperparameter tuning. "
            "Goodfellow Algorithm 8.7 — lr=0.001, β₁=0.9, β₂=0.999."
        ),
        category="adaptive",
        optimizer="adam",
        learning_rate=0.001,
        initializer="he_normal",
    ),
    "adam_batchnorm": TrainingPreset(
        name="Adam + BatchNorm",
        description=(
            "Solves internal covariate shift. Stabilizes each layer's input distribution "
            "(mean=0, var=1). Expressive power preserved via learnable γ and β. "
            "Goodfellow Sec 8.7.1 — allows higher learning rates."
        ),
        category="adaptive",
        optimizer="adam",
        learning_rate=0.001,
        initializer="he_normal",
        normalization="batch",
    ),
    "adam_full_stack": TrainingPreset(
        name="Adam + Full Stack",
        description=(
            "Modern deep learning toolkit: Adam + He init + BatchNorm + "
            "gradient clipping + cosine annealing + dropout. "
            "Production-grade configuration — all Week 5 techniques combined."
        ),
        category="advanced",
        optimizer="adam",
        learning_rate=0.001,
        initializer="he_normal",
        normalization="batch",
        gradient_clip_norm=1.0,
        lr_schedule="cosine",
        regularization="dropout",
        dropout_rate=0.3,
    ),
    "rmsprop_stable": TrainingPreset(
        name="RMSProp Stable",
        description=(
            "Fixes AdaGrad's 'learning rate → 0' flaw via exponentially weighted "
            "moving average. Ideal for non-stationary loss surfaces. "
            "Goodfellow Algorithm 8.5 — ρ=0.9 decay factor."
        ),
        category="adaptive",
        optimizer="rmsprop",
        learning_rate=0.001,
        initializer="he_normal",
    ),
    "conservative": TrainingPreset(
        name="Conservative",
        description=(
            "Low LR + Xavier init + LayerNorm + L2 + cosine schedule + clipping. "
            "Slow but stable. Maximum protection against overfitting. "
            "Glorot for Sigmoid/Tanh — Var(output) = Var(input)."
        ),
        category="advanced",
        optimizer="adam",
        learning_rate=0.0005,
        initializer="glorot_uniform",
        normalization="layer",
        gradient_clip_norm=1.0,
        lr_schedule="cosine",
        regularization="l2",
        l2_lambda=0.01,
    ),
    "aggressive": TrainingPreset(
        name="Aggressive",
        description=(
            "High LR (0.05) + high momentum (0.99) + BatchNorm. "
            "Targets fast convergence but high instability risk. "
            "Gradient clipping=5.0 provides cliff structure protection."
        ),
        category="advanced",
        optimizer="sgd",
        learning_rate=0.05,
        momentum=0.99,
        initializer="he_normal",
        normalization="batch",
        gradient_clip_norm=5.0,
        lr_schedule="step_decay",
        lr_decay_factor=0.1,
        lr_decay_epochs=20,
    ),
}

PRESET_DISPLAY_NAMES: dict[str, str] = {k: v.name for k, v in PRESETS.items()}

CATEGORY_COLORS: dict[str, str] = {
    "basic": "#e74c3c",
    "adaptive": "#3498db",
    "advanced": "#2ecc71",
}

PRESET_COLORS: dict[str, str] = {
    "vanilla_sgd": "#e74c3c",
    "sgd_momentum": "#e67e22",
    "sgd_nesterov": "#f39c12",
    "adam_default": "#3498db",
    "adam_batchnorm": "#2980b9",
    "adam_full_stack": "#1abc9c",
    "rmsprop_stable": "#9b59b6",
    "conservative": "#27ae60",
    "aggressive": "#c0392b",
}
