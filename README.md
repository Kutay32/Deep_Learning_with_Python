# Deep Learning Optimization Sandbox — Week 5

A Streamlit-based interactive dashboard for the comparative analysis of deep learning optimization strategies on the Fashion-MNIST dataset.

Covers **Goodfellow & Bengio, Deep Learning, Chapter 8**: optimizers, weight initialization, normalization, gradient clipping, and learning rate scheduling — all configurable through a preset system.

---

## Overview

The project deliberately uses a **small training subset (1 000 samples)** from Fashion-MNIST so that differences between optimization strategies are clearly visible. Each **preset** bundles a complete training configuration (optimizer, initializer, normalization, regularization, LR schedule) into a single selectable profile.

---

## Presets

| Key | Name | Category | Key Concepts |
|---|---|---|---|
| `vanilla_sgd` | Vanilla SGD | Basic | Algorithm 8.1 · Random Normal init · zigzag |
| `sgd_momentum` | SGD + Momentum | Basic | Algorithm 8.2 · α=0.9 · terminal velocity |
| `sgd_nesterov` | SGD + Nesterov Momentum | Basic | Algorithm 8.3 · look-ahead gradient |
| `adam_default` | Adam Default | Adaptive | Algorithm 8.7 · bias correction |
| `adam_batchnorm` | Adam + BatchNorm | Adaptive | Sec 8.7.1 · internal covariate shift |
| `adam_full_stack` | Adam + Full Stack | Advanced | All Week 5 techniques combined |
| `rmsprop_stable` | RMSProp Stable | Adaptive | Algorithm 8.5 · EWMA decay |
| `conservative` | Conservative | Advanced | Xavier · LayerNorm · L2 · cosine schedule |
| `aggressive` | Aggressive | Advanced | High LR/momentum · divergence risk |

---

## Project Structure

```
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit dashboard (main entry point)
├── main.py                 # CLI training script
├── src/
│   ├── presets.py          # TrainingPreset dataclass and PRESETS dict
│   ├── config.py           # Hyperparameters and global constants
│   ├── data.py             # Dataset loading and preprocessing
│   ├── model.py            # MLP builder (preset-based + legacy)
│   ├── train.py            # Training pipeline (preset-based + legacy)
│   ├── optimizers.py       # Optimizer factory (SGD/Adam/RMSProp/AdaGrad)
│   ├── schedulers.py       # LR schedule factory (constant/step/cosine)
│   ├── callbacks.py        # GradientNormCallback, LRHistoryCallback, NaNDetectorCallback
│   ├── evaluate.py         # Metrics and comparison table builder
│   └── visualize.py        # All plotting functions
├── notebooks/
│   └── experiment.ipynb    # Interactive experiment notebook
└── results/                # Saved output plots
```

---

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Running

### Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**How to use the dashboard:**
1. Select an **Optimization preset** from the sidebar.
2. Optionally expand **Manual Override** to tweak individual parameters.
3. Click **Start Training** to train the active preset.
4. Click **Compare** to train multiple presets side-by-side.
5. Inspect results across three tabs: **Training Curves**, **Comparison**, **Model Details**.

### CLI — Single Preset

```bash
python main.py --preset adam_default
```

### CLI — Compare All Presets

```bash
python main.py --compare
```

### CLI — Legacy Week 4 Experiments

```bash
python main.py
```

Runs Baseline / L2 / Dropout / Early Stopping experiments and saves plots to `results/`.

---

## Dashboard Features

| Tab | Contents |
|---|---|
| **Training Curves** | Loss & accuracy curves · LR schedule · gradient norm history · per-epoch stats |
| **Comparison** | Validation loss/accuracy overlay · test accuracy bar · convergence speed · overfitting gap · metrics table |
| **Model Details** | Confusion matrix · classification report · model summary · full active configuration · educational notes |

---

## Key Technical Concepts

| Concept | Goodfellow Reference | Presets |
|---|---|---|
| SGD variants | Algorithms 8.1–8.3 | `vanilla_sgd`, `sgd_momentum`, `sgd_nesterov` |
| Adam optimizer | Algorithm 8.7 | `adam_default`, `adam_batchnorm`, `adam_full_stack` |
| RMSProp | Algorithm 8.5 | `rmsprop_stable` |
| Weight initialization | Glorot / He | all presets |
| Batch Normalization | Section 8.7.1 | `adam_batchnorm`, `adam_full_stack`, `aggressive` |
| Layer Normalization | — | `conservative` |
| Gradient Clipping | Section 8.2.4 | `adam_full_stack`, `conservative`, `aggressive` |
| LR Scheduling | — | step decay, cosine annealing |
| Regularization | — | L2 (`conservative`), Dropout (`adam_full_stack`) |

---

## Tech Stack

- **TensorFlow / Keras** — Model building and training
- **Streamlit** — Interactive web dashboard
- **Matplotlib / Seaborn** — Visualization
- **Scikit-learn** — Confusion matrix and classification report
- **Pandas** — Comparison table
- **NumPy** — Numerical operations
