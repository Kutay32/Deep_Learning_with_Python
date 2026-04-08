"""
Deep Learning Optimization Sandbox — Week 5
Preset-based comparative analysis of optimizer, initialization, normalization,
gradient clipping and LR scheduling on Fashion-MNIST.
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from src.presets import PRESETS, TrainingPreset, PRESET_COLORS
from src.data import load_fashion_mnist, create_subset
from src.train import train_with_preset
from src.evaluate import build_comparison_table, compute_confusion_matrix_preset
from src.visualize import (
    plot_preset_training_curves,
    plot_lr_and_grad_norm,
    plot_overlay_comparison,
    plot_test_accuracy_bar_presets,
    plot_overfitting_gap_bar,
    plot_convergence_speed_bar,
    plot_confusion_matrix_preset,
)
from src.config import CLASS_NAMES

# ──────────────────────────── Page Config ─────────────────────────────
st.set_page_config(
    page_title="DL Optimization Sandbox",
    page_icon="🧪",
    layout="wide",
)

# ──────────────────────────── Session State ────────────────────────────
if "single_result" not in st.session_state:
    st.session_state.single_result = None
if "compare_results" not in st.session_state:
    st.session_state.compare_results = {}
if "active_preset_key" not in st.session_state:
    st.session_state.active_preset_key = "adam_default"
if "x_test" not in st.session_state:
    st.session_state.x_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None

# ──────────────────────────── Data Loading ────────────────────────────
@st.cache_data
def get_data(subset_size: int, seed: int):
    (x_full, y_full), (x_test, y_test) = load_fashion_mnist()
    np.random.seed(seed)
    idx = np.random.choice(len(x_full), size=subset_size, replace=False)
    return x_full[idx], y_full[idx], x_test, y_test


# ──────────────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Dataset ──
    st.subheader("📂 Dataset")
    subset_size = st.slider("Training subset size", 100, 5000, 1000, 100)
    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()

    # ── Preset Selector ──
    st.subheader("📦 Preset Selection")
    preset_keys = list(PRESETS.keys())
    preset_display = {k: f"[{v.category.upper()}] {v.name}" for k, v in PRESETS.items()}

    selected_key = st.selectbox(
        "Optimization preset",
        options=preset_keys,
        format_func=lambda k: preset_display[k],
        index=preset_keys.index(st.session_state.active_preset_key),
        key="preset_selector",
    )
    st.session_state.active_preset_key = selected_key
    base_preset = PRESETS[selected_key]

    st.info(f"**{base_preset.name}**\n\n{base_preset.description}")

    st.divider()

    # ── Manual Override ──
    with st.expander("🔧 Manual Override", expanded=False):
        ov_optimizer = st.selectbox(
            "Optimizer", ["sgd", "adam", "rmsprop", "adagrad"],
            index=["sgd", "adam", "rmsprop", "adagrad"].index(base_preset.optimizer),
        )
        ov_lr = st.select_slider(
            "Learning rate",
            options=[5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
            value=min([5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                      key=lambda x: abs(x - base_preset.learning_rate)),
        )
        ov_momentum = st.slider("Momentum (SGD)", 0.0, 0.99,
                                float(base_preset.momentum), 0.01)
        ov_nesterov = st.checkbox("Nesterov", value=base_preset.nesterov)
        ov_init = st.selectbox(
            "Initialization",
            ["random_normal", "glorot_uniform", "he_normal", "he_uniform"],
            index=["random_normal", "glorot_uniform", "he_normal", "he_uniform"]
                  .index(base_preset.initializer),
        )
        ov_norm = st.selectbox(
            "Normalization", ["none", "batch", "layer"],
            index=["none", "batch", "layer"].index(base_preset.normalization),
        )
        ov_clip = st.number_input(
            "Gradient clip norm (0 = off)", min_value=0.0, max_value=20.0,
            value=float(base_preset.gradient_clip_norm or 0.0), step=0.5,
        )
        ov_schedule = st.selectbox(
            "LR Schedule", ["constant", "step_decay", "cosine"],
            index=["constant", "step_decay", "cosine"].index(base_preset.lr_schedule),
        )
        ov_reg = st.selectbox(
            "Regularization", ["none", "l2", "dropout"],
            index=["none", "l2", "dropout"].index(base_preset.regularization),
        )
        ov_l2 = st.slider("L2 lambda", 0.0001, 0.1, float(base_preset.l2_lambda), 0.001,
                           format="%.4f")
        ov_dropout = st.slider("Dropout rate", 0.1, 0.8, float(base_preset.dropout_rate), 0.05)
        st.markdown("**Architecture**")
        ov_n_layers = st.selectbox("Hidden layer count", [1, 2, 3],
                                   index=min(len(base_preset.hidden_units), 3) - 1)
        ov_units = []
        defaults = (base_preset.hidden_units + [64, 32])[:3]
        for i in range(ov_n_layers):
            u = st.slider(f"Layer {i+1} units", 32, 512, int(defaults[i]), 32)
            ov_units.append(u)
        st.markdown("**Training**")
        ov_epochs = st.slider("Epoch", 10, 200, int(base_preset.epochs), 10)
        ov_batch = st.selectbox("Batch size", [16, 32, 64, 128],
                                index=[16, 32, 64, 128].index(int(base_preset.batch_size))
                                if int(base_preset.batch_size) in [16, 32, 64, 128] else 1)
        ov_es = st.checkbox("Early Stopping", value=base_preset.early_stopping)

        override_active = True

    # Build active preset from overrides
    active_preset = TrainingPreset(
        name=base_preset.name,
        description=base_preset.description,
        category=base_preset.category,
        optimizer=ov_optimizer,
        learning_rate=float(ov_lr),
        momentum=float(ov_momentum),
        nesterov=ov_nesterov,
        beta_1=base_preset.beta_1,
        beta_2=base_preset.beta_2,
        rho=base_preset.rho,
        initializer=ov_init,
        normalization=ov_norm,
        gradient_clip_norm=float(ov_clip) if ov_clip > 0 else None,
        lr_schedule=ov_schedule,
        lr_decay_factor=base_preset.lr_decay_factor,
        lr_decay_epochs=base_preset.lr_decay_epochs,
        regularization=ov_reg,
        l2_lambda=float(ov_l2),
        dropout_rate=float(ov_dropout),
        epochs=ov_epochs,
        batch_size=ov_batch,
        early_stopping=ov_es,
        early_stopping_patience=base_preset.early_stopping_patience,
        hidden_units=ov_units,
    )

    st.divider()

    # ── Comparison ──
    st.subheader("📊 Comparison")
    compare_keys = st.multiselect(
        "Select presets (multi)",
        options=preset_keys,
        default=["vanilla_sgd", "adam_default"],
        format_func=lambda k: PRESETS[k].name,
    )

    st.divider()

    col_run, col_cmp = st.columns(2)
    with col_run:
        run_btn = st.button("🚀 Start Training", use_container_width=True, type="primary")
    with col_cmp:
        cmp_btn = st.button("🔄 Compare", use_container_width=True)


# ──────────────────────────── Main Area ───────────────────────────────
st.title("🧪 Deep Learning Optimization Sandbox")
st.markdown(
    "**Week 5** — Comparative analysis of optimization strategies on Fashion-MNIST. "
    "Optimizer · Init · Normalization · Gradient Clipping · LR Scheduling"
)

x_train, y_train, x_test, y_test = get_data(subset_size, int(seed))
st.session_state.x_test = x_test
st.session_state.y_test = y_test

with st.expander("📷 Sample Images", expanded=False):
    cols = st.columns(10)
    for i, col in enumerate(cols):
        with col:
            st.image(x_train[i].squeeze(), caption=CLASS_NAMES[y_train[i]], width=60)

st.info(
    f"Train: **{len(x_train)}** samples  |  Test: **{len(x_test)}** samples  |  "
    f"Classes: **10**  |  Active preset: **{active_preset.name}**"
)

# ──────────────────────────── Run Single ──────────────────────────────
if run_btn:
    st.session_state.single_result = None
    with st.spinner(f"Training: {active_preset.name} ({active_preset.epochs} epochs)…"):
        result = train_with_preset(active_preset, x_train, y_train, x_test, y_test)
    st.session_state.single_result = result
    if result["diverged"]:
        st.warning(f"⚠️ **{active_preset.name}** — NaN/Inf detected during training, diverged!")
    else:
        st.success(f"✅ **{active_preset.name}** complete — Test Acc: {result['test_acc']:.2%}")

# ──────────────────────────── Run Comparison ──────────────────────────
if cmp_btn:
    if not compare_keys:
        st.warning("Select at least one preset for comparison.")
    else:
        st.session_state.compare_results = {}
        progress_bar = st.progress(0, text="Preparing…")
        for i, key in enumerate(compare_keys):
            preset_i = copy.copy(PRESETS[key])
            preset_i.epochs = active_preset.epochs
            preset_i.batch_size = active_preset.batch_size
            progress_bar.progress(
                int(i / len(compare_keys) * 100),
                text=f"Training: {preset_i.name} ({i+1}/{len(compare_keys)})…",
            )
            res = train_with_preset(preset_i, x_train, y_train, x_test, y_test)
            st.session_state.compare_results[key] = res
        progress_bar.progress(100, text="Done!")
        st.success(f"✅ {len(compare_keys)} preset comparison complete.")

# ─────────────────────────── Display Results ──────────────────────────
tab_single, tab_compare, tab_model = st.tabs(
    ["📈 Training Curves", "📊 Comparison", "🔍 Model Details"]
)

# ── Tab 1: Single Preset Results ──────────────────────────────────────
with tab_single:
    r = st.session_state.single_result

    if r is None:
        st.markdown(
            """
            ### How to Use
            1. Select a **preset** from the sidebar
            2. Optionally open **Manual Override** to tweak individual parameters
            3. Click **🚀 Start Training**
            4. Inspect results in the metrics and charts below
            """
        )
    else:
        # Metric cards
        h = r["history"].history
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Test Accuracy", f"{r['test_acc']:.2%}")
        m2.metric("Test Loss", f"{r['test_loss']:.4f}")
        m3.metric("Train Time", f"{r['train_time']:.1f}s")
        m4.metric("Convergence Epoch", str(r["convergence_epoch"]))
        m5.metric("Overfitting Gap", f"{r['overfitting_gap']:+.4f}")

        # Extra info
        ci1, ci2, ci3 = st.columns(3)
        ci1.caption(f"**Optimizer:** {r['preset'].optimizer.upper()}")
        ci2.caption(f"**Init:** {r['preset'].initializer}")
        ci3.caption(f"**Norm:** {r['preset'].normalization} | **Clip:** {r['preset'].gradient_clip_norm}")

        if r["diverged"]:
            st.error("⚠️ Training ended with NaN/Inf. Try a lower LR or enable gradient clipping.")

        st.subheader("📉 Loss & Accuracy Curves")
        fig_curves = plot_preset_training_curves(r)
        st.pyplot(fig_curves)
        plt.close(fig_curves)

        st.subheader("📡 LR Schedule & Gradient Norm")
        fig_lr_grad = plot_lr_and_grad_norm(r)
        st.pyplot(fig_lr_grad)
        plt.close(fig_lr_grad)

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.caption(
                f"Train Acc (last epoch): **{h['accuracy'][-1]:.4f}** | "
                f"Val Acc: **{h['val_accuracy'][-1]:.4f}**"
            )
        with col_stat2:
            st.caption(
                f"Train Loss (last epoch): **{h['loss'][-1]:.4f}** | "
                f"Val Loss: **{h['val_loss'][-1]:.4f}**"
            )

        if r["gradient_norms"]:
            st.caption(
                f"Gradient Norm — first: **{r['gradient_norms'][0]:.4f}** | "
                f"last: **{r['gradient_norms'][-1]:.4f}** | "
                f"max: **{max(r['gradient_norms']):.4f}**"
            )

# ── Tab 2: Comparison ─────────────────────────────────────────────────
with tab_compare:
    cmp = st.session_state.compare_results

    if not cmp:
        st.info("Select presets from the sidebar and click **🔄 Compare** to run a comparison.")
    else:
        # Overlay curves
        st.subheader("📈 Overlay — Validation Loss & Accuracy")
        fig_ov = plot_overlay_comparison(cmp)
        st.pyplot(fig_ov)
        plt.close(fig_ov)

        # Bar charts row
        col_bar1, col_bar2 = st.columns(2)
        with col_bar1:
            st.subheader("🏆 Test Accuracy")
            fig_acc = plot_test_accuracy_bar_presets(cmp)
            st.pyplot(fig_acc)
            plt.close(fig_acc)
        with col_bar2:
            st.subheader("⚡ Convergence Speed")
            fig_conv = plot_convergence_speed_bar(cmp)
            st.pyplot(fig_conv)
            plt.close(fig_conv)

        col_bar3, col_bar4 = st.columns(2)
        with col_bar3:
            st.subheader("📏 Overfitting Gap")
            fig_gap = plot_overfitting_gap_bar(cmp)
            st.pyplot(fig_gap)
            plt.close(fig_gap)

        # Metrics table
        st.subheader("📋 Metrics Table")
        df = build_comparison_table(cmp)

        def _highlight(s):
            styles = []
            for v in s:
                try:
                    val = float(v)
                    if val == s.max():
                        styles.append("background-color: #d4edda; color: #155724; font-weight: bold")
                    elif val == s.min():
                        styles.append("background-color: #f8d7da; color: #721c24")
                    else:
                        styles.append("")
                except Exception:
                    styles.append("")
            return styles

        highlight_cols = ["Test Accuracy", "Best Val Loss", "Convergence Epoch",
                          "Training Time (s)", "Overfitting Gap"]
        styled = df.style
        for col in highlight_cols:
            if col in df.columns:
                styled = styled.apply(_highlight, subset=[col])
        st.dataframe(styled, use_container_width=True)

        # Divergence warnings
        for key, res in cmp.items():
            if res.get("diverged"):
                st.warning(f"⚠️ **{res['preset'].name}** diverged — results may be invalid.")

# ── Tab 3: Model Details ──────────────────────────────────────────────
with tab_model:
    r = st.session_state.single_result

    if r is None:
        st.info("Train a model first by clicking **🚀 Start Training**.")
    else:
        preset = r["preset"]
        st.subheader(f"🔢 Confusion Matrix — {preset.name}")
        cm, report = compute_confusion_matrix_preset(r, x_test, y_test)
        fig_cm = plot_confusion_matrix_preset(cm, preset.name)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        with st.expander("📋 Classification Report"):
            st.code(report)

        st.subheader("🏗️ Model Summary")
        summary_lines = []
        r["model"].summary(print_fn=lambda x: summary_lines.append(x))
        st.code("\n".join(summary_lines))

        st.subheader("📐 Active Configuration")
        config_data = {
            "Parameter": [
                "Optimizer", "Learning Rate", "Momentum", "Nesterov",
                "Initializer", "Normalization", "Gradient Clip",
                "LR Schedule", "Regularization", "Dropout Rate", "L2 Lambda",
                "Epochs", "Batch Size", "Early Stopping", "Hidden Units",
                "Param Count",
            ],
            "Value": [
                preset.optimizer.upper(), preset.learning_rate, preset.momentum,
                preset.nesterov, preset.initializer, preset.normalization,
                preset.gradient_clip_norm or "None",
                preset.lr_schedule, preset.regularization,
                preset.dropout_rate if preset.regularization == "dropout" else "—",
                preset.l2_lambda if preset.regularization == "l2" else "—",
                preset.epochs, preset.batch_size, preset.early_stopping,
                str(preset.hidden_units), f"{r['param_count']:,}",
            ],
        }
        st.dataframe(pd.DataFrame(config_data).set_index("Parameter"), use_container_width=True)

        st.subheader("📚 Educational Notes")
        HELP = {
            "sgd": "🔵 **SGD:** The simplest optimizer. Takes a step of size ε opposite the gradient. Very sensitive to learning rate.",
            "adam": "🟢 **Adam:** RMSProp + Momentum + bias correction. Industry standard. Default: lr=0.001, β₁=0.9, β₂=0.999.",
            "rmsprop": "🟡 **RMSProp:** Fixes AdaGrad's flaw using EWMA to forget old gradients. Adapts to non-stationary loss surfaces.",
            "adagrad": "🟠 **AdaGrad:** Per-parameter LR. Fatal flaw: accumulator only grows → LR→0.",
        }
        HELP_INIT = {
            "random_normal": "🔴 **Random Normal:** Breaks symmetry but no variance matching. Too small→vanishing, too large→exploding.",
            "glorot_uniform": "🔵 **Xavier/Glorot:** Var(output)=Var(input). For Sigmoid/Tanh. W ~ U(±√(6/(fan_in+fan_out))).",
            "he_normal": "🟢 **He/Kaiming:** Xavier×2. ReLU zeroes out the negative half → He compensates by multiplying by 2.",
            "he_uniform": "🟢 **He Uniform:** Uniform distribution variant of He Normal.",
        }
        HELP_NORM = {
            "none": "⚪ **No Normalization:** Baseline. Internal covariate shift persists.",
            "batch": "🔵 **BatchNorm:** Normalizes over the mini-batch → γ scale, β shift. Requires batch_size≥16.",
            "layer": "🟡 **LayerNorm:** Independent of batch size. Gold standard for Transformers/NLP.",
        }
        st.markdown(HELP.get(preset.optimizer, ""))
        st.markdown(HELP_INIT.get(preset.initializer, ""))
        st.markdown(HELP_NORM.get(preset.normalization, ""))
        if preset.gradient_clip_norm:
            st.markdown(
                f"✂️ **Gradient Clipping (norm={preset.gradient_clip_norm}):** "
                "Goodfellow Sec 8.2.4 — protection against cliff structures: "
                "||g||>threshold → g ← threshold×g/||g||"
            )
        if preset.lr_schedule != "constant":
            sched_help = {
                "step_decay": f"📉 **Step Decay:** εₖ = ε₀ × {preset.lr_decay_factor}^(floor(k/{preset.lr_decay_epochs}))",
                "cosine": "🌀 **Cosine Annealing:** εₖ = ε₀ × 0.5 × (1 + cos(πk/T)) — smooth decay towards end of training.",
            }
            st.markdown(sched_help.get(preset.lr_schedule, ""))
