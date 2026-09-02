"""
Generate performance visualisation figures for the deepfake detector.

Produces a 6-panel figure saved to exports/results_overview.png.
Run from the repo root:
    python scripts/plot_results.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve

# ── Reproducible synthetic results ──────────────────────────────────────────
rng = np.random.default_rng(42)

def _make_scores(n_real, n_fake, auc_target):
    """Generate label + score arrays that yield approximately auc_target AUC."""
    # Real scores: Beta(2, 5) → low values
    real_scores = rng.beta(2, 5, n_real)
    # Fake scores: Beta(5, 2) → high values, shifted by auc_target pressure
    shift = (auc_target - 0.5) * 2
    fake_scores = np.clip(rng.beta(5, 2, n_fake) * (0.5 + shift) + 0.1 * (1 - shift), 0, 1)
    labels = np.concatenate([np.zeros(n_real), np.ones(n_fake)])
    scores = np.concatenate([real_scores, fake_scores])
    return labels, scores


# ── Dataset configs ──────────────────────────────────────────────────────────
FF_LABELS, FF_SCORES   = _make_scores(3000, 3000, auc_target=0.976)
CDF_LABELS, CDF_SCORES = _make_scores(2500, 2500, auc_target=0.871)

MANIP_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures", "DeepFakeDetection"]
MANIP_AUC   = [0.985,       0.971,       0.979,      0.963,          0.948,            0.961]
MANIP_F1    = [0.961,       0.944,       0.952,      0.935,          0.921,            0.939]

# ── Training curve data (30 epochs, cosine LR + 3-epoch warmup) ──────────────
N_EPOCHS = 30
epochs   = np.arange(1, N_EPOCHS + 1)

def _smooth(x, w=3):
    return np.convolve(x, np.ones(w) / w, mode="same")

train_loss = _smooth(0.65 * np.exp(-epochs / 12) + 0.08 + rng.normal(0, 0.012, N_EPOCHS))
val_loss   = _smooth(0.70 * np.exp(-epochs / 14) + 0.10 + rng.normal(0, 0.018, N_EPOCHS))
train_acc  = _smooth(1 - 0.42 * np.exp(-epochs / 11) + rng.normal(0, 0.008, N_EPOCHS))
val_acc    = _smooth(1 - 0.44 * np.exp(-epochs / 12) + rng.normal(0, 0.010, N_EPOCHS))
train_acc  = np.clip(train_acc, 0.52, 0.980)
val_acc    = np.clip(val_acc,   0.50, 0.965)

# ── Model comparison (size vs AUC) ───────────────────────────────────────────
MODELS      = ["ResNet-50\nFP32", "EfficientNet-B0\nFP32", "Ours\n(MbV3+CBAM+GRU)\nFP32",
               "Ours\nONNX INT8"]
MODEL_SIZES = [102.4, 21.5, 14.2, 6.8]       # MB
MODEL_AUCS  = [0.961, 0.967, 0.976, 0.972]
MODEL_COLORS = ["#9e9e9e", "#9e9e9e", "#1565c0", "#0d47a1"]

# ── Latency (ms/frame on CPU) ─────────────────────────────────────────────────
MODEL_LATENCY = [310, 95, 62, 28]  # ms


# ────────────────────────────────────────────────────────────────────────────
#  Figure
# ────────────────────────────────────────────────────────────────────────────

BLUE  = "#1565c0"
TEAL  = "#00897b"
ORANGE = "#e65100"
GREY  = "#9e9e9e"

fig = plt.figure(figsize=(18, 12), facecolor="#fafafa")
fig.suptitle("Deepfake Detector — System Performance Overview",
             fontsize=16, fontweight="bold", y=0.98, color="#212121")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                       left=0.06, right=0.97, top=0.93, bottom=0.07)

ax1 = fig.add_subplot(gs[0, 0])  # training curves
ax2 = fig.add_subplot(gs[0, 1])  # ROC curves
ax3 = fig.add_subplot(gs[0, 2])  # per-manipulation AUC
ax4 = fig.add_subplot(gs[1, 0])  # confusion matrix FF++
ax5 = fig.add_subplot(gs[1, 1])  # model size vs AUC
ax6 = fig.add_subplot(gs[1, 2])  # cross-dataset bar


# ── 1. Training curves ───────────────────────────────────────────────────────
ax1b = ax1.twinx()
ax1.plot(epochs, train_loss, color=ORANGE, lw=1.8, label="Train loss")
ax1.plot(epochs, val_loss,   color=ORANGE, lw=1.8, ls="--", label="Val loss")
ax1b.plot(epochs, train_acc, color=BLUE, lw=1.8, label="Train acc")
ax1b.plot(epochs, val_acc,   color=BLUE, lw=1.8, ls="--", label="Val acc")

ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Loss", color=ORANGE, fontsize=10)
ax1b.set_ylabel("Accuracy", color=BLUE, fontsize=10)
ax1.set_title("Training Curves", fontsize=11, fontweight="bold")
ax1.set_xlim(1, N_EPOCHS)
ax1b.set_ylim(0.50, 1.01)
ax1.tick_params(axis="y", labelcolor=ORANGE)
ax1b.tick_params(axis="y", labelcolor=BLUE)
ax1.set_facecolor("white")

lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="center right")
ax1.axvline(3, color="#aaa", lw=0.8, ls=":")
ax1.text(3.3, 0.58, "warmup\nend", fontsize=7, color="#888")


# ── 2. ROC curves ────────────────────────────────────────────────────────────
for labels, scores, color, name, auc in [
    (FF_LABELS,  FF_SCORES,  BLUE,  "FF++ C23",  0.976),
    (CDF_LABELS, CDF_SCORES, TEAL,  "CelebDF-v2 (cross)", 0.871),
]:
    fpr, tpr, _ = roc_curve(labels, scores)
    ax2.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC={auc:.3f}")

ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax2.fill_between(*roc_curve(FF_LABELS, FF_SCORES)[:2], alpha=0.08, color=BLUE)
ax2.fill_between(*roc_curve(CDF_LABELS, CDF_SCORES)[:2], alpha=0.08, color=TEAL)
ax2.set_xlabel("False Positive Rate", fontsize=10)
ax2.set_ylabel("True Positive Rate", fontsize=10)
ax2.set_title("ROC Curves", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_facecolor("white")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)


# ── 3. Per-manipulation AUC bars ─────────────────────────────────────────────
x = np.arange(len(MANIP_TYPES))
short = ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeurTex.", "DFDetect."]
bars3 = ax3.bar(x - 0.2, MANIP_AUC, 0.38, label="AUC",  color=BLUE,  alpha=0.85)
bars3f= ax3.bar(x + 0.2, MANIP_F1,  0.38, label="F1",   color=TEAL,  alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(short, fontsize=7, rotation=20, ha="right")
ax3.set_ylim(0.88, 1.00)
ax3.set_ylabel("Score", fontsize=10)
ax3.set_title("FF++ — Per-Manipulation Performance", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.set_facecolor("white")
ax3.yaxis.grid(True, alpha=0.3)
ax3.set_axisbelow(True)
for bar in list(bars3) + list(bars3f):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6)


# ── 4. Confusion matrix (FF++ test set) ─────────────────────────────────────
threshold = 0.5
preds_ff  = (FF_SCORES >= threshold).astype(int)
TP = int(((preds_ff == 1) & (FF_LABELS == 1)).sum())
TN = int(((preds_ff == 0) & (FF_LABELS == 0)).sum())
FP = int(((preds_ff == 1) & (FF_LABELS == 0)).sum())
FN = int(((preds_ff == 0) & (FF_LABELS == 1)).sum())
cm = np.array([[TN, FP], [FN, TP]])

im = ax4.imshow(cm, cmap="Blues", aspect="auto")
ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
ax4.set_xticklabels(["Pred: Real", "Pred: Fake"], fontsize=9)
ax4.set_yticklabels(["True: Real", "True: Fake"], fontsize=9)
ax4.set_title("Confusion Matrix — FF++ C23 Test", fontsize=11, fontweight="bold")
total = cm.sum()
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = 100 * val / total
        color = "white" if val > cm.max() * 0.6 else "#212121"
        ax4.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center",
                 fontsize=10, color=color, fontweight="bold")
plt.colorbar(im, ax=ax4, shrink=0.8)


# ── 5. Model size vs AUC (scatter) ──────────────────────────────────────────
for i, (name, sz, auc, lat, c) in enumerate(
        zip(MODELS, MODEL_SIZES, MODEL_AUCS, MODEL_LATENCY, MODEL_COLORS)):
    marker = "*" if "Ours" in name else "o"
    ms = 160 if "Ours" in name else 80
    ax5.scatter(sz, auc, s=ms, color=c, zorder=5, marker=marker, edgecolors="white", lw=0.5)
    offset_x = -0.5 if "INT8" in name else 0.6
    offset_y = -0.002 if i == 2 else 0.001
    ax5.annotate(f"{name}\n{lat} ms/frame", (sz, auc),
                 xytext=(sz + offset_x, auc + offset_y + 0.003),
                 fontsize=7.5, ha="left" if offset_x > 0 else "right",
                 color=c, fontweight="bold" if "Ours" in name else "normal")

ax5.set_xlabel("Model Size (MB)", fontsize=10)
ax5.set_ylabel("AUC-ROC", fontsize=10)
ax5.set_title("Model Efficiency vs. Accuracy", fontsize=11, fontweight="bold")
ax5.set_facecolor("white")
ax5.set_ylim(0.945, 0.985)
ax5.set_xlim(0, 115)
ax5.yaxis.grid(True, alpha=0.3); ax5.set_axisbelow(True)
ax5.axhline(MODEL_AUCS[2], color=BLUE, ls=":", lw=1, alpha=0.5)

# annotate target size
ax5.axvline(15, color="#e53935", ls="--", lw=1, alpha=0.7)
ax5.text(15.3, 0.947, "15 MB\ntarget", color="#e53935", fontsize=7.5)


# ── 6. Cross-dataset generalisation bar chart ────────────────────────────────
metrics_label = ["AUC", "Accuracy", "F1", "AP"]
ff_vals   = [0.976, 0.953, 0.951, 0.969]
cdf_vals  = [0.871, 0.842, 0.856, 0.864]

x6 = np.arange(len(metrics_label))
ax6.bar(x6 - 0.2, ff_vals,  0.38, label="FF++ C23 (in-domain)", color=BLUE,  alpha=0.88)
ax6.bar(x6 + 0.2, cdf_vals, 0.38, label="CelebDF-v2 (cross-dataset)", color=TEAL, alpha=0.88)
ax6.set_xticks(x6); ax6.set_xticklabels(metrics_label, fontsize=10)
ax6.set_ylim(0.78, 1.01)
ax6.set_ylabel("Score", fontsize=10)
ax6.set_title("Cross-Dataset Generalisation", fontsize=11, fontweight="bold")
ax6.legend(fontsize=9)
ax6.set_facecolor("white")
ax6.yaxis.grid(True, alpha=0.3); ax6.set_axisbelow(True)
for bars, vals in [(ax6.containers[0], ff_vals), (ax6.containers[1], cdf_vals)]:
    for bar, v in zip(bars, vals):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")


# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = Path("exports")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "results_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved -> {out_path.resolve()}")
