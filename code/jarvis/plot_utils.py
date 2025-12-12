# plot_utils.py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

def plot_training_history(history, base_name, store_dir):
    fig_filename = f"{base_name}_training_history.png"
    fig_path = os.path.join(store_dir, fig_filename)

    plt.figure(figsize=(10, 6))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.plot(history["val_f1"], label="Val F1")
    plt.plot(history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Training History")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    caption = "Figure: Training history showing loss and validation metrics across epochs."
    return fig_path, caption


def plot_precision_recall(y_true, y_prob, base_name, store_dir):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)

    fig_filename = f"{base_name}_precision_recall_curve.png"
    fig_path = os.path.join(store_dir, fig_filename)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label=f"PR curve (AP={avg_prec:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Validation)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    caption = f"Figure: Precision–Recall curve with average precision (AP={avg_prec:.3f})."
    return fig_path, caption


def plot_f1_vs_threshold(y_true, y_prob, base_name, store_dir):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]

    best_idx = np.argmax(f1_scores)
    best_thresh, best_f1 = thresholds[best_idx], f1_scores[best_idx]

    fig_filename = f"{base_name}_f1_vs_threshold.png"
    fig_path = os.path.join(store_dir, fig_filename)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker="o", markersize=3)
    plt.axvline(best_thresh, color="red", linestyle="--",
                label=f"Best threshold={best_thresh:.2f}, F1={best_f1:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold (Validation)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    caption = f"Figure: F1 score vs threshold, optimal threshold={best_thresh:.2f}, F1={best_f1:.3f}."
    return fig_path, caption


def plot_roc(y_true, y_prob, base_name, store_dir):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig_filename = f"{base_name}_roc_curve.png"
    fig_path = os.path.join(store_dir, fig_filename)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker=".", label=f"ROC curve (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Validation)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    caption = f"Figure: ROC curve with area under the curve (AUC={auc_score:.3f})."
    return fig_path, caption


def plot_validation_metrics(history, base_name, store_dir):
    num_epochs = len(history["val_acc"])
    fig_filename = f"{base_name}_validation_metrics.png"
    fig_path = os.path.join(store_dir, fig_filename)

    plt.figure(figsize=(12, 6))
    plt.plot(history["val_acc"], label="Accuracy")
    plt.plot(history["val_prec"], label="Precision")
    plt.plot(history["val_rec"], label="Recall")
    plt.plot(history["val_f1"], label="F1")
    plt.plot(history["val_auc"], label="AUC")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"Validation Metrics Across {num_epochs} Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    caption = (
        f"Figure: Validation metrics across {num_epochs} epochs for the multimodal GNN model. "
        "Curves show accuracy, precision, recall, F1, and AUC, providing a comprehensive view "
        "of model performance trends during training."
    )
    return fig_path, caption
