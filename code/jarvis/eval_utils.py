# eval_utils.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

def evaluate(model, loader, threshold=0.5, title="Evaluation", base_name=None, store_dir=None):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for X_num, X_cat_dict, counts, graphs, y in loader:
            logits = model(X_num, X_cat_dict, counts, graphs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            y_prob.extend(probs.tolist())
            y_true.extend(y.cpu().numpy().ravel().tolist())

    y_pred = (np.array(y_prob) >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    print(f"{title} -- Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{title} (Threshold={threshold})")

    fig_path, caption = None, None
    if base_name and store_dir:
        fig_filename = f"{base_name}_confusion_matrix.png"
        fig_path = os.path.join(store_dir, fig_filename)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix figure saved to: {fig_path}")
        caption = (
            f"Figure: Confusion matrix for the multimodal GNN model ({title}) at threshold {threshold:.2f}. "
            "The heatmap shows counts of true negatives, false positives, false negatives, and true positives."
        )
        print(caption)

    plt.show()
    return acc, prec, rec, f1, auc, fig_path, caption
