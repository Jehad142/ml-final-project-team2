# train_utils.py
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_num, X_cat_dict, counts, graphs, y in loader:
        optimizer.zero_grad()
        logits = model(X_num, X_cat_dict, counts, graphs)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_one_epoch(model, loader, threshold, device):
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

    return acc, prec, rec, f1, auc

def save_checkpoint(model, optimizer, history, epoch, config, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config
    }, path)

'''

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    history = checkpoint.get("history", {})
    start_epoch = checkpoint.get("epoch", 0)
    return model, optimizer, history, start_epoch
'''

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    history = checkpoint.get("history", {})
    start_epoch = checkpoint.get("epoch", 0)
    return model, optimizer, history, start_epoch
