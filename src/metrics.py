"""Evaluation metrics using only numpy (no scikit-learn needed).

Macro-F1 is the average F1 over classes that appear in the test set. Unlike
plain accuracy it counts a rare class as much as a common one, so it shows
whether the model is only good at the big classes.
"""
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """cm[i, j] = number of images of true class i predicted as class j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_metrics(cm: np.ndarray):
    """Returns precision, recall, f1, support (one value per class)."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)          # how many true images of each class
    predicted = cm.sum(axis=0)        # how many times each class was predicted
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def summarize(cm: np.ndarray) -> dict:
    _, _, f1, support = per_class_metrics(cm)
    present = support > 0
    total = cm.sum()
    return {
        "n_images": int(total),
        "n_classes": int(present.sum()),
        "accuracy": float(np.diag(cm).sum() / total),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
    }


def top_confusions(cm: np.ndarray, class_names, k: int = 15):
    """Most frequent mistakes as (true class, predicted class, count)."""
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    order = np.argsort(errors, axis=None)[::-1][:k]
    rows = []
    for flat in order:
        i, j = np.unravel_index(flat, errors.shape)
        if errors[i, j] > 0:
            rows.append((class_names[i], class_names[j], int(errors[i, j])))
    return rows


def coverage_table(confidence, correct, thresholds):
    """For each confidence threshold: share of images kept and accuracy on them."""
    confidence = np.asarray(confidence)
    correct = np.asarray(correct, dtype=bool)
    rows = []
    for t in thresholds:
        keep = confidence >= t
        acc = float(correct[keep].mean()) if keep.any() else float("nan")
        rows.append((float(t), float(keep.mean()), acc))
    return rows