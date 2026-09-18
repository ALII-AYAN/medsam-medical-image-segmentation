"""Segmentation metrics, computed with NumPy so they work without TensorFlow.

Every function takes flat or shaped arrays of the same shape and returns a
plain float in ``[0, 1]``. Ground-truth and predicted masks are expected to be
binary (0/1) or already thresholded.
"""

from __future__ import annotations

import numpy as np


def _as_binary(array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(array, dtype=np.float32) > threshold).astype(np.float32)


def _flat(y_true: np.ndarray, y_pred: np.ndarray, threshold: float):
    true_flat = _as_binary(y_true, threshold).ravel()
    pred_flat = _as_binary(y_pred, threshold).ravel()
    if true_flat.shape != pred_flat.shape:
        raise ValueError(f"shape mismatch: {true_flat.shape} vs {pred_flat.shape}")
    return true_flat, pred_flat


def dice_score(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    """Sorensen-Dice coefficient: 2|X&Y| / (|X| + |Y|)."""
    t, p = _flat(y_true, y_pred, threshold)
    intersection = float(np.sum(t * p))
    denominator = float(np.sum(t) + np.sum(p))
    if denominator == 0.0:
        # Both empty counts as a perfect match, not a division error.
        return 1.0
    return (2.0 * intersection + smooth) / (denominator + smooth)


def iou_score(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    """Jaccard index: |X&Y| / |X|Y|."""
    t, p = _flat(y_true, y_pred, threshold)
    intersection = float(np.sum(t * p))
    union = float(np.sum(t) + np.sum(p) - intersection)
    if union == 0.0:
        return 1.0
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of pixels classified correctly (background included)."""
    t, p = _flat(y_true, y_pred, threshold)
    if t.size == 0:
        return 0.0
    return float(np.mean(t == p))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    t, p = _flat(y_true, y_pred, threshold)
    predicted_positive = float(np.sum(p))
    if predicted_positive == 0.0:
        return 0.0
    return float(np.sum(t * p)) / predicted_positive


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    t, p = _flat(y_true, y_pred, threshold)
    actual_positive = float(np.sum(t))
    if actual_positive == 0.0:
        return 0.0
    return float(np.sum(t * p)) / actual_positive


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    precision = precision_score(y_true, y_pred, threshold)
    recall = recall_score(y_true, y_pred, threshold)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, class_value: int, threshold: float = 0.5
) -> float:
    """Accuracy restricted to the pixels of one class.

    Overall accuracy is dominated by background whenever the mask is
    imbalanced, so the per-class numbers are the ones worth reading.
    """
    t, p = _flat(y_true, y_pred, threshold)
    mask = t == class_value
    if not np.any(mask):
        return float("nan")
    return float(np.mean(t[mask] == p[mask]))


def foreground_fraction(mask: np.ndarray, threshold: float = 0.5) -> float:
    """Percentage of the image predicted as foreground."""
    flat = _as_binary(mask, threshold).ravel()
    if flat.size == 0:
        return 0.0
    return float(np.mean(flat)) * 100.0


def class_distribution(mask: np.ndarray, threshold: float = 0.5) -> dict[int, float]:
    """Percentage of pixels belonging to each class."""
    flat = _as_binary(mask, threshold).ravel()
    if flat.size == 0:
        return {}
    total = flat.size
    return {
        0: float(np.sum(flat == 0)) / total * 100.0,
        1: float(np.sum(flat == 1)) / total * 100.0,
    }


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Every headline metric in one call."""
    return {
        "accuracy": pixel_accuracy(y_true, y_pred, threshold),
        "dice": dice_score(y_true, y_pred, threshold),
        "iou": iou_score(y_true, y_pred, threshold),
        "f1": f1_score(y_true, y_pred, threshold),
        "precision": precision_score(y_true, y_pred, threshold),
        "recall": recall_score(y_true, y_pred, threshold),
    }
