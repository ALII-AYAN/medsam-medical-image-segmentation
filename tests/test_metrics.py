"""Metric maths, checked against hand-computed values."""

import numpy as np
import pytest

from medsam_seg.metrics import (
    class_accuracy,
    class_distribution,
    compute_all,
    dice_score,
    f1_score,
    foreground_fraction,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
)


def test_dice_identical_is_one():
    mask = np.ones((8, 8), dtype=np.float32)
    assert dice_score(mask, mask) == pytest.approx(1.0)


def test_dice_disjoint_is_zero():
    truth = np.ones((8, 8), dtype=np.float32)
    prediction = np.zeros((8, 8), dtype=np.float32)
    assert dice_score(truth, prediction, smooth=0.0) == pytest.approx(0.0)


def test_dice_partial_overlap(flat_masks):
    truth, prediction = flat_masks
    # |X&Y| = 8, |X| = 8, |Y| = 12  ->  2*8/(8+12) = 0.8
    assert dice_score(truth, prediction, smooth=0.0) == pytest.approx(2 * 8 / 20)


def test_smoothing_only_guards_against_zero(flat_masks):
    """smooth exists to avoid 0/0; on real masks it is numerically irrelevant."""
    truth, prediction = flat_masks
    exact = dice_score(truth, prediction, smooth=0.0)
    smoothed = dice_score(truth, prediction, smooth=1.0)
    assert smoothed > exact
    # 65 536 pixels is one 256x256 mask: the difference vanishes.
    big_truth = np.repeat(np.repeat(truth, 128, axis=0), 128, axis=1)
    big_pred = np.repeat(np.repeat(prediction, 128, axis=0), 128, axis=1)
    assert dice_score(big_truth, big_pred, smooth=0.0) == pytest.approx(
        dice_score(big_truth, big_pred, smooth=1.0), abs=1e-5
    )


def test_iou_partial_overlap(flat_masks):
    truth, prediction = flat_masks
    # intersection 8, union 12  ->  8/12
    assert iou_score(truth, prediction, smooth=0.0) == pytest.approx(8 / 12)


def test_dice_and_iou_relation(flat_masks):
    """Dice = 2*IoU/(1+IoU) must hold, or one of them is wrong."""
    truth, prediction = flat_masks
    iou = iou_score(truth, prediction, smooth=0.0)
    assert dice_score(truth, prediction, smooth=0.0) == pytest.approx(2 * iou / (1 + iou))


def test_both_empty_counts_as_perfect():
    empty = np.zeros((4, 4), dtype=np.float32)
    assert dice_score(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0


def test_pixel_accuracy(flat_masks):
    truth, prediction = flat_masks
    # 16 pixels, one column of 4 wrong -> 12/16
    assert pixel_accuracy(truth, prediction) == pytest.approx(0.75)


def test_accuracy_is_dominated_by_background():
    """The reason per-class accuracy exists: 90% background is 90% 'accurate'."""
    truth = np.zeros((10, 10), dtype=np.float32)
    truth[0, 0] = 1.0  # a single foreground pixel
    prediction = np.zeros((10, 10), dtype=np.float32)
    assert pixel_accuracy(truth, prediction) == pytest.approx(0.99)
    assert dice_score(truth, prediction, smooth=0.0) == pytest.approx(0.0)


def test_precision_and_recall(flat_masks):
    truth, prediction = flat_masks
    # 8 true positives, 4 false positives, 0 false negatives
    assert precision_score(truth, prediction) == pytest.approx(8 / 12)
    assert recall_score(truth, prediction) == pytest.approx(1.0)


def test_f1_is_the_harmonic_mean(flat_masks):
    truth, prediction = flat_masks
    precision = precision_score(truth, prediction)
    recall = recall_score(truth, prediction)
    assert f1_score(truth, prediction) == pytest.approx(
        2 * precision * recall / (precision + recall)
    )


def test_per_class_accuracy(flat_masks):
    truth, prediction = flat_masks
    # 8 background pixels, 4 of them predicted foreground -> 0.5
    assert class_accuracy(truth, prediction, 0) == pytest.approx(0.5)
    assert class_accuracy(truth, prediction, 1) == pytest.approx(1.0)  # all fg found


def test_per_class_accuracy_of_absent_class_is_nan():
    empty = np.zeros((4, 4), dtype=np.float32)
    assert np.isnan(class_accuracy(empty, empty, 1))


def test_foreground_fraction():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[:5, :] = 1.0
    assert foreground_fraction(mask) == pytest.approx(50.0)


def test_class_distribution_sums_to_100():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[:2, :] = 1.0
    distribution = class_distribution(mask)
    assert sum(distribution.values()) == pytest.approx(100.0)
    assert distribution[1] == pytest.approx(20.0)


def test_threshold_is_respected():
    truth = np.array([[1.0, 0.0]])
    soft = np.array([[0.9, 0.9]])
    assert dice_score(truth, soft, threshold=0.5, smooth=0.0) == pytest.approx(2 / 3)
    assert dice_score(truth, soft, threshold=0.95, smooth=0.0) == pytest.approx(0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        dice_score(np.zeros((4, 4)), np.zeros((8, 8)))


def test_compute_all_returns_every_metric(flat_masks):
    truth, prediction = flat_masks
    scores = compute_all(truth, prediction)
    assert set(scores) == {"accuracy", "dice", "iou", "f1", "precision", "recall"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
