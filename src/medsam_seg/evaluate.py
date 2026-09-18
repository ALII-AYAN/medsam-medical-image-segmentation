"""Evaluate a saved model on the held-out split and write a report."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from medsam_seg import data as data_module
from medsam_seg import metrics
from medsam_seg.config import AppConfig


def load_model(model_path: Path):
    """Load a saved Keras model from .keras, .h5 or a saved_model directory."""
    from tensorflow.keras.models import load_model

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    return load_model(model_path, compile=False)


def evaluate(config: AppConfig, model_path: Path | None = None) -> dict:
    """Score a model on the validation split and save metrics + plots."""
    model_path = Path(model_path or config.inference.model_path)
    model = load_model(model_path)

    pairs = data_module.discover_pairs(
        config.data.image_dir, config.data.mask_dir, config.data.mask_suffix
    )
    _, test_pairs = data_module.split_pairs(pairs, config.data.val_split, config.data.seed)

    X_test, y_test = data_module.load_split(
        test_pairs, tuple(config.data.image_size), config.data.mask_threshold
    )

    start = time.time()
    y_pred = model.predict(X_test, verbose=0)
    elapsed = time.time() - start
    y_pred_binary = (y_pred > config.eval.threshold).astype(np.float32)

    scores = metrics.compute_all(y_test, y_pred_binary, config.eval.threshold)
    scores["n_test"] = len(test_pairs)
    scores["total_time_s"] = round(elapsed, 3)
    scores["mean_inference_s"] = round(elapsed / max(len(test_pairs), 1), 4)
    scores["class_0_accuracy"] = metrics.class_accuracy(
        y_test, y_pred_binary, 0, config.eval.threshold
    )
    scores["class_1_accuracy"] = metrics.class_accuracy(
        y_test, y_pred_binary, 1, config.eval.threshold
    )
    scores["class_distribution"] = {
        str(k): round(v, 2)
        for k, v in metrics.class_distribution(y_test, config.eval.threshold).items()
    }

    out_dir = Path(config.eval.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    from medsam_seg.train import plot_samples

    plot_samples(X_test, y_test, y_pred_binary, out_dir / "evaluation_samples.png",
                 config.eval.num_samples)

    print(f"Evaluated {len(test_pairs)} images with {model_path.name}")
    for key in ("accuracy", "dice", "iou", "f1"):
        print(f"  {key:<10} {scores[key]:.4f}")
    print(f"  {'inference':<10} {scores['mean_inference_s']:.4f} s/image")
    print(f"Report written to {out_dir / 'evaluation.json'}")
    return scores


def per_image_scores(config: AppConfig, model_path: Path | None = None) -> list[dict]:
    """Per-image metrics, useful for spotting which cases fail."""
    model_path = Path(model_path or config.inference.model_path)
    model = load_model(model_path)

    pairs = data_module.discover_pairs(
        config.data.image_dir, config.data.mask_dir, config.data.mask_suffix
    )
    _, test_pairs = data_module.split_pairs(pairs, config.data.val_split, config.data.seed)

    rows = []
    for pair in test_pairs:
        image, mask = data_module.load_pair(
            pair, tuple(config.data.image_size), config.data.mask_threshold
        )
        prediction = model.predict(image[np.newaxis, ...], verbose=0)[0]
        rows.append({
            "name": pair.name,
            **metrics.compute_all(mask, prediction, config.eval.threshold),
            "foreground_percent": metrics.foreground_fraction(prediction,
                                                              config.eval.threshold),
        })
    rows.sort(key=lambda row: row["dice"])
    return rows
