"""Training loop for the segmentation models.

Run directly or through the CLI:

    python -m medsam_seg train
    python -m medsam_seg train --arch unet --epochs 5 --config configs/default.yaml
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from medsam_seg import data as data_module
from medsam_seg import metrics
from medsam_seg.config import AppConfig


def dice_loss(y_true, y_pred, smooth: float = 1.0):
    """1 - Dice, so it can be minimised alongside cross-entropy."""
    import tensorflow as tf

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def weighted_bce(y_true, y_pred, pos_weight: float = 1.0):
    """Binary cross-entropy with a weight on the positive (foreground) class.

    Background usually dominates medical masks; ``pos_weight`` above 1 makes a
    missed foreground pixel cost more than a false alarm.
    """
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    weight = y_true * pos_weight + (1.0 - y_true)
    return tf.reduce_mean(bce * weight)


def build_loss(name: str, pos_weight: float = 1.0):
    """Pick the loss function described by the config."""
    if name == "bce":
        return lambda y_true, y_pred: weighted_bce(y_true, y_pred, pos_weight)
    if name == "dice":
        return dice_loss
    if name == "bce_dice":

        def combined(y_true, y_pred):
            return weighted_bce(y_true, y_pred, pos_weight) + dice_loss(y_true, y_pred)

        return combined
    raise ValueError(f"unknown loss '{name}'")


class EpochLogger:
    """Keras callback printing one compact line per epoch."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs
        self.epoch_times: list[float] = []

    def on_train_begin(self, logs=None):
        print("\n" + "=" * 60)
        print(f"TRAINING - {self.total_epochs} epochs")
        print("=" * 60)

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()
        print(f"\nEpoch {epoch + 1}/{self.total_epochs}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self._start
        self.epoch_times.append(elapsed)
        print(
            f"   loss {logs.get('loss', 0):.4f} | acc {logs.get('accuracy', 0):.4f}"
            f" | val_loss {logs.get('val_loss', 0):.4f}"
            f" | val_acc {logs.get('val_accuracy', 0):.4f}"
            f"  ({elapsed:.1f}s)"
        )

    def on_train_end(self, logs=None):
        print("\n" + "=" * 60)
        print("TRAINING FINISHED")
        print("=" * 60)


def plot_history(history, destination: Path) -> Path:
    """Save the accuracy and loss curves instead of blocking on plt.show()."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history.history["accuracy"], "b-o", label="Train")
    ax1.plot(epochs, history.history["val_accuracy"], "r-o", label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.history["loss"], "b-o", label="Train")
    ax2.plot(epochs, history.history["val_loss"], "r-o", label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=130)
    plt.close(fig)
    return destination


def plot_samples(images, masks_true, masks_pred, destination: Path, num_samples: int = 3) -> Path:
    """Side-by-side input / ground truth / prediction / overlay grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(num_samples, len(images))
    fig, axes = plt.subplots(count, 4, figsize=(16, 4 * count))
    if count == 1:
        axes = np.asarray(axes).reshape(1, 4)

    for i in range(count):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(masks_true[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(masks_pred[i].squeeze(), cmap="gray")
        axes[i, 2].set_title("Prediction")

        # Red channel marks the predicted region.
        overlay = images[i].copy()
        region = masks_pred[i].squeeze() > 0.5
        overlay[..., 0] = np.where(region, 1.0, overlay[..., 0])
        overlay[..., 1] = np.where(region, 0.5, overlay[..., 1])
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay (red = prediction)")

    for axis in axes.ravel():
        axis.axis("off")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=130)
    plt.close(fig)
    return destination


def train(config: AppConfig) -> dict:
    """Train, evaluate and save artefacts. Returns the metrics dictionary."""
    from medsam_seg.models import build_model

    pairs = data_module.discover_pairs(
        config.data.image_dir, config.data.mask_dir, config.data.mask_suffix
    )
    if not pairs:
        raise FileNotFoundError(
            f"no image/mask pairs found in {config.data.image_dir} and {config.data.mask_dir}"
        )

    train_pairs, test_pairs = data_module.split_pairs(
        pairs, config.data.val_split, config.data.seed
    )
    print(f"Found {len(pairs)} image/mask pairs")
    print(f"Training: {len(train_pairs)}   Validation: {len(test_pairs)}")

    print("Loading images...")
    X_train, y_train = data_module.load_split(
        train_pairs, tuple(config.data.image_size), config.data.mask_threshold
    )
    X_test, y_test = data_module.load_split(
        test_pairs, tuple(config.data.image_size), config.data.mask_threshold
    )
    print(f"Train tensors {X_train.shape}   Test tensors {X_test.shape}")

    model = build_model(
        arch=config.model.arch,
        input_shape=(*config.data.image_size, 3),
        filters=tuple(config.model.filters),
        dropout=config.model.dropout,
        depth=config.model.depth,
    )
    print(f"Architecture: {config.model.arch}   Parameters: {model.count_params():,}")

    from tensorflow.keras.optimizers import Adam

    model.compile(
        optimizer=Adam(learning_rate=config.training.learning_rate),
        loss=build_loss(config.training.loss, config.training.pos_weight),
        metrics=["accuracy"],
    )

    config.training.model_save_path.parent.mkdir(parents=True, exist_ok=True)
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    callbacks = [
        ModelCheckpoint(
            str(config.training.model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.training.reduce_lr_patience,
            min_lr=1e-5,
            verbose=0,
        ),
        EpochLogger(config.training.epochs),
    ]
    if config.training.early_stopping_patience > 0:
        from tensorflow.keras.callbacks import EarlyStopping

        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config.training.early_stopping_patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0,
    )

    print("Evaluating...")
    start = time.time()
    y_pred = model.predict(X_test, verbose=0)
    total_time = time.time() - start
    y_pred_binary = (y_pred > config.eval.threshold).astype(np.float32)

    scores = metrics.compute_all(y_test, y_pred_binary, config.eval.threshold)
    scores["n_test"] = len(test_pairs)
    scores["mean_inference_ms"] = total_time / max(len(test_pairs), 1) * 1000.0
    scores["class_0_accuracy"] = metrics.class_accuracy(
        y_test, y_pred_binary, 0, config.eval.threshold
    )
    scores["class_1_accuracy"] = metrics.class_accuracy(
        y_test, y_pred_binary, 1, config.eval.threshold
    )

    out_dir = Path(config.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, out_dir / "training_history.png")
    plot_samples(
        X_test, y_test, y_pred_binary, out_dir / "sample_predictions.png", config.eval.num_samples
    )
    (out_dir / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key in ("accuracy", "dice", "iou", "f1", "precision", "recall"):
        print(f"  {key:<12} {scores[key]:.4f}")
    print(f"  {'class 0 acc':<12} {scores['class_0_accuracy']:.4f}")
    print(f"  {'class 1 acc':<12} {scores['class_1_accuracy']:.4f}")
    print(f"  {'inference':<12} {scores['mean_inference_ms']:.1f} ms/image")
    print(f"\nModel saved to {config.training.model_save_path}")
    print(f"Plots and metrics in {out_dir}")
    return scores
