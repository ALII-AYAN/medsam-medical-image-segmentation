"""Single-image inference and the multi-architecture comparison.

The comparison runs *real* inference for every architecture that has saved
weights. Architectures without weights are reported as unavailable rather than
estimated, so the table never contains numbers the model did not produce.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

COMPARE_ARCHITECTURES = ("medsam", "unet", "deeplabv3", "segnet")

# Filenames looked for when loading each architecture's weights.
WEIGHT_CANDIDATES = ("{name}.keras", "{name}.h5", "medsam_model.keras", "medsam_model.h5")


@dataclass
class PredictionResult:
    """Output of one model on one image."""

    model_name: str
    mask: np.ndarray
    inference_time: float
    foreground_percent: float
    dice: float | None = None
    iou: float | None = None
    available: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "time_s": round(self.inference_time, 4) if self.available else None,
            "foreground_percent": round(self.foreground_percent, 2) if self.available else None,
            "dice": round(self.dice, 4) if self.dice is not None else None,
            "iou": round(self.iou, 4) if self.iou is not None else None,
            "available": self.available,
            "note": self.note,
        }


class Predictor:
    """Wraps a loaded Keras model and keeps preprocessing in one place.

    Training and inference must resize and normalise identically, so both go
    through :meth:`preprocess`. Changing it in one place silently invalidates
    every metric otherwise.
    """

    def __init__(self, model, name: str = "medsam", image_size=(256, 256)) -> None:
        self.model = model
        self.name = name
        self.image_size = tuple(image_size)

    @classmethod
    def from_path(cls, path: Path, name: str = "medsam",
                  image_size=(256, 256)) -> "Predictor":
        from tensorflow.keras.models import load_model

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"weights not found: {path}")
        return cls(load_model(path, compile=False), name=name, image_size=image_size)

    def preprocess(self, image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
        """Accept a path, a PIL image or an array; return a 1CHW batch in [0, 1]."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                image = handle.convert("RGB")
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.max() > 1.0:
            array = array / 255.0

        # Resize expects (height, width); PIL stores images that way already.
        pil = Image.fromarray((array * 255.0).clip(0, 255).astype(np.uint8))
        resized = np.asarray(pil.resize(self.image_size, Image.BILINEAR),
                             dtype=np.float32) / 255.0
        return resized[np.newaxis, ...]

    def predict(self, image, threshold: float = 0.5) -> PredictionResult:
        from medsam_seg.metrics import foreground_fraction

        batch = self.preprocess(image)
        start = time.perf_counter()
        raw = self.model.predict(batch, verbose=0)
        elapsed = time.perf_counter() - start

        mask = (np.asarray(raw)[0] > threshold).astype(np.float32)
        return PredictionResult(
            model_name=self.name,
            mask=mask,
            inference_time=elapsed,
            foreground_percent=foreground_fraction(mask, threshold),
        )


def score_against(result: PredictionResult, ground_truth: np.ndarray,
                  threshold: float = 0.5) -> PredictionResult:
    """Fill in Dice/IoU once a ground-truth mask is available."""
    from medsam_seg.metrics import dice_score, iou_score

    result.dice = dice_score(ground_truth, result.mask, threshold)
    result.iou = iou_score(ground_truth, result.mask, threshold)
    return result


def find_weights(name: str, directory: Path) -> Path | None:
    """Locate the weights file for one architecture."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for pattern in WEIGHT_CANDIDATES:
        candidate = directory / pattern.format(name=name)
        if candidate.exists():
            return candidate
    # Fall back to any file whose stem mentions the architecture.
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and name in candidate.stem.lower():
            return candidate
    return None


@dataclass
class ComparisonReport:
    """Result of running every available architecture on one image."""

    results: list[PredictionResult] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "missing": self.missing,
        }


def compare_models(image, weights_dir: Path, image_size=(256, 256),
                   threshold: float = 0.5,
                   ground_truth: np.ndarray | None = None) -> ComparisonReport:
    """Run each architecture that has weights; list the ones that do not.

    Architectures with no saved weights are returned in ``missing`` so the UI
    can say so explicitly instead of inventing a number.
    """
    from medsam_seg.metrics import dice_score, iou_score

    report = ComparisonReport()
    for name in COMPARE_ARCHITECTURES:
        path = find_weights(name, Path(weights_dir))
        if path is None:
            report.missing.append(name)
            continue
        try:
            predictor = Predictor.from_path(path, name=name, image_size=image_size)
            result = predictor.predict(image, threshold)
            if ground_truth is not None:
                result.dice = dice_score(ground_truth, result.mask, threshold)
                result.iou = iou_score(ground_truth, result.mask, threshold)
            report.results.append(result)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            report.results.append(
                PredictionResult(name, np.zeros(image_size, dtype=np.float32), 0.0, 0.0,
                                 available=False, note=f"load failed: {exc}")
            )
    return report


def save_mask(mask: np.ndarray, destination: Path) -> Path:
    """Write a predicted mask as a PNG."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = (np.asarray(mask).squeeze() > 0.5).astype(np.uint8) * 255
    Image.fromarray(array).save(destination)
    return destination


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the predicted region in red over the original image."""
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    overlay = array.copy()
    region = np.asarray(mask).squeeze() > 0.5
    if region.shape != overlay.shape[:2]:
        pil = Image.fromarray((region * 255).astype(np.uint8)).resize(
            (overlay.shape[1], overlay.shape[0]), Image.NEAREST
        )
        region = np.asarray(pil) > 127
    overlay[..., 0] = np.where(region, 1.0, (1 - alpha) * overlay[..., 0])
    overlay[..., 1] *= 1 - alpha * region
    overlay[..., 2] *= 1 - alpha * region
    return np.clip(overlay, 0.0, 1.0)


def save_overlay(image: np.ndarray, mask: np.ndarray, destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = (make_overlay(image, mask) * 255).astype(np.uint8)
    Image.fromarray(array).save(destination)
    return destination
