"""Dataset discovery and loading.

The dataset is a folder of images and a folder of matching masks. Masks are
matched to images by filename, and three naming conventions are recognised so
the project works with a dataset dropped in as-is:

    image:  case_001.png      mask:  case_001_mask.png   (suffix, default)
    image:  case_001.png      mask:  case_001.png        (same name)
    image:  case_001.jpg      mask:  case_001.png        (same stem, any ext)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


@dataclass
class Pair:
    """One image and its ground-truth mask."""

    image_path: Path
    mask_path: Path

    @property
    def name(self) -> str:
        return self.image_path.stem


def _collect(directory: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(directory.glob(pattern))
        files.extend(directory.glob(pattern.upper()))
    return sorted(set(files))


def find_mask(image_path: Path, mask_dir: Path, mask_suffix: str = "_mask") -> Path | None:
    """Resolve the mask belonging to ``image_path`` under any known convention."""
    stem = image_path.stem
    candidates = [
        mask_dir / f"{stem}{mask_suffix}{image_path.suffix}",
        mask_dir / f"{stem}{mask_suffix}.png",
        mask_dir / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Same stem with a different mask extension.
    for candidate in mask_dir.iterdir() if mask_dir.is_dir() else []:
        if candidate.stem in (stem, f"{stem}{mask_suffix}"):
            return candidate
    return None


def discover_pairs(image_dir: Path, mask_dir: Path, mask_suffix: str = "_mask") -> list[Pair]:
    """Return every image that has a matching mask, in sorted order."""
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask directory not found: {mask_dir}")

    pairs: list[Pair] = []
    unmatched: list[str] = []
    for image_path in _collect(image_dir):
        mask_path = find_mask(image_path, mask_dir, mask_suffix)
        if mask_path is None:
            unmatched.append(image_path.name)
            continue
        pairs.append(Pair(image_path, mask_path))
    return pairs


def load_image(
    path: Path, size: tuple[int, int] = (256, 256), grayscale: bool = False
) -> np.ndarray:
    """Load an image as a float32 array in [0, 1].

    RGB for inputs, single-channel for masks. Values are divided by 255 so the
    sigmoid output of the network lives on the same scale.
    """
    mode = "L" if grayscale else "RGB"
    with Image.open(path) as handle:
        image = handle.convert(mode)
        image = image.resize(size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if grayscale:
        array = array[..., np.newaxis]
    return array


def load_pair(
    pair: Pair, size: tuple[int, int] = (256, 256), mask_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image/mask pair, binarising the mask."""
    image = load_image(pair.image_path, size, grayscale=False)
    mask = load_image(pair.mask_path, size, grayscale=True)
    mask = (mask > mask_threshold).astype(np.float32)
    return image, mask


def split_pairs(
    pairs: list[Pair], val_split: float = 0.2, seed: int = 42
) -> tuple[list[Pair], list[Pair]]:
    """Deterministic shuffle then split. The seed makes runs reproducible."""
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_split))
    return shuffled[:-n_val], shuffled[-n_val:]


def load_split(pairs: list[Pair], size: tuple[int, int] = (256, 256), mask_threshold: float = 0.5):
    """Load a whole split into two NumPy arrays."""
    images, masks = [], []
    for pair in pairs:
        image, mask = load_pair(pair, size, mask_threshold)
        images.append(image)
        masks.append(mask)
    return np.stack(images), np.stack(masks)


def dataset_summary(
    pairs: list[Pair],
    size: tuple[int, int] = (256, 256),
    mask_threshold: float = 0.5,
    sample: int | None = 200,
) -> dict:
    """Cheap overview: counts, mask balance and a foreground estimate."""
    from medsam_seg.metrics import class_distribution

    distributions = [
        class_distribution(load_pair(p, size, mask_threshold)[1], mask_threshold)
        for p in (pairs[:sample] if sample else pairs)
    ]
    foreground = [d.get(1, 0.0) for d in distributions if d]
    return {
        "n_pairs": len(pairs),
        "image_size": list(size),
        "background_percent": round(float(np.mean([d.get(0, 0.0) for d in distributions])), 2)
        if foreground
        else 0.0,
        "foreground_percent": round(float(np.mean(foreground)), 2) if foreground else 0.0,
        "sampled": len(distributions),
    }
