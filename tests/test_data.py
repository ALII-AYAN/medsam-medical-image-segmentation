"""Dataset discovery, mask matching and splitting."""

import numpy as np
import pytest
from PIL import Image

from medsam_seg.data import (
    discover_pairs,
    find_mask,
    load_image,
    load_pair,
    load_split,
    split_pairs,
)


def test_discovers_every_matched_pair(dataset_dir):
    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    # 6 real pairs; the orphan image and ghost mask must both be ignored.
    assert len(pairs) == 6


def test_orphan_image_is_skipped(dataset_dir):
    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    assert "orphan" not in {p.name for p in pairs}


def test_ghost_mask_is_skipped(dataset_dir):
    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    assert all("ghost" not in p.mask_path.name for p in pairs)


def test_suffix_convention(dataset_dir):
    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    assert pairs[0].mask_path.name == "case0_mask.png"


def test_same_name_convention(tmp_path):
    """Masks with identical filenames must also resolve."""
    images, masks = tmp_path / "i", tmp_path / "m"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (8, 8)).save(images / "a.png")
    Image.new("L", (8, 8)).save(masks / "a.png")
    assert len(discover_pairs(images, masks)) == 1


def test_different_extension(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (8, 8)).save(images / "a.jpg")
    Image.new("L", (8, 8)).save(masks / "a_mask.png")
    assert len(discover_pairs(images, masks)) == 1


def test_find_mask_returns_none_when_absent(tmp_path):
    images = tmp_path / "i"
    images.mkdir()
    assert find_mask(images / "missing.png", tmp_path / "nowhere") is None


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="image directory"):
        discover_pairs(tmp_path / "nope", tmp_path / "nope")


def test_load_image_is_normalised_and_rgb(dataset_dir):
    array = load_image(dataset_dir / "images" / "case0.png", (32, 32))
    assert array.shape == (32, 32, 3)
    assert array.dtype == np.float32
    assert 0.0 <= array.min() and array.max() <= 1.0


def test_load_image_grayscale_has_channel_axis(dataset_dir):
    array = load_image(dataset_dir / "masks" / "case3_mask.png", (16, 16), grayscale=True)
    assert array.shape == (16, 16, 1)


def test_load_pair_binarises_the_mask(dataset_dir):
    _image, mask = load_pair(
        __import__("medsam_seg.data", fromlist=["Pair"]).Pair(
            dataset_dir / "images" / "case4.png", dataset_dir / "masks" / "case4_mask.png"
        ),
        (16, 16),
    )
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_split_is_deterministic(dataset_dir):
    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    first = split_pairs(pairs, 0.2, seed=42)
    second = split_pairs(pairs, 0.2, seed=42)
    assert [p.name for p in first[0]] == [p.name for p in second[0]]


def test_split_sizes(dataset_dir):
    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    train, val = split_pairs(pairs, 0.2, seed=42)
    assert len(train) == 5 and len(val) == 1
    assert len(train) + len(val) == len(pairs)


def test_split_rejects_bad_ratio(dataset_dir):
    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    with pytest.raises(ValueError, match="val_split"):
        split_pairs(pairs, 1.5)


def test_load_split_stacks_into_arrays(dataset_dir):
    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    images, masks = load_split(pairs, (16, 16))
    assert images.shape == (6, 16, 16, 3)
    assert masks.shape == (6, 16, 16, 1)


def test_dataset_summary_reports_balance(dataset_dir):
    from medsam_seg.data import dataset_summary

    pairs = discover_pairs(dataset_dir / "images", dataset_dir / "masks")
    summary = dataset_summary(pairs, (16, 16), sample=6)
    assert summary["n_pairs"] == 6
    assert 0.0 <= summary["foreground_percent"] <= 100.0
    assert summary["background_percent"] + summary["foreground_percent"] == pytest.approx(
        100.0, abs=0.05
    )
