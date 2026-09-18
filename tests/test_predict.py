"""Preprocessing, weight discovery and overlay - the parts that need no TF."""

import numpy as np
import pytest
from PIL import Image

from medsam_seg.predict import (
    COMPARE_ARCHITECTURES, PredictionResult, find_weights, make_overlay,
    save_mask,
)


class _StubModel:
    """Minimal stand-in: returns a constant half-image mask."""

    def __init__(self, size=(256, 256)):
        self.size = size

    def predict(self, batch, verbose=0):
        out = np.zeros((batch.shape[0], *self.size, 1), dtype=np.float32)
        out[..., : self.size[0] // 2, :] = 0.9   # top half is "foreground"
        return out


def _predictor(tmp_path=None, size=(64, 64)):
    from medsam_seg.predict import Predictor

    return Predictor(_StubModel(size), name="medsam", image_size=size)


def test_preprocess_from_path(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (120, 80), color=(10, 20, 30)).save(path)
    batch = _predictor().preprocess(str(path))
    assert batch.shape == (1, 64, 64, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_preprocess_from_pil():
    image = Image.new("RGB", (50, 50), color=(255, 0, 0))
    assert _predictor().preprocess(image).shape == (1, 64, 64, 3)


def test_preprocess_from_array_is_scaled():
    array = np.full((40, 40, 3), 200, dtype=np.float32)   # 0-255 range
    batch = _predictor().preprocess(array)
    assert batch.max() <= 1.0
    assert batch.max() > 0.5


def test_preprocess_expands_grayscale():
    array = np.full((30, 30), 128, dtype=np.float32)
    assert _predictor().preprocess(array).shape == (1, 64, 64, 3)


def test_preprocess_is_idempotent():
    """Running it twice must not change the result (no accidental rescaling)."""
    predictor = _predictor()
    array = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    first = predictor.preprocess(array)
    second = predictor.preprocess(first[0])
    np.testing.assert_allclose(first, second, atol=1e-2)


def test_predict_returns_real_timing_and_area():
    result = _predictor().predict(np.zeros((40, 40, 3), dtype=np.float32))
    assert isinstance(result, PredictionResult)
    assert result.inference_time >= 0.0
    assert result.foreground_percent == pytest.approx(50.0, abs=1.0)
    assert result.mask.shape == (64, 64, 1)


def test_predict_threshold_changes_the_mask():
    predictor = _predictor()
    image = np.zeros((40, 40, 3), dtype=np.float32)
    assert predictor.predict(image, threshold=0.5).foreground_percent > 0
    assert predictor.predict(image, threshold=0.95).foreground_percent == 0.0


def test_find_weights_prefers_keras(tmp_path):
    (tmp_path / "unet.keras").write_bytes(b"")
    (tmp_path / "unet.h5").write_bytes(b"")
    assert find_weights("unet", tmp_path).name == "unet.keras"


def test_find_weights_falls_back_to_h5(tmp_path):
    (tmp_path / "deeplabv3.h5").write_bytes(b"")
    assert find_weights("deeplabv3", tmp_path).name == "deeplabv3.h5"


def test_find_weights_matches_stem(tmp_path):
    (tmp_path / "best_segnet_model.keras").write_bytes(b"")
    assert find_weights("segnet", tmp_path) is not None


def test_find_weights_returns_none_for_missing_dir(tmp_path):
    assert find_weights("unet", tmp_path / "nope") is None


def test_find_weights_returns_none_when_absent(tmp_path):
    assert find_weights("unet", tmp_path) is None


def test_all_compare_architectures_are_known():
    assert COMPARE_ARCHITECTURES == ("medsam", "unet", "deeplabv3", "segnet")


def test_overlay_marks_foreground_red():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    mask = np.zeros((8, 8, 1), dtype=np.float32)
    mask[:4, :] = 1.0
    overlay = make_overlay(image, mask)
    assert overlay[:4, :, 0].mean() > overlay[4:, :, 0].mean()
    assert overlay.max() <= 1.0


def test_overlay_handles_mismatched_sizes():
    image = np.zeros((64, 64, 3), dtype=np.float32)
    mask = np.zeros((16, 16, 1), dtype=np.float32)
    mask[:8, :] = 1.0
    assert make_overlay(image, mask).shape == (64, 64, 3)


def test_save_mask_writes_a_binary_png(tmp_path):
    mask = np.zeros((16, 16, 1), dtype=np.float32)
    mask[:8, :] = 1.0
    path = save_mask(mask, tmp_path / "nested" / "m.png")
    assert path.exists()
    reloaded = np.asarray(Image.open(path).convert("L"))
    assert set(np.unique(reloaded)) <= {0, 255}


def test_result_to_dict_is_json_safe():
    import json

    payload = _predictor().predict(np.zeros((8, 8, 3), dtype=np.float32)).to_dict()
    assert json.dumps(payload)
    assert payload["available"] is True
