"""Configuration parsing and validation."""

import pytest
from pydantic import ValidationError

from medsam_seg.config import AppConfig


def test_defaults_point_into_the_project():
    config = AppConfig()
    assert config.data.image_dir.name == "images"
    assert config.data.mask_dir.name == "masks"
    assert config.model.arch == "unet"
    assert config.training.epochs == 5


def test_partial_yaml_merges_over_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("training:\n  epochs: 20\nmodel:\n  arch: segnet\n", encoding="utf-8")
    config = AppConfig.from_yaml(path)
    assert config.training.epochs == 20
    assert config.model.arch == "segnet"
    assert config.training.batch_size == 4  # untouched default


def test_yaml_roundtrip(tmp_path):
    original = AppConfig.from_dict({"training": {"epochs": 3}, "model": {"arch": "deeplabv3"}})
    restored = AppConfig.from_yaml(original.to_yaml(tmp_path / "c.yaml"))
    assert restored.training.epochs == 3 and restored.model.arch == "deeplabv3"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AppConfig.from_yaml(tmp_path / "nope.yaml")


def test_non_mapping_yaml_raises(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        AppConfig.from_yaml(path)


def test_invalid_values_are_rejected():
    with pytest.raises(ValidationError):
        AppConfig.from_dict({"data": {"val_split": 0.9}})
    with pytest.raises(ValidationError):
        AppConfig.from_dict({"training": {"epochs": 0}})
    with pytest.raises(ValidationError):
        AppConfig.from_dict({"training": {"learning_rate": -1.0}})
    with pytest.raises(ValidationError):
        AppConfig.from_dict({"model": {"arch": "resnet"}})


def test_filters_must_have_four_entries():
    with pytest.raises(ValidationError):
        AppConfig.from_dict({"model": {"filters": [16, 32]}})


def test_boundary_values_are_accepted():
    assert AppConfig.from_dict({"data": {"val_split": 0.05}}).data.val_split == 0.05
    assert AppConfig.from_dict({"data": {"val_split": 0.5}}).data.val_split == 0.5
    assert AppConfig.from_dict({"training": {"loss": "bce_dice"}}).training.loss == "bce_dice"
