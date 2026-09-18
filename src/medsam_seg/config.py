"""Configuration for training, evaluation and the GUI.

Every value has a usable default so the code runs with no config file at all;
a YAML file (or CLI flags) overrides only what you change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataConfig(BaseModel):
    """Where the images and masks live and how they are prepared."""

    image_dir: Path = Field(default=PROJECT_ROOT / "data" / "images")
    mask_dir: Path = Field(default=PROJECT_ROOT / "data" / "masks")
    image_size: tuple[int, int] = Field(default=(256, 256))
    # How the mask filename relates to the image filename: "cat.png" -> ?
    mask_suffix: str = Field(default="_mask")
    val_split: float = Field(default=0.2, ge=0.05, le=0.5)
    seed: int = Field(default=42)
    mask_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    """Architecture settings."""

    arch: Literal["unet", "deeplabv3", "segnet"] = Field(default="unet")
    filters: tuple[int, int, int, int] = Field(default=(32, 64, 128, 256))
    dropout: float = Field(default=0.2, ge=0.0, le=0.9)
    depth: int = Field(default=3, ge=1, le=4)

    @field_validator("filters")
    @classmethod
    def _check_filters(cls, value):
        if len(value) != 4:
            raise ValueError("filters must have exactly 4 entries")
        return value


class TrainConfig(BaseModel):
    """Training hyperparameters."""

    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    loss: Literal["bce", "dice", "bce_dice"] = Field(default="bce_dice")
    # Positive-class weight for binary cross-entropy. Medical masks are usually
    # heavily background-dominated, so 1.0 means "ignore the imbalance".
    pos_weight: float = Field(default=1.0, ge=0.05, le=50.0)
    output_dir: Path = Field(default=PROJECT_ROOT / "outputs")
    model_save_path: Path = Field(default=PROJECT_ROOT / "models" / "medsam_model.keras")
    reduce_lr_patience: int = Field(default=2, ge=1)
    early_stopping_patience: int = Field(default=0, ge=0)  # 0 disables it


class EvalConfig(BaseModel):
    """Evaluation settings."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    num_samples: int = Field(default=3, ge=1, le=10)
    output_dir: Path = Field(default=PROJECT_ROOT / "outputs")


class InferenceConfig(BaseModel):
    """Prediction and GUI settings."""

    model_path: Path = Field(default=PROJECT_ROOT / "models" / "medsam_model.keras")
    compare_dir: Path = Field(default=PROJECT_ROOT / "models" / "compare")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    """Top-level configuration."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @classmethod
    def from_dict(cls, payload: dict) -> "AppConfig":
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return cls.from_dict(payload)

    def to_yaml(self, path: Path | str) -> Path:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(self.model_dump_json())
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path
