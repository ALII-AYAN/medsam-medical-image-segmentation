"""Command line entry point.

python -m medsam_seg check          verify the dataset layout
python -m medsam_seg info           print the resolved configuration
python -m medsam_seg train          train a model
python -m medsam_seg evaluate       score a saved model
python -m medsam_seg predict IMAGE  segment one image
python -m medsam_seg compare IMAGE  compare every architecture with weights
python -m medsam_seg gui            open the desktop application
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from medsam_seg.config import AppConfig

app = typer.Typer(add_completion=False, help="Medical image segmentation toolkit.")


def _load(config: Path | None) -> AppConfig:
    return AppConfig.from_yaml(config) if config else AppConfig()


@app.command()
def check(
    image_dir: Path | None = typer.Option(None, help="Override the image directory"),
    mask_dir: Path | None = typer.Option(None, help="Override the mask directory"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    sample: int = typer.Option(200, help="How many masks to sample for balance stats"),
) -> None:
    """Verify the dataset is readable and report the mask balance."""
    from medsam_seg import data as data_module

    settings = _load(config)
    images = Path(image_dir or settings.data.image_dir)
    masks = Path(mask_dir or settings.data.mask_dir)

    try:
        pairs = data_module.discover_pairs(images, masks, settings.data.mask_suffix)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not pairs:
        typer.echo(f"error: no image/mask pairs found under {images}", err=True)
        typer.echo("  Masks must match image filenames, e.g. case1.png -> case1_mask.png")
        raise typer.Exit(1)

    summary = data_module.dataset_summary(
        pairs, tuple(settings.data.image_size), settings.data.mask_threshold, sample
    )
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"\nDataset OK: {summary['n_pairs']} pairs")


@app.command()
def info(config: Path | None = typer.Option(None, "--config", "-c")) -> None:
    """Print the resolved configuration."""
    settings = _load(config)
    typer.echo(settings.model_dump_json(indent=2))


@app.command()
def train(
    config: Path | None = typer.Option(None, "--config", "-c"),
    arch: str = typer.Option(None, help="unet | deeplabv3 | segnet"),
    epochs: int = typer.Option(None),
    batch_size: int = typer.Option(None),
    learning_rate: float = typer.Option(None),
    loss: str = typer.Option(None, help="bce | dice | bce_dice"),
    pos_weight: float = typer.Option(None, help="Weight for foreground pixels"),
    image_dir: Path | None = typer.Option(None),
    mask_dir: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None, help="Where to save the model"),
) -> None:
    """Train a segmentation model."""
    from medsam_seg.train import train as run_training

    settings = _load(config)
    if arch:
        settings.model.arch = arch
    if epochs:
        settings.training.epochs = epochs
    if batch_size:
        settings.training.batch_size = batch_size
    if learning_rate:
        settings.training.learning_rate = learning_rate
    if loss:
        settings.training.loss = loss
    if pos_weight:
        settings.training.pos_weight = pos_weight
    if image_dir:
        settings.data.image_dir = Path(image_dir)
    if mask_dir:
        settings.data.mask_dir = Path(mask_dir)
    if output:
        settings.training.model_save_path = Path(output)

    try:
        run_training(settings)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def evaluate(
    model: Path | None = typer.Option(None, "--model", "-m", help="Weights file"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Evaluate saved weights on the validation split."""
    from medsam_seg.evaluate import evaluate as run_evaluation

    settings = _load(config)
    run_evaluation(settings, model)


def _load_ground_truth(path: Path, size: tuple[int, int]):
    """Read a reference mask as a binary array shaped (H, W, 1)."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as handle:
        array = (
            np.asarray(handle.convert("L").resize(size, Image.NEAREST), dtype=np.float32) / 255.0
        )
    return (array > 0.5).astype(np.float32)[..., np.newaxis]


@app.command()
def predict(
    image: Path = typer.Argument(..., help="Image to segment"),
    model: Path | None = typer.Option(None, "--model", "-m"),
    mask: Path | None = typer.Option(None, help="Ground-truth mask for Dice/IoU"),
    output_dir: Path = typer.Option(Path("outputs"), "--output", "-o"),
    threshold: float = typer.Option(0.5),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Segment one image and save the mask and overlay."""
    from medsam_seg.metrics import dice_score, iou_score
    from medsam_seg.predict import Predictor, save_mask, save_overlay

    settings = _load(config)
    model_path = Path(model or settings.inference.model_path)
    if not model_path.exists():
        typer.echo(f"error: weights not found: {model_path}", err=True)
        raise typer.Exit(1)

    predictor = Predictor.from_path(model_path, image_size=tuple(settings.data.image_size))
    result = predictor.predict(str(image), threshold)

    if mask:
        ground = _load_ground_truth(Path(mask), tuple(settings.data.image_size))
        result.dice = dice_score(ground, result.mask, threshold)
        result.iou = iou_score(ground, result.mask, threshold)

    # Reuse the exact tensor the model saw, so the overlay lines up with the mask.
    display_image = predictor.preprocess(str(image))[0]
    save_mask(result.mask, output_dir / f"{Path(image).stem}_mask.png")
    save_overlay(display_image, result.mask, output_dir / f"{Path(image).stem}_overlay.png")
    typer.echo(json.dumps(result.to_dict(), indent=2))
    typer.echo(f"\nSaved to {output_dir}")


@app.command()
def compare(
    image: Path = typer.Argument(..., help="Image to compare on"),
    weights_dir: Path = typer.Option(Path("models"), help="Folder holding the weights"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Compare every architecture that has weights on one image."""
    from medsam_seg.predict import compare_models

    settings = _load(config)
    report = compare_models(
        str(image),
        Path(weights_dir),
        image_size=tuple(settings.data.image_size),
        threshold=settings.inference.threshold,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2))
    if report.missing:
        typer.echo(f"\nNo weights for: {', '.join(report.missing)}")
        typer.echo("Train them with:  python -m medsam_seg train --arch <name>")


@app.command()
def gui(
    model: Path | None = typer.Option(None, "--model", "-m"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Open the desktop application."""
    from medsam_seg.gui import launch

    settings = _load(config)
    if model:
        settings.inference.model_path = Path(model)
    launch(settings)


@app.command()
def init(output: Path = typer.Option(Path("configs/default.yaml"), "--output", "-o")) -> None:
    """Write a default configuration file."""
    settings = AppConfig()
    path = settings.to_yaml(output)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
