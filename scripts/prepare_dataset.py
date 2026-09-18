#!/usr/bin/env python3
"""Copy a raw dataset into data/images and data/masks with consistent naming.

Handles the case where masks already use a suffix (``case1.png`` ->
``case1_mask.png``) and the case where they do not, so a dataset dropped in
as-is becomes loadable in one step.

Usage
-----
    python scripts/prepare_dataset.py --images RAW/selected_images \\
        --masks RAW/selected_masks --output data
    python scripts/prepare_dataset.py --images RAW/img --masks RAW/masks \\
        --output data --mask-suffix _mask --limit 500
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medsam_seg.data import IMAGE_EXTENSIONS, find_mask


def collect(directory: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(directory.glob(pattern))
        files.extend(directory.glob(pattern.upper()))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", default=PROJECT_ROOT / "data", type=Path)
    parser.add_argument("--mask-suffix", default="_mask")
    parser.add_argument("--limit", type=int, help="Copy only the first N pairs")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    args = parser.parse_args()

    images_dir, masks_dir = Path(args.images), Path(args.masks)
    for label, path in (("images", images_dir), ("masks", masks_dir)):
        if not path.is_dir():
            print(f"error: {label} directory not found: {path}")
            sys.exit(1)

    out_images = Path(args.output) / "images"
    out_masks = Path(args.output) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    transfer = shutil.move if args.move else shutil.copy2
    copied, skipped, unmatched = 0, 0, []

    for image_path in collect(images_dir):
        if args.limit and copied >= args.limit:
            break
        # Accept the suffix convention or an identical filename.
        mask_path = find_mask(image_path, masks_dir, args.mask_suffix)
        if mask_path is None:
            unmatched.append(image_path.name)
            continue

        target_image = out_images / image_path.name
        target_mask = out_masks / f"{image_path.stem}{args.mask_suffix}.png"
        if not target_image.exists():
            transfer(image_path, target_image)
        if not target_mask.exists():
            transfer(mask_path, target_mask)
        copied += 1

    print(f"Copied {copied} image/mask pairs -> {Path(args.output)}")
    if unmatched:
        print(f"Skipped {len(unmatched)} images with no matching mask")
        for name in unmatched[:10]:
            print(f"  {name}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
    if skipped:
        print(f"{skipped} already present, left untouched")

    print("\nVerify with:")
    print("  python -m medsam_seg check")


if __name__ == "__main__":
    main()
