import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture()
def dataset_dir(tmp_path):
    """A small image/mask dataset using the 'X.png' -> 'X_mask.png' convention."""
    from PIL import Image

    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    for index in range(6):
        # Images vary so a model cannot trivially memorise them.
        array = [
            [255 if (row + col + index) % 3 == 0 else 0 for col in range(16)] for row in range(16)
        ]
        Image.fromarray(_to_uint8(array)).save(images / f"case{index}.png")
        # Masks: a growing square, so foreground fraction differs per sample.
        mask = [
            [255 if (row < index and col < index) else 0 for col in range(16)] for row in range(16)
        ]
        Image.fromarray(_to_uint8(mask)).save(masks / f"case{index}_mask.png")

    # Deliberate defects: an image with no mask, and a mask with no image.
    Image.fromarray(_to_uint8(array)).save(images / "orphan.png")
    Image.fromarray(_to_uint8(mask)).save(masks / "ghost_mask.png")
    return tmp_path


def _to_uint8(grid):
    import numpy as np

    return np.asarray(grid, dtype=np.uint8)


@pytest.fixture()
def flat_masks():
    import numpy as np

    # 4x4: truth is the left half, prediction gets 3 of 4 correct columns.
    truth = np.array([[1, 1, 0, 0]] * 4, dtype=np.float32)
    prediction = np.array([[1, 1, 1, 0]] * 4, dtype=np.float32)
    return truth, prediction
