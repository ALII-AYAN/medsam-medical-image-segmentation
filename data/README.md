# Where your dataset goes

Put your images and masks here:

```
data/
├── images/     # 5,176 medical images   (case0001.png, case0002.png, ...)
└── masks/      # 5,176 ground-truth masks (case0001_mask.png, ...)
```

## Accepted naming

The mask filename must relate to the image filename in one of these ways:

| Image | Mask | Works |
| --- | --- | --- |
| `case1.png` | `case1_mask.png` | ✅ default convention |
| `case1.png` | `case1.png` | ✅ identical name |
| `case1.jpg` | `case1_mask.png` | ✅ different extension |

Anything else is skipped with a warning. Check what was found with:

```bash
python -m medsam_seg check
```

## Coming from the original folders

If your data is still in `selected_images/` and `selected_masks/`, this copies
it into place with clean naming:

```bash
python scripts/prepare_dataset.py \
    --images "path/to/dataset_complete/selected_images" \
    --masks  "path/to/dataset_complete/selected_masks" \
    --output data
```

## Do I commit the dataset?

**No.** It is 5,176 image pairs — far too large for GitHub, and medical data
usually carries licence or privacy restrictions. `data/images/*` and
`data/masks/*` are git-ignored, so the files stay on your machine.

If you want the dataset available to others, use one of:

- **Git LFS** — `git lfs track "data/**"` (GitHub gives 1 GB free)
- **A hosted dataset** — Kaggle, Zenodo, or a Hugging Face dataset repo, linked
  from the main README
- **A release asset** — attach a zip to a GitHub Release

Anyone cloning the repo then runs `scripts/prepare_dataset.py` on their own
copy, and everything works the same way.
