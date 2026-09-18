# Trained weights

Put your trained model here:

```
models/
├── medsam_model.h5        # your trained weights (or .keras)
├── unet.keras             # optional: for the four-way comparison
├── deeplabv3.keras
└── segnet.keras
```

Point the config at it:

```yaml
inference:
  model_path: models/medsam_model.h5
```

or pass it per command:

```bash
python -m medsam_seg predict scan.png --model models/medsam_model.h5
python -m medsam_seg gui --model models/medsam_model.h5
```

Both `.keras` and `.h5` are accepted.

## Do I commit the model?

Usually **no** — a 1.95M-parameter checkpoint is 20–25 MB, which bloats the
repository, and `.gitignore` already excludes `models/*.h5` and `models/*.keras`.

Better options:

1. **GitHub Release** — attach the `.h5` to a release; it stays downloadable
   and out of git history.
2. **Hugging Face Hub** — `huggingface-cli upload` and link it in the README.
3. **Git LFS** — `git lfs track "models/*.h5"` if you want it versioned.

Whichever you choose, document the download in the README so a visitor can
reproduce your results.

## The four-way comparison

The comparison table reports **only models that have weights here**. Any
architecture without a weights file is listed as unavailable rather than
estimated, so the table never contains a number a model did not produce.

Train the comparison models with:

```bash
python -m medsam_seg train --arch unet      --config configs/default.yaml
python -m medsam_seg train --arch deeplabv3 -c configs/deeplabv3.yaml
python -m medsam_seg train --arch segnet    -c configs/segnet.yaml
```
