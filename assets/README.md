# Figures

Two charts are committed here and are generated directly from the reported
numbers, so the README renders correctly on a fresh clone:

| File | Contents |
| --- | --- |
| `inference_comparison.png` | inference time per model |
| `class_balance.png` | mask class distribution and reported scores |

## Add your own figures

Drop your report figures in this folder with these names and the README will
pick them up:

| Name it | What it should show |
| --- | --- |
| `architecture.png` | U-Net architecture diagram |
| `training_history.png` | accuracy and loss curves (report Figure 4) |
| `sample_predictions.png` | input / ground truth / prediction grid (Figure 4) |
| `confusion_matrix.png` | pixel confusion matrix |
| `gui_screenshot.png` | the desktop application |

```bash
cp "path/to/your/figure.png" assets/training_history.png
```

Anything not present is simply not shown — nothing breaks.
