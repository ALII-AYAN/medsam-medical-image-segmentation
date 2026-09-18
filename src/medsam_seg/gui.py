"""Tkinter desktop application for clinical segmentation.

Layout:
    left   - input image, prediction, overlay
    right  - comparison table and charts
    bottom - load / run / compare / save controls and a status bar

Run with::

    python -m medsam_seg gui
    python -m medsam_seg gui --model models/medsam_model.h5
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from medsam_seg.config import AppConfig
from medsam_seg.predict import (
    Predictor,
    compare_models,
    make_overlay,
    save_mask,
    save_overlay,
)

BG = "#ffffff"
PRIMARY = "#2c3e50"
ACCENT = "#3498db"
SUCCESS = "#27ae60"
MUTED = "#95a5a6"

MODEL_COLOURS = {
    "medsam": "#27ae60",
    "unet": "#3498db",
    "deeplabv3": "#9b59b6",
    "segnet": "#e67e22",
}


class SegmentationApp:
    """Desktop window for running and comparing segmentation models."""

    def __init__(self, root: tk.Tk, config: AppConfig) -> None:
        self.root = root
        self.config = config
        self.image_size = tuple(config.data.image_size)
        self.threshold = config.inference.threshold

        self.root.title("Medical Image Segmentation")
        self.root.geometry("1150x760")
        self.root.configure(bg=BG)

        self.image_path: Path | None = None
        self.image_array: np.ndarray | None = None
        self.ground_truth: np.ndarray | None = None
        self.predictor: Predictor | None = None
        self.latest_mask: np.ndarray | None = None
        self._photos: list[ImageTk.PhotoImage] = []  # keep refs; Tk does not copy

        self._build_layout()
        self._load_model()
        self._set_status("Ready - load an image to begin")

    # ------------------------------------------------------------------ setup

    def _load_model(self) -> None:
        path = Path(self.config.inference.model_path)
        if not path.exists():
            self._set_status(f"No weights at {path} - load one with --model", MUTED)
            return
        try:
            self.predictor = Predictor.from_path(path, name="medsam", image_size=self.image_size)
            self._set_status(f"Loaded {path.name}", SUCCESS)
        except Exception as exc:  # noqa: BLE001 - surface anything to the user
            messagebox.showerror("Model error", str(exc))
            self._set_status("Model failed to load", "#e74c3c")

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg=PRIMARY, height=70)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="Medical Image Segmentation",
            font=("Arial", 19, "bold"),
            bg=PRIMARY,
            fg="white",
        ).pack(pady=18)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_canvas = self._labelled_canvas(left, " Input image ")
        self.mask_canvas = self._labelled_canvas(left, " Predicted mask ")
        self.overlay_canvas = self._labelled_canvas(left, " Overlay (red = prediction) ")

        right = tk.Frame(main, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))
        self._build_table(right)
        self._build_charts(right)

        self._build_controls()

        self.status = tk.Label(
            self.root,
            text="",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#ecf0f1",
            fg=PRIMARY,
            font=("Arial", 9),
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _labelled_canvas(self, parent, title: str) -> tk.Canvas:
        frame = tk.LabelFrame(parent, text=title, font=("Arial", 10, "bold"), bg=BG, fg=PRIMARY)
        frame.pack(fill=tk.BOTH, expand=True, pady=4)
        canvas = tk.Canvas(frame, bg="white", highlightthickness=1, highlightbackground="#dfe6e9")
        canvas.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)
        canvas.create_text(160, 90, text="No image", fill=MUTED, font=("Arial", 11))
        return canvas

    def _build_table(self, parent) -> None:
        frame = tk.LabelFrame(
            parent, text=" Comparison ", font=("Arial", 10, "bold"), bg=BG, fg=PRIMARY
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        columns = ("Model", "Time (s)", "Dice", "IoU", "Foreground %")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", height=6)
        for column in columns:
            self.table.heading(column, text=column)
            self.table.column(column, width=95, anchor="center")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def _build_charts(self, parent) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        frame = tk.LabelFrame(
            parent, text=" Charts ", font=("Arial", 10, "bold"), bg=BG, fg=PRIMARY
        )
        frame.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(7, 3.2), dpi=100)
        self.figure.patch.set_facecolor(BG)
        self.ax_time = self.figure.add_subplot(121)
        self.ax_score = self.figure.add_subplot(122)
        self.canvas_widget = FigureCanvasTkAgg(self.figure, frame)
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._draw_empty_charts()

    def _build_controls(self) -> None:
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(pady=8)
        self.load_btn = self._button(bar, "Load image", self.load_image, ACCENT)
        self.gt_btn = self._button(
            bar, "Load ground truth", self.load_ground_truth, "#7f8c8d", state="disabled"
        )
        self.run_btn = self._button(bar, "Segment", self.run_prediction, SUCCESS, state="disabled")
        self.compare_btn = self._button(
            bar, "Compare models", self.run_comparison, "#9b59b6", state="disabled"
        )
        self.save_btn = self._button(
            bar, "Save results", self.save_results, "#34495e", state="disabled"
        )

    def _button(self, parent, text, command, colour, state="normal") -> tk.Button:
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 10, "bold"),
            bg=colour,
            fg="white",
            width=15,
            padx=8,
            pady=5,
            state=state,
        )
        button.pack(side=tk.LEFT, padx=4)
        return button

    # ------------------------------------------------------------- rendering

    def _show(self, canvas: tk.Canvas, array: np.ndarray | Image.Image) -> None:
        if isinstance(array, np.ndarray):
            if array.ndim == 2 or array.shape[-1] == 1:
                data = np.asarray(array).squeeze()
                image = Image.fromarray(
                    (data * 255).clip(0, 255).astype(np.uint8), mode="L"
                ).convert("RGB")
            else:
                image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
        else:
            image = array.convert("RGB")

        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 120)
        height = max(canvas.winfo_height(), 120)
        image.thumbnail((width - 10, height - 10), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._photos.append(photo)  # prevent garbage collection
        canvas.delete("all")
        canvas.create_image(width // 2, height // 2, image=photo)

    def _set_status(self, message: str, colour: str = PRIMARY) -> None:
        self.status.config(text=f"  {message}", fg=colour)
        self.root.update_idletasks()

    def _clear_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

    def _add_row(self, values) -> None:
        self.table.insert("", "end", values=values)

    # ---------------------------------------------------------------- actions

    def load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select medical image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with Image.open(path) as handle:
                rgb = handle.convert("RGB")
                self.image_array = (
                    np.asarray(rgb.resize(self.image_size, Image.BILINEAR), dtype=np.float32)
                    / 255.0
                )
                preview = rgb.copy()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Cannot open image: {exc}")
            return

        self.image_path = Path(path)
        self.ground_truth = None
        self.latest_mask = None
        self._show(self.input_canvas, preview)
        self._clear_table()
        self._clear_canvases()
        self._draw_empty_charts()

        self.gt_btn.config(state="normal")
        self.run_btn.config(state="normal" if self.predictor else "disabled")
        self.compare_btn.config(state="normal")
        self.save_btn.config(state="disabled")
        self._set_status(f"Loaded {self.image_path.name}", SUCCESS)

    def load_ground_truth(self) -> None:
        path = filedialog.askopenfilename(
            title="Select ground-truth mask",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with Image.open(path) as handle:
                mask = handle.convert("L").resize(self.image_size, Image.NEAREST)
                self.ground_truth = (np.asarray(mask, dtype=np.float32) / 255.0 > 0.5).astype(
                    np.float32
                )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Cannot open mask: {exc}")
            return
        self._set_status("Ground truth loaded - Dice and IoU will be reported", SUCCESS)

    def _clear_canvases(self) -> None:
        for canvas in (self.mask_canvas, self.overlay_canvas):
            canvas.delete("all")
            canvas.create_text(160, 90, text="Run segmentation", fill=MUTED, font=("Arial", 11))

    def run_prediction(self) -> None:
        if self.predictor is None:
            messagebox.showerror("Error", "No model loaded")
            return
        if self.image_array is None:
            messagebox.showerror("Error", "Load an image first")
            return

        self._set_status("Running segmentation...", SUCCESS)
        result = self.predictor.predict(self.image_array, self.threshold)
        if self.ground_truth is not None:
            from medsam_seg.metrics import dice_score, iou_score

            result.dice = dice_score(self.ground_truth, result.mask, self.threshold)
            result.iou = iou_score(self.ground_truth, result.mask, self.threshold)

        self.latest_mask = result.mask
        self._show(self.mask_canvas, result.mask)
        self._show(self.overlay_canvas, make_overlay(self.image_array, result.mask))
        self._clear_table()
        self._add_row(self._row(result))
        self._draw_charts([result])
        self.save_btn.config(state="normal")
        self._set_status(f"Segmentation finished in {result.inference_time:.3f}s", SUCCESS)

    def run_comparison(self) -> None:
        if self.image_array is None:
            messagebox.showerror("Error", "Load an image first")
            return

        self._set_status("Comparing models...", "#9b59b6")
        report = compare_models(
            self.image_array,
            Path(self.config.inference.compare_dir),
            image_size=self.image_size,
            threshold=self.threshold,
            ground_truth=self.ground_truth,
        )

        self._clear_table()
        for result in report.results:
            self._add_row(self._row(result))
        for name in report.missing:
            self._add_row((name.upper(), "-", "-", "-", "-"))

        available = [r for r in report.results if r.available]
        if available:
            self._draw_charts(available)
            self.latest_mask = available[0].mask
            self._show(self.mask_canvas, available[0].mask)
            self._show(self.overlay_canvas, make_overlay(self.image_array, available[0].mask))
            self.save_btn.config(state="normal")

        if report.missing:
            self._set_status(
                f"No weights for: {', '.join(report.missing)} - train them to compare",
                "#e67e22",
            )
        else:
            self._set_status(f"Compared {len(available)} models", SUCCESS)

    def _row(self, result) -> tuple:
        if not result.available:
            return (result.model_name.upper(), "-", "-", "-", "-")
        dice = f"{result.dice:.4f}" if result.dice is not None else "n/a"
        iou = f"{result.iou:.4f}" if result.iou is not None else "n/a"
        return (
            result.model_name.upper(),
            f"{result.inference_time:.3f}",
            dice,
            iou,
            f"{result.foreground_percent:.2f}",
        )

    def save_results(self) -> None:
        if self.latest_mask is None or self.image_path is None:
            messagebox.showerror("Error", "Nothing to save yet")
            return
        directory = filedialog.askdirectory(title="Choose an output folder")
        if not directory:
            return
        destination = Path(directory)
        save_mask(self.latest_mask, destination / f"{self.image_path.stem}_mask.png")
        save_overlay(
            self.image_array, self.latest_mask, destination / f"{self.image_path.stem}_overlay.png"
        )
        self._set_status(f"Saved mask and overlay to {destination}", SUCCESS)

    # ----------------------------------------------------------------- charts

    def _draw_empty_charts(self) -> None:
        for axis, title in ((self.ax_time, "Inference time"), (self.ax_score, "Dice / IoU")):
            axis.clear()
            axis.text(
                0.5,
                0.5,
                "Run a comparison\nto see charts",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        self.figure.tight_layout()
        self.canvas_widget.draw()

    def _draw_charts(self, results) -> None:
        names = [r.model_name.upper() for r in results]
        colours = [MODEL_COLOURS.get(r.model_name, "#7f8c8d") for r in results]

        self.ax_time.clear()
        self.ax_time.bar(names, [r.inference_time for r in results], color=colours)
        self.ax_time.set_title("Inference time (s)", fontsize=9)
        self.ax_time.tick_params(axis="x", labelsize=7)
        self.ax_time.tick_params(axis="y", labelsize=7)
        self.ax_time.grid(axis="y", alpha=0.3)

        self.ax_score.clear()
        scored = [(r.model_name.upper(), r.dice, r.iou) for r in results if r.dice is not None]
        if scored:
            positions = np.arange(len(scored))
            width = 0.38
            self.ax_score.bar(
                positions - width / 2, [s[1] for s in scored], width, label="Dice", color="#27ae60"
            )
            self.ax_score.bar(
                positions + width / 2, [s[2] for s in scored], width, label="IoU", color="#3498db"
            )
            self.ax_score.set_xticks(positions)
            self.ax_score.set_xticklabels([s[0] for s in scored], fontsize=7)
            self.ax_score.legend(fontsize=7)
        else:
            self.ax_score.text(
                0.5,
                0.5,
                "Load a ground-truth\nmask for Dice/IoU",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )
            self.ax_score.set_xticks([])
        self.ax_score.set_title("Accuracy vs ground truth", fontsize=9)
        self.ax_score.tick_params(axis="y", labelsize=7)
        self.ax_score.set_ylim(0, 1)
        self.ax_score.grid(axis="y", alpha=0.3)

        self.figure.tight_layout()
        self.canvas_widget.draw()


def launch(config: AppConfig) -> None:
    """Open the GUI window."""
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001 - non-Windows or older Windows
        pass

    root = tk.Tk()
    SegmentationApp(root, config)
    root.mainloop()
