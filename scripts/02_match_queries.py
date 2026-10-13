"""Interactively crop one image and match its patch against the feature bank.

Usage:
    python scripts/02_match_queries.py "C:\\path\\to\\image.jpg"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from circe.config import load_config, resolve_path
from circe.feature_bank import load_feature_bank
from circe.matching import match_query
from circe.model import DinoV2Encoder
from circe.visualization import show_prediction_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Original image from which to select the query ROI.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["max", "top_k_mean"], help="Overrides aggregation.method")
    return parser.parse_args()


def select_roi(image: Image.Image, title: str) -> list[int] | None:
    """Display an image and return an interactively selected [left, top, width, height]."""
    selection: list[int] | None = None
    action = {"value": None}
    figure, axis = plt.subplots(figsize=(12, 9))
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_title(f"{title}\nDrag ROI | Enter: use selection | R: clear | Q/Escape: cancel")

    def on_select(start, end) -> None:
        nonlocal selection
        if None in (start.xdata, start.ydata, end.xdata, end.ydata):
            return
        left, right = sorted((start.xdata, end.xdata))
        top, bottom = sorted((start.ydata, end.ydata))
        if right - left >= 1 and bottom - top >= 1:
            selection = [round(left), round(top), round(right - left), round(bottom - top)]

    selector = RectangleSelector(axis, on_select, useblit=True, button=[1], interactive=True)

    def on_key(event) -> None:
        nonlocal selection
        if event.key in {"enter", "return"} and selection:
            action["value"] = "use"
            plt.close(figure)
        elif event.key == "r":
            selection = None
            selector.extents = (0, 0, 0, 0)
            figure.canvas.draw_idle()
        elif event.key in {"q", "escape"}:
            action["value"] = "cancel"
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return selection if action["value"] == "use" else None


def crop_patch_from_roi(image: Image.Image, roi: list[int], patch_size: int) -> tuple[Image.Image, list[int]]:
    """Crop a square inside a large ROI or centered on a smaller ROI."""
    if image.width < patch_size or image.height < patch_size:
        raise ValueError(f"Image {image.size} is smaller than configured patch size {patch_size}.")
    left, top, width, height = roi
    right, bottom = left + width, top + height
    crop_left = round(left + width / 2 - patch_size / 2)
    crop_top = round(top + height / 2 - patch_size / 2)
    if width >= patch_size:
        crop_left = min(max(crop_left, left), right - patch_size)
    if height >= patch_size:
        crop_top = min(max(crop_top, top), bottom - patch_size)
    crop_left = min(max(crop_left, 0), image.width - patch_size)
    crop_top = min(max(crop_top, 0), image.height - patch_size)
    crop = image.crop((crop_left, crop_top, crop_left + patch_size, crop_top + patch_size))
    return crop, [crop_left, crop_top, patch_size, patch_size]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    image_path = args.image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    config, root = load_config(args.config)
    method = args.method or config["aggregation"]["method"]
    patch_size = int(config["patches"]["size"])
    register_heif_opener()
    with Image.open(image_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    roi = select_roi(original, f"{image_path.name} — select the query ROI")
    if roi is None:
        logging.info("No ROI selected: matching cancelled. No files were saved.")
        return
    query, crop_box = crop_patch_from_roi(original, roi, patch_size)
    logging.info("Selected ROI: %s | query patch: %s", roi, crop_box)
    bank, metadata = load_feature_bank(resolve_path(root, config["paths"]["feature_bank"]), resolve_path(root, config["paths"]["metadata"]))
    encoder = DinoV2Encoder(config["model"]["id"], resolve_path(root, config["model"]["cache_dir"]), config["device"])
    if bank.shape[1] != encoder.embedding_dim:
        raise ValueError(f"Feature bank dimension {bank.shape[1]} differs from model dimension {encoder.embedding_dim}.")
    result = match_query(encoder.embed([query])[0], bank, metadata, config["components"], method, int(config["aggregation"]["top_k"]))
    result["method"] = method
    logging.info("Predicted: %s | Top-2: %s | margin: %.4f", result["predicted_component"], result["second_component"], result["margin"])
    show_prediction_figure(query, result, root, ground_truth=None)
    logging.info("No CSV, image or ROI file was saved.")


if __name__ == "__main__":
    main()
