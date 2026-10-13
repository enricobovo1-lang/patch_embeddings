"""Interactively select ROIs for all images in a reference or query folder.

Usage:
    python scripts/select_rois.py data/input/references/component_A

Controls in the image window:
    drag mouse  draw or replace the ROI
    Enter       save ROI and advance
    N           skip/remove ROI for this image and advance
    R           clear the current selection
    Q / Escape  save selections made so far and quit
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
import yaml

from circe.config import load_config, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def usage_error(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def image_group(folder: Path, config: dict, root: Path) -> tuple[str, Path]:
    for group in ("references", "queries"):
        base = resolve_path(root, config["paths"][group]).resolve()
        try:
            folder.relative_to(base)
            return group, base
        except ValueError:
            continue
    usage_error("The folder must be inside data/input/references or data/input/queries.")


def write_rois(path: Path, rois: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Managed by scripts/select_rois.py. Coordinates are [left, top, width, height].\n"
        + yaml.safe_dump({"roi": rois}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        usage_error("Usage: python scripts/select_rois.py <image_folder>")
    folder = Path(sys.argv[1]).expanduser().resolve()
    if not folder.is_dir():
        usage_error(f"Folder not found: {folder}")
    config, root = load_config("configs/default.yaml")
    group, base = image_group(folder, config, root)
    images = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        usage_error(f"No supported images found in: {folder}")
    roi_path = resolve_path(root, config.get("roi_file", "configs/rois.yaml"))
    rois = {"references": {}, "queries": {}}
    if roi_path.exists():
        rois = (yaml.safe_load(roi_path.read_text(encoding="utf-8")) or {}).get("roi", rois)
        rois.setdefault("references", {}); rois.setdefault("queries", {})
    register_heif_opener()
    print("Controls: drag=ROI, Enter=save/next, N=skip, R=clear, Q/Escape=quit.")

    for index, image_path in enumerate(images, start=1):
        key = image_path.relative_to(base).as_posix()
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        selection: list[int] | None = rois[group].get(key)
        action = {"value": None}
        figure, axis = plt.subplots(figsize=(12, 9))
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(f"{index}/{len(images)} — {key}\nDrag ROI | Enter save/next | N skip | R clear | Q quit")

        def on_select(start, end) -> None:
            nonlocal selection
            left, right = sorted((start.xdata, end.xdata))
            top, bottom = sorted((start.ydata, end.ydata))
            if right - left >= 1 and bottom - top >= 1:
                selection = [round(left), round(top), round(right - left), round(bottom - top)]

        selector = RectangleSelector(axis, on_select, useblit=True, button=[1], interactive=True)
        if selection:
            selector.extents = (selection[0], selection[0] + selection[2], selection[1], selection[1] + selection[3])

        def on_key(event) -> None:
            nonlocal selection
            if event.key in {"enter", "return"} and selection:
                action["value"] = "save"; plt.close(figure)
            elif event.key == "n":
                action["value"] = "skip"; plt.close(figure)
            elif event.key == "r":
                selection = None; selector.set_active(False); selector.set_active(True); figure.canvas.draw_idle()
            elif event.key in {"q", "escape"}:
                action["value"] = "quit"; plt.close(figure)

        figure.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        if action["value"] == "save":
            rois[group][key] = selection
            write_rois(roi_path, rois)
            print(f"Saved {group}/{key}: {selection}")
        elif action["value"] == "skip":
            rois[group].pop(key, None)
            write_rois(roi_path, rois)
            print(f"No ROI for {group}/{key}")
        elif action["value"] == "quit":
            write_rois(roi_path, rois)
            print(f"Saved selections to: {roi_path}")
            return
        else:
            # Closing the window with its close button is equivalent to quitting safely.
            write_rois(roi_path, rois)
            print(f"Stopped. Selections saved to: {roi_path}")
            return
    print(f"Completed. Selections saved to: {roi_path}")


if __name__ == "__main__":
    main()
