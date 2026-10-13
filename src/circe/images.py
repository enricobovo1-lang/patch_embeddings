from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_rgb_image(path: Path) -> Image.Image:
    """Open a supported image and normalize its EXIF orientation."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def apply_roi(image: Image.Image, roi: list[int] | tuple[int, int, int, int] | None) -> Image.Image:
    if roi is None:
        return image
    if len(roi) != 4:
        raise ValueError("ROI must have [left, top, width, height].")
    left, top, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0 or left < 0 or top < 0:
        raise ValueError(f"Invalid ROI: {roi}")
    right, bottom = left + width, top + height
    if right > image.width or bottom > image.height:
        raise ValueError(f"ROI {roi} is outside image dimensions {image.size}.")
    return image.crop((left, top, right, bottom))


def roi_for(config: dict, group: str, relative_path: str) -> list[int] | None:
    values = config.get("roi", {}).get(group, {})
    return values.get(relative_path.replace("\\", "/"))


def sliding_windows(
    image: Image.Image, patch_size: int, overlap: float, scale: float, cover_edges: bool
) -> Iterator[tuple[Image.Image, int, int, int, int]]:
    """Yield only complete windows; optional edge alignment preserves full coverage."""
    if patch_size <= 0 or not 0 <= overlap < 1 or scale <= 0:
        raise ValueError("patch_size > 0, 0 <= overlap < 1 and scale > 0 are required.")
    scaled_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(scaled_size, Image.Resampling.LANCZOS) if scale != 1 else image
    if resized.width < patch_size or resized.height < patch_size:
        return
    stride = max(1, round(patch_size * (1 - overlap)))
    xs = list(range(0, resized.width - patch_size + 1, stride))
    ys = list(range(0, resized.height - patch_size + 1, stride))
    if cover_edges:
        if xs[-1] != resized.width - patch_size:
            xs.append(resized.width - patch_size)
        if ys[-1] != resized.height - patch_size:
            ys.append(resized.height - patch_size)
    for y in ys:
        for x in xs:
            patch = resized.crop((x, y, x + patch_size, y + patch_size))
            # Coordinates are mapped to the ROI/original image coordinate system.
            yield patch, round(x / scale), round(y / scale), round(patch_size / scale), round(patch_size / scale)
