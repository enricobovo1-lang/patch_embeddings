"""Convert HEIC/HEIF images into a sibling <source>_jpg directory.

Usage:
    python scripts/convert_heic_to_jpeg.py "C:\\path\\to\\iphone_photos"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

HEIC_SUFFIXES = {".heic", ".heif"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_folder", type=Path, help="Folder containing HEIC/HEIF images.")
    args = parser.parse_args()
    source = args.source_folder.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"Folder not found: {source}")
    destination = source.parent / f"{source.name}_jpg"
    register_heif_opener()
    inputs = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in HEIC_SUFFIXES)
    if not inputs:
        print(f"No HEIC/HEIF images found in: {source}")
        return
    converted, failures = 0, []
    for image_path in inputs:
        target = (destination / image_path.relative_to(source)).with_suffix(".jpg")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.save(target, "JPEG", quality=95, optimize=True)
            converted += 1
        except Exception as exc:
            failures.append((image_path, str(exc)))
            print(f"ERROR: {image_path.name}: {exc}", file=sys.stderr)
    print(f"Converted {converted}/{len(inputs)} images to: {destination}")
    if failures:
        raise SystemExit(f"{len(failures)} image(s) could not be converted.")


if __name__ == "__main__":
    main()
