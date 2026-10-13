"""Build DINOv2 patch embeddings from data/input/references."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circe.config import load_config, resolve_path
from circe.feature_bank import save_feature_bank
from circe.images import apply_roi, list_images, load_rgb_image, roi_for, sliding_windows
from circe.model import DinoV2Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, root = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    references = resolve_path(root, config["paths"]["references"])
    patches_root = resolve_path(root, config["paths"]["patches"])
    model = DinoV2Encoder(config["model"]["id"], resolve_path(root, config["model"]["cache_dir"]), config["device"])
    logging.info("DINOv2: %s | device: %s | embedding: %d", model.model_id, model.device, model.embedding_dim)
    patch_config = config["patches"]
    pending_images, pending_metadata, vectors, invalid = [], [], [], []

    def flush() -> None:
        if not pending_images:
            return
        vectors.append(model.embed(pending_images))
        pending_images.clear()

    for component in config["components"]:
        for image_path in list_images(references / component):
            relative = image_path.relative_to(references).as_posix()
            try:
                image = load_rgb_image(image_path)
                roi = roi_for(config, "references", f"references/{relative}") or roi_for(config, "references", relative)
                origin_x, origin_y = (int(roi[0]), int(roi[1])) if roi else (0, 0)
                image = apply_roi(image, roi)
                patch_index = 0
                for scale in patch_config["scales"]:
                    for patch, x, y, width, height in sliding_windows(
                        image, int(patch_config["size"]), float(patch_config["overlap"]), float(scale), bool(patch_config["cover_edges"])
                    ):
                        patch_name = f"{image_path.stem}_s{str(scale).replace('.', '_')}_{patch_index:05d}.jpg"
                        destination = patches_root / component / patch_name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        patch.save(destination, quality=95)
                        pending_images.append(patch)
                        pending_metadata.append({
                            "component_id": component, "source_image": relative, "patch_id": f"{component}:{image_path.stem}:{patch_index}",
                            "patch_path": destination.relative_to(root).as_posix(), "x": origin_x + x, "y": origin_y + y,
                            "width": width, "height": height, "scale": float(scale), "model_id": model.model_id,
                            "preprocessing": "AutoImageProcessor associated with model checkpoint",
                        })
                        patch_index += 1
                        if len(pending_images) >= int(config["batch_size"]):
                            flush()
                if patch_index == 0:
                    invalid.append({"path": relative, "reason": "Image/ROI is smaller than the configured patch size."})
            except Exception as exc:  # one iPhone image must not stop a full rebuild
                logging.warning("Skipping %s: %s", image_path.name, exc)
                invalid.append({"path": relative, "reason": str(exc)})
    flush()
    if not pending_metadata:
        raise RuntimeError("No patches were produced. Add reference images or lower patches.size.")
    import numpy as np
    embeddings = np.vstack(vectors)
    bank_path, metadata_path = resolve_path(root, config["paths"]["feature_bank"]), resolve_path(root, config["paths"]["metadata"])
    save_feature_bank(embeddings, pending_metadata, bank_path, metadata_path)
    counts = Counter(row["component_id"] for row in pending_metadata)
    report = {
        "model_id": model.model_id, "device": str(model.device), "embedding_dimension": model.embedding_dim,
        "reference_images": {component: len(list_images(references / component)) for component in config["components"]},
        "patches_and_embeddings": dict(counts), "invalid_images": invalid,
    }
    report_path = resolve_path(root, config["paths"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved %d embeddings to %s", len(pending_metadata), bank_path)
    logging.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
