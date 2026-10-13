"""Download/load the checkpoint and verify DINOv2 inference with an artificial image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from circe.config import load_config, resolve_path
from circe.model import DinoV2Encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Temporary device override for this verification.")
    parser.add_argument("--report", help="Optional JSON report path, relative to the repository root.")
    args = parser.parse_args()
    config, root = load_config(args.config)
    encoder = DinoV2Encoder(config["model"]["id"], resolve_path(root, config["model"]["cache_dir"]), args.device or config["device"])
    artificial = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    vector = encoder.embed([artificial])[0]
    assert vector.shape == (encoder.embedding_dim,), vector.shape
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5), np.linalg.norm(vector)
    result = {"model_id": encoder.model_id, "device": str(encoder.device), "embedding_dimension": encoder.embedding_dim,
              "embedding_l2_norm": float(np.linalg.norm(vector)), "input": "artificial black RGB image (224x224)"}
    if args.report:
        report_path = Path(args.report)
        report_path = report_path if report_path.is_absolute() else root / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"OK — model={encoder.model_id}; device={encoder.device}; embedding_dim={encoder.embedding_dim}; L2={np.linalg.norm(vector):.6f}")


if __name__ == "__main__":
    main()
