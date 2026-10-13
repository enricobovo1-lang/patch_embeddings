from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def save_feature_bank(embeddings: np.ndarray, metadata: list[dict[str, Any]], bank_path: Path, metadata_path: Path) -> None:
    if embeddings.ndim != 2 or len(embeddings) != len(metadata):
        raise ValueError("Feature rows and metadata rows must have the same length.")
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bank_path, embeddings=embeddings.astype(np.float32))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if metadata:
        with metadata_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)


def load_feature_bank(bank_path: Path, metadata_path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not bank_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Feature bank not found. Run scripts/01_build_feature_bank.py first.")
    with np.load(bank_path) as payload:
        embeddings = payload["embeddings"].astype(np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata):
        raise ValueError("Feature bank and metadata are not aligned.")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Feature bank has non-normalized embeddings.")
    return embeddings, metadata
