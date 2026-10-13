from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image


def _prediction_figure(query: Image.Image, result: dict[str, Any], root: Path, ground_truth: str | None):
    matches = result["top_patches"]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(query)
    axes[0].set_title(f"Query\nPredetta: {result['predicted_component']}\nGT: {ground_truth or 'n/d'}")
    axes[0].axis("off")
    for axis, item in zip(axes[1:], matches):
        patch_path = root / item["metadata"]["patch_path"]
        with Image.open(patch_path) as patch:
            axis.imshow(patch.convert("RGB"))
        axis.set_title(f"{item['metadata']['component_id']}\ncos={item['score']:.3f}")
        axis.axis("off")
    for axis in axes[1 + len(matches):]:
        axis.axis("off")
    scores = " | ".join(f"{name}: {value:.3f}" for name, value in result["component_scores"].items())
    figure.suptitle(f"{scores}  |  metodo: {result.get('method', '')}", fontsize=10)
    figure.tight_layout()
    return figure


def save_prediction_figure(query: Image.Image, result: dict[str, Any], root: Path, output_path: Path, ground_truth: str | None) -> None:
    figure = _prediction_figure(query, result, root, ground_truth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def show_prediction_figure(query: Image.Image, result: dict[str, Any], root: Path, ground_truth: str | None) -> None:
    """Display a prediction interactively without creating an output file."""
    figure = _prediction_figure(query, result, root, ground_truth)
    plt.show()
    plt.close(figure)
