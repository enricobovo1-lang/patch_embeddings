"""Evaluate max and top-k-mean aggregation using query folder names as ground truth."""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from circe.config import load_config, resolve_path
from circe.feature_bank import load_feature_bank
from circe.images import apply_roi, list_images, load_rgb_image, roi_for
from circe.matching import match_query
from circe.model import DinoV2Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--methods", nargs="+", choices=["max", "top_k_mean"], default=["max", "top_k_mean"])
    return parser.parse_args()


def save_graphs(records: list[dict], components: list[str], method: str, output: Path) -> None:
    matrix = np.zeros((len(components), len(components)), dtype=int)
    positions = {name: index for index, name in enumerate(components)}
    for row in records:
        matrix[positions[row["ground_truth"]], positions[row["predicted_component"]]] += 1
    fig, axis = plt.subplots(figsize=(6, 5))
    view = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(view, ax=axis)
    axis.set(xticks=range(len(components)), yticks=range(len(components)), xticklabels=components, yticklabels=components,
             xlabel="Predetto", ylabel="Ground truth", title=f"Confusion matrix — {method}")
    for row in range(len(components)):
        for column in range(len(components)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    fig.tight_layout(); fig.savefig(output / f"confusion_matrix_{method}.png", dpi=150); plt.close(fig)
    correct, incorrect, margins = [], [], []
    for row in records:
        correct.append(float(row[f"score_{row['ground_truth']}"]))
        incorrect.extend(float(row[f"score_{component}"]) for component in components if component != row["ground_truth"])
        margins.append(float(row["margin"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([correct, incorrect], bins=15, label=["componente corretto", "componenti errati"], alpha=0.75)
    axes[0].set(title="Distribuzione score", xlabel="cosine aggregata"); axes[0].legend()
    axes[1].hist(margins, bins=15, color="tab:purple")
    axes[1].set(title="Distribuzione margini Top-1/Top-2", xlabel="margine")
    fig.tight_layout(); fig.savefig(output / f"score_distributions_{method}.png", dpi=150); plt.close(fig)


def main() -> None:
    args = parse_args()
    config, root = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output = resolve_path(root, config["paths"]["evaluations"]); output.mkdir(parents=True, exist_ok=True)
    bank, metadata = load_feature_bank(resolve_path(root, config["paths"]["feature_bank"]), resolve_path(root, config["paths"]["metadata"]))
    encoder = DinoV2Encoder(config["model"]["id"], resolve_path(root, config["model"]["cache_dir"]), config["device"])
    queries = resolve_path(root, config["paths"]["queries"])
    query_embeddings: list[tuple[str, str, np.ndarray]] = []
    for ground_truth in config["components"]:
        for path in list_images(queries / ground_truth):
            relative = path.relative_to(queries).as_posix()
            try:
                image = load_rgb_image(path)
                image = apply_roi(image, roi_for(config, "queries", f"queries/{relative}") or roi_for(config, "queries", relative))
                query_embeddings.append((relative, ground_truth, encoder.embed([image])[0]))
            except Exception as exc:
                logging.error("Skipping %s: %s", relative, exc)
    if not query_embeddings:
        raise RuntimeError("No evaluable query images found.")
    metric_rows = []
    for method in args.methods:
        records = []
        for relative, truth, embedding in query_embeddings:
            result = match_query(embedding, bank, metadata, config["components"], method, int(config["aggregation"]["top_k"]))
            records.append({"query_path": relative, "ground_truth": truth, "predicted_component": result["predicted_component"],
                            "top_2_component": result["second_component"], "margin": result["margin"], "method": method,
                            **{f"score_{name}": score for name, score in result["component_scores"].items()}})
        with (output / f"predictions_{method}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        accuracy = sum(row["ground_truth"] == row["predicted_component"] for row in records) / len(records)
        metric_rows.append({"method": method, "metric": "top1_accuracy", "component": "overall", "value": accuracy, "n_queries": len(records)})
        for component in config["components"]:
            subset = [row for row in records if row["ground_truth"] == component]
            metric_rows.append({"method": method, "metric": "accuracy_per_component", "component": component,
                                "value": sum(r["predicted_component"] == component for r in subset) / len(subset), "n_queries": len(subset)})
        save_graphs(records, config["components"], method, output)
        logging.info("%s Top-1 accuracy: %.3f (%d queries)", method, accuracy, len(records))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)


if __name__ == "__main__":
    main()
