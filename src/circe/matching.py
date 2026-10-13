from __future__ import annotations

from typing import Any

import numpy as np


def aggregate(scores: np.ndarray, method: str, top_k: int) -> float:
    if scores.size == 0:
        return float("-inf")
    if method == "max":
        return float(scores.max())
    if method == "top_k_mean":
        return float(np.sort(scores)[-min(top_k, scores.size):].mean())
    raise ValueError("method must be max or top_k_mean")


def match_query(
    query_embedding: np.ndarray, bank_embeddings: np.ndarray, metadata: list[dict[str, Any]], components: list[str], method: str, top_k: int
) -> dict[str, Any]:
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if not np.isclose(np.linalg.norm(query_embedding), 1.0, atol=1e-4):
        raise ValueError("Query embedding must be L2-normalized.")
    similarities = bank_embeddings @ query_embedding
    component_scores: dict[str, float] = {}
    best_by_component: dict[str, dict[str, Any] | None] = {}
    for component in components:
        indices = np.array([row["component_id"] == component for row in metadata])
        local_indices = np.where(indices)[0]
        local_scores = similarities[local_indices]
        component_scores[component] = aggregate(local_scores, method, top_k)
        best_by_component[component] = None if not len(local_indices) else {
            "metadata": metadata[int(local_indices[np.argmax(local_scores)])], "score": float(local_scores.max())
        }
    ranking = sorted(component_scores, key=component_scores.get, reverse=True)
    top_indices = np.argsort(similarities)[::-1][: min(3, len(similarities))]
    return {
        "component_scores": component_scores,
        "ranking": ranking,
        "predicted_component": ranking[0] if ranking else None,
        "second_component": ranking[1] if len(ranking) > 1 else None,
        "margin": float(component_scores[ranking[0]] - component_scores[ranking[1]]) if len(ranking) > 1 else None,
        "best_by_component": best_by_component,
        "top_patches": [{"metadata": metadata[int(i)], "score": float(similarities[i])} for i in top_indices],
    }
