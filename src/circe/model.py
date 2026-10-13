from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    cuda_ready = torch.cuda.is_available()
    if requested == "cuda" and not cuda_ready:
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and cuda_ready) else "cpu")


class DinoV2Encoder:
    def __init__(self, model_id: str, cache_dir: Path, device_request: str):
        self.device = select_device(device_request)
        cache_dir.mkdir(parents=True, exist_ok=True)
        repository_cache = cache_dir / f"models--{model_id.replace('/', '--')}"
        # After a full first download, load strictly from the project cache. This makes
        # normal runs reproducible and avoids unnecessary network checks/offline delays.
        local_complete = any(repository_cache.glob("snapshots/*/config.json")) and any(
            repository_cache.glob("snapshots/*/model.safetensors")
        )
        load_options = {"cache_dir": str(cache_dir), "local_files_only": local_complete}
        self.processor = AutoImageProcessor.from_pretrained(model_id, **load_options)
        self.model = AutoModel.from_pretrained(model_id, **load_options).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model_id = model_id
        self.embedding_dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixels)
        vectors = outputs.last_hidden_state[:, 0, :]
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().numpy().astype(np.float32)
