"""
Qwen3-Embedding adapter for MemoryBase semantic search.

Wraps SentenceTransformer (Qwen3-Embedding-8B) to provide
embed() / embed_batch() interface expected by MemoryBase.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = ""


class QwenEmbedder:
    """Lightweight text embedder backed by Qwen3-Embedding-8B."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cuda",
        batch_size: int = 64,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.model = None

    def _ensure_model(self):
        if self.model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        logger.info(f"Loading Qwen3-Embedding from {self.model_path} ...")
        self.model = SentenceTransformer(
            self.model_path,
            device=self.device,
            model_kwargs={
                "attn_implementation": attn_impl,
                "torch_dtype": torch.bfloat16,
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        logger.info(f"Qwen3-Embedding loaded on {self.device}")

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed a single text string. Returns list of floats or None."""
        self._ensure_model()
        if not text or not text.strip():
            return None

        import torch
        with torch.no_grad():
            vec = self.model.encode(
                [text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return vec[0].astype(np.float32).tolist()

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed a batch of text strings. Returns list of float-lists (None for empty)."""
        self._ensure_model()

        # Separate empty / non-empty
        results: List[Optional[List[float]]] = [None] * len(texts)
        valid_indices = []
        valid_texts = []
        for i, t in enumerate(texts):
            if t and t.strip():
                valid_indices.append(i)
                valid_texts.append(t)

        if not valid_texts:
            return results

        logger.info(f"Encoding {len(valid_texts)} texts in batches of {self.batch_size} ...")

        import torch
        with torch.no_grad():
            vecs = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

        for idx, vec in zip(valid_indices, vecs):
            results[idx] = vec.tolist()

        return results