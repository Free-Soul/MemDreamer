"""
Embedding Server: FastAPI server for Qwen3-Embedding-8B

One server per GPU, provides POST /embed endpoint.
Usage: python embedding_server.py --port 8001 --device cuda:0
"""

import argparse
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model reference
_model = None
_device = None
_model_path = None


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[Optional[List[float]]]
    dim: int
    elapsed_ms: float


@asynccontextmanager
async def lifespan(app):
    global _model
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading Qwen3-Embedding on {_device} ...")

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        logger.info("flash_attn not installed, falling back to SDPA")

    _model = SentenceTransformer(
        _model_path,
        device=_device,
        model_kwargs={
            "attn_implementation": attn_impl,
            "torch_dtype": torch.bfloat16,
        },
        tokenizer_kwargs={"padding_side": "left"},
    )
    logger.info(f"Qwen3-Embedding loaded on {_device}")
    yield
    logger.info("Shutting down embedding server")


app = FastAPI(lifespan=lifespan)


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    start = time.time()
    texts = req.texts

    results: List[Optional[List[float]]] = [None] * len(texts)
    valid_indices = []
    valid_texts = []
    for i, t in enumerate(texts):
        if t and t.strip():
            valid_indices.append(i)
            valid_texts.append(t)

    if valid_texts:
        vecs = _model.encode(
            valid_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        for idx, vec in zip(valid_indices, vecs):
            results[idx] = vec.tolist()

    dim = len(results[0]) if results and results[0] is not None else 0
    elapsed = (time.time() - start) * 1000
    return EmbedResponse(embeddings=results, dim=dim, elapsed_ms=elapsed)


@app.get("/health")
async def health():
    return {"status": "ok", "device": _device}


def main():
    global _device, _model_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to embedding model (e.g. Qwen3-Embedding-8B)")
    args = parser.parse_args()
    _device = args.device
    _model_path = args.model_path

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()