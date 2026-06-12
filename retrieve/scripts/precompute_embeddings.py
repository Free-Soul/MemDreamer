"""
Precompute embeddings for all videos in memory_full.

Usage:
  # 8-GPU parallel precompute
  python precompute_embeddings.py --memory-dir <MEMORY_DIR> --num-workers 8
"""

import argparse
import json
import logging
from pathlib import Path
from multiprocessing import Process, Queue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def discover_videos(memory_dir: str, skip: list = None) -> list:
    """Find all videos with phase2+phase3 data."""
    skip = set(skip or [])
    videos = []
    for d in sorted(Path(memory_dir).iterdir()):
        if not d.is_dir():
            continue
        if d.name in skip:
            logger.info(f"Skipping {d.name}")
            continue
        phase2 = d / "phase2_topdown"
        phase3 = d / "phase3"
        if phase2.exists() and phase3.exists():
            videos.append(d.name)
        else:
            logger.warning(f"Missing phase2/phase3 for {d.name}")
    return videos


def precompute_one_video(video_key: str, memory_dir: str, embed_model_path: str, device: str, embedder=None):
    """Precompute and save embeddings for a single video.

    Args:
        embedder: Reusable QwenEmbedder instance. If None, a new one is created (not recommended for batch use).
    """
    import gc
    import torch
    from retrieve.core.memory_base import MemoryBase
    from retrieve.client.embedder import QwenEmbedder

    phase2_dir = str(Path(memory_dir) / video_key / "phase2_topdown")
    phase3_dir = str(Path(memory_dir) / video_key / "phase3")
    emb_path = str(Path(memory_dir) / video_key / "embeddings.json")

    # Check if already done
    if Path(emb_path).exists():
        logger.info(f"[{video_key}] Embeddings already exist, skipping")
        return

    logger.info(f"[{video_key}] Loading memory base on {device}...")
    mb = MemoryBase()
    mb.load(phase2_dir=phase2_dir, phase3_dir=phase3_dir)

    logger.info(f"[{video_key}] Computing embeddings...")
    if embedder is None:
        embedder = QwenEmbedder(model_path=embed_model_path, device=device)
    mb.set_embedder(embedder)
    mb.compute_embeddings()

    mb.save_embeddings(emb_path)
    stats = mb.stats()
    logger.info(f"[{video_key}] Done: {stats['embedded_nodes']} nodes embedded")

    # Free memory
    del mb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def worker(queue: Queue, memory_dir: str, embed_model_path: str, device: str):
    """Worker process that picks videos from queue. Model is loaded once and reused."""
    from retrieve.client.embedder import QwenEmbedder

    logger.info(f"[Worker {device}] Loading embedding model...")
    embedder = QwenEmbedder(model_path=embed_model_path, device=device)

    while True:
        video_key = queue.get()
        if video_key is None:
            break
        try:
            precompute_one_video(video_key, memory_dir, embed_model_path, device, embedder=embedder)
        except Exception as e:
            logger.error(f"[{video_key}] FAILED: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-dir", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--embed-model-path", type=str,
                        default="")
    parser.add_argument("--skip", type=str, nargs="*", default=[])
    args = parser.parse_args()

    videos = discover_videos(args.memory_dir, args.skip)
    logger.info(f"Found {len(videos)} videos to process")

    queue = Queue()
    for v in videos:
        queue.put(v)
    for _ in range(args.num_workers):
        queue.put(None)  # sentinel

    processes = []
    for i in range(args.num_workers):
        device = f"cuda:{i}"
        p = Process(target=worker, args=(queue, args.memory_dir, args.embed_model_path, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("All done!")


if __name__ == "__main__":
    main()