"""
RemoteEmbedder: HTTP client for embedding servers.

Routes embed() calls to one of N embedding servers (round-robin).
"""

import logging
import threading
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class RemoteEmbedder:
    """Embedder that calls remote embedding servers via HTTP."""

    def __init__(self, server_urls: List[str], timeout: int = 30):
        """
        Args:
            server_urls: List of embedding server URLs, e.g. ["http://localhost:8001", ...]
            timeout: Request timeout in seconds
        """
        self.server_urls = server_urls
        self.timeout = timeout
        self._counter = 0
        self._lock = threading.Lock()

    def _next_url(self) -> str:
        with self._lock:
            url = self.server_urls[self._counter % len(self.server_urls)]
            self._counter += 1
            return url

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed a single text string."""
        if not text or not text.strip():
            return None

        url = self._next_url()
        try:
            resp = requests.post(
                f"{url}/embed",
                json={"texts": [text]},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data["embeddings"][0]
            return emb
        except Exception as e:
            logger.warning(f"Embedding server error ({url}): {e}")
            # Try other servers
            for alt_url in self.server_urls:
                if alt_url == url:
                    continue
                try:
                    resp = requests.post(
                        f"{alt_url}/embed",
                        json={"texts": [text]},
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["embeddings"][0]
                except Exception:
                    continue
            logger.error("All embedding servers failed")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed a batch of text strings."""
        if not texts:
            return []

        url = self._next_url()
        try:
            resp = requests.post(
                f"{url}/embed",
                json={"texts": texts},
                timeout=max(self.timeout, len(texts) * 2),
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
        except Exception as e:
            logger.warning(f"Embedding server batch error ({url}): {e}, falling back to per-item")
            return [self.embed(t) for t in texts]