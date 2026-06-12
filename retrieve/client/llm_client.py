"""
LLM Client: Unified interface for LLM backends.

Supports:
- Gemini / OpenAI-compatible APIs
- Any OpenAI-compatible endpoint (e.g., vLLM, Together, etc.)
"""

import json
import base64
import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from pathlib import Path


class BaseLLMClient(ABC):
    """LLM Client base class."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None
    ) -> Dict:
        """
        Send chat request.

        Args:
            messages: Message list, format: [{"role": "user/assistant/system", "content": "..."}]
            images: Optional list of image paths to include in the message

        Returns:
            Dict with keys:
                - "content": str, the model response text
                - "usage": Dict or None, token usage info with keys:
                    - "input_tokens": int
                    - "output_tokens": int
                    - "total_tokens": int
        """
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """
    OpenAI-compatible API client.

    Works with any endpoint that implements the OpenAI Chat Completions API,
    including Gemini (via AI Studio or proxy), vLLM, Together, OpenRouter, etc.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 16384,
        temperature: float = 0.7,
        max_retries: int = 10,
        retry_delay: float = 10.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _encode_image_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def chat(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Send chat request with optional images.

        Returns:
            Dict with "content" and "usage" keys.
        """
        import time as _time

        client = self._get_client()
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        # Check if any message already has multimodal content
        has_multimodal = any(isinstance(msg.get("content"), list) for msg in messages)

        # If images provided and no existing multimodal, convert last user message
        if images and not has_multimodal:
            multimodal_messages = []
            last_user_idx = -1
            for i, msg in enumerate(messages):
                if msg.get("role") == "user":
                    last_user_idx = i

            for i, msg in enumerate(messages):
                if i == last_user_idx and images:
                    text = msg.get("content", "")
                    content = [{"type": "text", "text": text}]
                    for img_path in images:
                        if not Path(img_path).exists():
                            print(f"[WARN] Image not found: {img_path}")
                            continue
                        ext = Path(img_path).suffix.lower().lstrip('.')
                        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                                      "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
                        base64_image = self._encode_image_base64(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{base64_image}"}
                        })
                    multimodal_messages.append({"role": "user", "content": content})
                else:
                    multimodal_messages.append(msg)
            messages = multimodal_messages

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )

                response_text = response.choices[0].message.content
                if response_text and len(response_text.strip()) > 0:
                    if attempt > 0:
                        print(f"[LLM] Got valid output on attempt {attempt + 1}")
                    usage = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    return {"content": response_text, "usage": usage}

                print(f"[LLM] Empty response on attempt {attempt + 1}/{self.max_retries}")

            except Exception as e:
                error_msg = str(e)[:200]
                print(f"[LLM] Error on attempt {attempt + 1}/{self.max_retries}: {error_msg}")

            if attempt < self.max_retries - 1:
                print(f"[LLM] Retrying in {self.retry_delay}s...")
                _time.sleep(self.retry_delay)

        print(f"[LLM] All {self.max_retries} retries exhausted")
        return {"content": "", "usage": None}


# =============================================================================
# Factory function
# =============================================================================

def create_llm_client(
    backend: str = "openai",
    **kwargs
) -> BaseLLMClient:
    """
    Create an LLM client.

    Args:
        backend: Backend type. Use "openai" for any OpenAI-compatible API.
        **kwargs: Backend-specific arguments (model, api_key, base_url, etc.)

    Returns:
        LLM client instance

    Examples:
        # Gemini via Google AI Studio
        client = create_llm_client("openai",
            model="gemini-2.5-pro",
            api_key="YOUR_API_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai")

        # OpenAI
        client = create_llm_client("openai",
            model="gpt-4o",
            api_key="YOUR_API_KEY",
            base_url="https://api.openai.com/v1")

        # vLLM local server
        client = create_llm_client("openai",
            model="Qwen/Qwen3-VL-235B-A22B-Instruct",
            base_url="http://localhost:8000/v1")

        # Any OpenAI-compatible endpoint
        client = create_llm_client("openai",
            model="your-model",
            api_key="YOUR_KEY",
            base_url="https://your-endpoint/v1")
    """
    return OpenAICompatibleClient(**kwargs)
