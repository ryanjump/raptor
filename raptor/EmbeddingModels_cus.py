"""Bailian dense embedding adapter using the OpenAI-compatible API.

Default text-embedding-v4 limits: 8,192 tokens/input, at most 10 inputs/request.
RAPTOR supplies one text per call; document and query vectors use one model
and the same dimension. No input is silently truncated.
"""

import math
import os
from numbers import Real
from threading import BoundedSemaphore

from openai import OpenAI

from .EmbeddingModels import BaseEmbeddingModel


class BailianEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model=None, *, base_url=None, dimensions=None, timeout=60.0,
                 max_retries=2, max_concurrency=4, client=None):
        self.model = (model or os.getenv("DASHSCOPE_EMBEDDING_MODEL")
                      or "text-embedding-v4").strip()
        if not self.model:
            raise ValueError("embedding model must be a non-empty name")
        self._fixed_dimensions = self.model in ("text-embedding-v1", "text-embedding-v2")
        if dimensions is None:
            default_dimensions = "1536" if self._fixed_dimensions else "1024"
            try:
                dimensions = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS") or default_dimensions)
            except ValueError:
                raise ValueError("DASHSCOPE_EMBEDDING_DIMENSIONS must be a positive integer") from None
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        supported = {
            "text-embedding-v1": (1536,),
            "text-embedding-v2": (1536,),
            "text-embedding-v3": (64, 128, 256, 512, 768, 1024),
            "text-embedding-v4": (64, 128, 256, 512, 768, 1024, 1536, 2048),
        }
        if self.model in supported and dimensions not in supported[self.model]:
            raise ValueError(f"Unsupported {self.model} dimensions")
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self.dimensions = dimensions
        self._semaphore = BoundedSemaphore(max_concurrency)
        if client is None:
            api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
            if not api_key:
                raise ValueError("Set DASHSCOPE_API_KEY in the environment before initialization.")
            endpoint = base_url or os.getenv("DASHSCOPE_BASE_URL", "").strip()
            if not endpoint or "{" in endpoint or "}" in endpoint:
                raise ValueError(
                    "Set DASHSCOPE_BASE_URL to your region/workspace's actual "
                    "OpenAI-compatible base URL (ending in /compatible-mode/v1)."
                )
            client = OpenAI(
                api_key=api_key, base_url=endpoint,
                timeout=timeout, max_retries=max_retries,
            )
        self.client = client

    def create_embedding(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        # Retain the official OpenAIEmbeddingModel's newline normalization.
        text = text.replace("\n", " ")
        # RAPTOR creates leaves in parallel. Bound API concurrency without
        # changing tree construction, clustering, or retrieval.
        with self._semaphore:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                encoding_format="float",
                # openai==1.3.3 predates the explicit dimensions keyword.
                extra_body={} if self._fixed_dimensions else {"dimensions": self.dimensions},
            )
        if len(response.data) != 1:
            raise ValueError("Bailian must return exactly one embedding")
        vector = response.data[0].embedding
        if len(vector) != self.dimensions:
            raise ValueError("Bailian returned an unexpected embedding dimension")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               for v in vector):
            raise ValueError("Bailian returned a non-finite or non-numeric embedding")
        if not any(v != 0 for v in vector):
            raise ValueError("Bailian returned a zero embedding; cosine distance is undefined")
        return [float(v) for v in vector]
