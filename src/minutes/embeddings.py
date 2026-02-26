"""Embedding client with tiered models: fast (mxbai) and quality (Qwen3)."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # numpy only needed for vector search features

logger = logging.getLogger(__name__)

# Model registry — name → (hf_id, dims, description)
MODELS = {
    "fast": ("mixedbread-ai/mxbai-embed-large-v1", 1024, "SOTA retrieval, MRL support, 335M params"),
    "quality": ("Qwen/Qwen3-Embedding-0.6B", 1024, "Best-in-class, instruction-aware, 600M params"),
    # Legacy aliases
    "mpnet": ("all-mpnet-base-v2", 768, "Legacy default, 109M params"),
    "minilm": ("all-MiniLM-L6-v2", 384, "Lightweight, 22M params"),
}

DEFAULT_MODEL = "fast"

# Lazy-loaded model cache (supports multiple models loaded simultaneously)
_models: dict[str, Any] = {}


def _get_model(model_key: str = DEFAULT_MODEL) -> Any:  # noqa: D103
    """Lazy-load a sentence-transformers model by key or HF ID."""
    # Resolve alias → HF ID
    if model_key in MODELS:
        hf_id = MODELS[model_key][0]
    else:
        hf_id = model_key

    if hf_id not in _models:
        # Skip HuggingFace update checks — models are cached locally
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install search extras: pip install 'take-minutes[search]'"
            )
        logger.info("Loading embedding model: %s", hf_id)
        _models[hf_id] = SentenceTransformer(hf_id, trust_remote_code=True)

    return _models[hf_id]


def get_dims(model_key: str = DEFAULT_MODEL) -> int:  # noqa: D103
    """Get the embedding dimensions for a model key."""
    if model_key in MODELS:
        return MODELS[model_key][1]
    # Fall back to loading and checking
    model = _get_model(model_key)
    return model.get_sentence_embedding_dimension()


def embed(texts: list[str], model_key: str = DEFAULT_MODEL) -> Any:  # noqa: D103
    """Embed a batch of texts into normalized float32 vectors.

    Args:
        texts: List of strings to embed.
        model_key: Model alias ("fast", "quality", "minilm") or HF model ID.

    Returns:
        2D numpy array of shape (len(texts), dim), float32, L2-normalized.
    """
    if not texts:
        if np is None:
            return []
        return np.array([], dtype=np.float32)

    model = _get_model(model_key)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_one(text: str, model_key: str = DEFAULT_MODEL) -> Any:  # noqa: D103
    """Embed a single text string.

    Returns:
        1D numpy array of shape (dim,), float32, L2-normalized.
    """
    return embed([text], model_key=model_key)[0]
