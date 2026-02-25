"""LLM extraction pipeline for structured knowledge from transcripts."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import openai

from minutes.config import Config
from minutes.extractor_cleanup import cleanup_result
from minutes.extractor_chunking import chunk_transcript, extract_json_block
from minutes.extractor_dedup import merge_results
from minutes.models import ExtractionResult

logger = logging.getLogger(__name__)


class GatewayBackend:
    """LLM backend using the model gateway."""

    def __init__(self, model: str = "qwen3-4b", base_url: str = "http://localhost:8800/v1") -> None:  # noqa: D102
        self.client = openai.OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:  # noqa: D102
        """Generate text using the model gateway.

        Args:
            system_prompt: System context/instructions
            user_prompt: User query

        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


def get_backend(config: Config) -> tuple[GatewayBackend, str]:  # noqa: D103
    """Get LLM backend via model gateway.

    Auto-starts the gateway if not running.

    Args:
        config: Configuration object with gateway settings

    Returns:
        Tuple of (backend instance, backend_type string)

    Raises:
        RuntimeError: If gateway cannot be started
    """
    try:
        from model_gateway.server import ensure_gateway_running
        base_url = ensure_gateway_running()
    except ImportError:
        # model-gateway not installed — use configured URL directly
        base_url = config.gateway_url
    except Exception as e:
        logger.warning(f"Gateway auto-start failed: {e}, using configured URL")
        base_url = config.gateway_url

    backend = GatewayBackend(model=config.gateway_model, base_url=base_url)
    print(f"✓ Using gateway ({config.gateway_model} via {base_url})")
    return backend, "gateway"


def extract_structured(
    backend: GatewayBackend,
    config: Config,
    transcript: str,
) -> ExtractionResult:  # noqa: D103
    """Extract structured knowledge from transcript using LLM.

    Retries up to config.max_retries times on validation error.
    Returns empty result on final failure.
    """
    schema = ExtractionResult.model_json_schema()

    for attempt in range(config.max_retries):
        try:
            user_prompt = config.extraction_prompt.format(
                schema=json.dumps(schema), transcript=transcript
            )

            response = backend.generate(config.system_prompt, user_prompt)

            json_str = extract_json_block(response)
            data = json.loads(json_str)

            result = ExtractionResult(**data)

            if config.verbose:
                logger.info(
                    f"Extraction successful on attempt {attempt + 1}"
                )

            return result

        except (json.JSONDecodeError, ValueError) as e:
            if config.verbose:
                logger.warning(
                    f"Extraction attempt {attempt + 1} failed: {e}"
                )

            if attempt == config.max_retries - 1:
                if config.verbose:
                    logger.error(
                        f"Extraction failed after {config.max_retries} attempts"
                    )
                return ExtractionResult()

    return ExtractionResult()


def process_transcript(
    backend: GatewayBackend,
    config: Config,
    transcript: str,
    *,
    file_size: int = 0,
    session_id: str = "",
    file_hash: str = "",
    store: Any | None = None,
    on_chunk_done: Callable | None = None,
    on_chunks_ready: Callable[[int, int], None] | None = None,
) -> ExtractionResult:  # noqa: D103
    """Process transcript, chunking if necessary and merging results.

    Args:
        backend: LLM backend for extraction.
        config: Configuration object.
        transcript: Full transcript text.
        file_size: Original file size in bytes (for chunk tier selection).
        session_id: Session ID (for resume support).
        file_hash: File hash (for resume support).
        store: MinutesStore instance (for resume support).
        on_chunk_done: Callback invoked after each chunk extraction.
        on_chunks_ready: Callback(total_chunks, already_completed) once chunk count is known.
    """
    if not transcript:
        return ExtractionResult()

    effective_chunk_size = config.get_chunk_size(file_size) if file_size else config.max_chunk_size

    if len(transcript) <= effective_chunk_size:
        if on_chunks_ready:
            on_chunks_ready(1, 0)
        result = extract_structured(backend, config, transcript)
        if on_chunk_done:
            on_chunk_done()
        return cleanup_result(result, transcript)

    chunks = chunk_transcript(
        transcript, effective_chunk_size, config.chunk_overlap
    )
    total_chunks = len(chunks)

    # Resume: load previously completed chunks
    completed: dict[int, ExtractionResult] = {}
    if store and session_id and file_hash:
        prior = store.get_chunk_progress(session_id, file_hash)
        if prior and prior[0]["total_chunks"] == total_chunks and prior[0]["chunk_size"] == effective_chunk_size:
            for row in prior:
                completed[row["chunk_index"]] = ExtractionResult(**json.loads(row["result_json"]))
            if config.verbose:
                logger.info(f"Resuming: {len(completed)}/{total_chunks} chunks already done")
        elif prior:
            # Chunk params changed (file grew or tier changed) — discard partial
            store.clear_chunk_progress(session_id, file_hash)
            if config.verbose:
                logger.info("Chunk parameters changed, discarding partial progress")

    if on_chunks_ready:
        on_chunks_ready(total_chunks, len(completed))

    results: list[ExtractionResult] = []
    for i, chunk in enumerate(chunks):
        if i in completed:
            results.append(completed[i])
            if on_chunk_done:
                on_chunk_done()
            continue

        if config.verbose:
            logger.info(f"Processing chunk {i + 1}/{total_chunks}")

        result = extract_structured(backend, config, chunk)
        results.append(result)

        # Save chunk progress for resume
        if store and session_id and file_hash:
            store.save_chunk_result(
                session_id, file_hash, i, effective_chunk_size, total_chunks, result,
            )

        if on_chunk_done:
            on_chunk_done()

    # Clear chunk progress on successful completion
    if store and session_id and file_hash:
        store.clear_chunk_progress(session_id, file_hash)

    merged = merge_results(results)
    merged = cleanup_result(merged, transcript)

    if config.verbose:
        logger.info(f"Merged {total_chunks} chunks into final result")

    return merged
