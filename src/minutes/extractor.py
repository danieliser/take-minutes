"""LLM extraction pipeline for structured knowledge from transcripts."""

from __future__ import annotations

import json
import logging

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
) -> ExtractionResult:  # noqa: D103
    """Process transcript, chunking if necessary and merging results."""
    if not transcript:
        return ExtractionResult()

    if len(transcript) <= config.max_chunk_size:
        result = extract_structured(backend, config, transcript)
        return cleanup_result(result, transcript)

    chunks = chunk_transcript(
        transcript, config.max_chunk_size, config.chunk_overlap
    )

    results = []
    for i, chunk in enumerate(chunks):
        if config.verbose:
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")

        result = extract_structured(backend, config, chunk)
        results.append(result)

    merged = merge_results(results)
    merged = cleanup_result(merged, transcript)

    if config.verbose:
        logger.info(f"Merged {len(chunks)} chunks into final result")

    return merged
