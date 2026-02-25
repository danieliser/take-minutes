"""Transcript splitting: chunk_transcript, extract_json_block."""

from __future__ import annotations

import json


def extract_json_block(text: str) -> str:  # noqa: D103
    """Extract JSON from LLM response, handling various formats.

    Handles:
    - ```json ... ``` code blocks
    - ``` ... ``` code blocks
    - Raw JSON without code blocks

    Args:
        text: Response text potentially containing JSON

    Returns:
        Extracted JSON string

    Raises:
        json.JSONDecodeError: If extracted text is invalid JSON
    """
    text = text.strip()

    # Try ```json ... ``` code block
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end != -1:
            json_str = text[start:end].strip()
            json.loads(json_str)  # Validate
            return json_str

    # Try ``` ... ``` code block
    if "```" in text:
        start = text.find("```") + len("```")
        end = text.find("```", start)
        if end != -1:
            json_str = text[start:end].strip()
            json.loads(json_str)  # Validate
            return json_str

    # Try raw JSON
    json_str = text
    json.loads(json_str)  # Validate
    return json_str


def chunk_transcript(text: str, max_size: int, overlap: int) -> list[str]:  # noqa: D103
    """Split transcript into overlapping chunks.

    Prefers paragraph boundaries (double newlines) for chunk breaks.

    Args:
        text: Transcript to chunk
        max_size: Target size in characters (approximate)
        overlap: Character overlap between adjacent chunks

    Returns:
        List of text chunks
    """
    if len(text) <= max_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # End position for this chunk
        end = min(start + max_size, len(text))

        # If not at end, try to find paragraph boundary near end
        if end < len(text):
            # Look for paragraph break near the end position
            search_start = max(start, end - max_size // 4)
            para_break = text.rfind("\n\n", search_start, end)

            if para_break != -1 and para_break > start:
                end = para_break + 2  # Include the double newline
            # If no para break in range, use max_size

        chunks.append(text[start:end])

        # Move start position, accounting for overlap
        # Ensure forward progress even with small chunks
        new_start = end - overlap
        if new_start <= start:
            new_start = end  # No overlap if chunk was too small
        start = new_start

    return chunks if chunks else [text]
