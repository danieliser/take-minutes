"""Intent extraction from user prompts in a coding session."""

from __future__ import annotations

import json
import logging
from minutes.reader import JsonlReader
from minutes.models import IntentSummary
from minutes.parser import _SYSTEM_REMINDER_RE, _TEAMMATE_MSG_RE

logger = logging.getLogger(__name__)

INTENT_CHUNK_SIZE = 8000  # Max chars for concatenated prompts sent to LLM


def extract_user_prompts(file_path: str, strict: bool = False) -> list[str]:
    """Extract and clean user prompts from a JSONL transcript.

    Args:
        file_path: Path to JSONL file
        strict: If True, raise on malformed JSON; else skip bad lines

    Returns:
        List of cleaned prompt strings
    """
    reader = JsonlReader(file_path, strict=strict)
    prompts: list[str] = []

    for user_message in reader.user_messages():
        content = user_message.get("content")

        if isinstance(content, str):
            # Simple string content
            text = _clean_prompt(content)
            if text:
                prompts.append(text)
        elif isinstance(content, list):
            # Content is a list of blocks
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                # Skip tool_result blocks
                if block_type == "tool_result":
                    continue

                # Extract text from text blocks
                if block_type == "text":
                    text = block.get("text", "")
                    text = _clean_prompt(text)
                    if text:
                        prompts.append(text)

    return prompts


def _clean_prompt(text: str) -> str:
    """Clean a text prompt by stripping tags and filtering protocol messages.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text, or empty string if it's a protocol message
    """
    # Strip system-reminder tags
    text = _SYSTEM_REMINDER_RE.sub("", text)

    # Strip teammate message tags
    text = _TEAMMATE_MSG_RE.sub("", text)

    text = text.strip()

    if not text:
        return ""

    # Check for protocol messages: JSON with protocol type fields
    if _is_protocol_message(text):
        return ""

    return text


def _is_protocol_message(text: str) -> bool:
    """Check if text is a protocol message (idle notification, shutdown, etc.).

    Args:
        text: Text to check

    Returns:
        True if text is a protocol message
    """
    stripped = text.strip()

    # Protocol message indicators
    protocol_patterns = (
        '"type":"idle_notification"',
        '"type":"shutdown_approved"',
        '"type":"shutdown_request"',
        '"type":"teammate_terminated"',
        '"type": "idle_notification"',
        '"type": "shutdown_approved"',
        '"type": "shutdown_request"',
        '"type": "teammate_terminated"',
    )

    for pattern in protocol_patterns:
        if pattern in stripped:
            return True

    return False


def summarize_intent(backend, prompts: list[str]) -> IntentSummary:
    """Summarize user intent from a list of prompts using LLM.

    Args:
        backend: OpenAI-compatible LLM backend with chat.completions.create()
        prompts: List of user prompt strings

    Returns:
        IntentSummary with extracted goals and constraints
    """
    # Early return for empty prompts
    if not prompts:
        return IntentSummary(prompt_count=0)

    # Concatenate prompts
    concatenated = "\n---\n".join(prompts)

    # Chunk if too large
    if len(concatenated) > INTENT_CHUNK_SIZE:
        concatenated = _chunk_prompts(prompts)

    # Build system prompt
    system_prompt = (
        "Given the following user prompts from a coding session, summarize:\n"
        "1. The primary goal (one sentence)\n"
        "2. Sub-goals or supporting tasks (bullet list)\n"
        "3. Any constraints mentioned (budget, timeline, tech requirements)\n"
        "\n"
        "Return JSON matching this schema:\n"
        '{"primary_goal": "...", "sub_goals": ["..."], "constraints": ["..."]}'
    )

    try:
        # Call backend
        response = backend.chat.completions.create(
            model=backend._model if hasattr(backend, "_model") else "default",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": concatenated},
            ],
            response_format={"type": "json_object"},
        )

        # Parse response
        response_text = response.choices[0].message.content
        data = json.loads(response_text)

        return IntentSummary(
            primary_goal=data.get("primary_goal", ""),
            sub_goals=data.get("sub_goals", []),
            constraints=data.get("constraints", []),
            prompt_count=len(prompts),
        )

    except Exception as e:
        logger.warning(f"Intent summarization failed: {e}")
        return IntentSummary(prompt_count=len(prompts))


def _chunk_prompts(prompts: list[str]) -> str:
    """Reduce prompts to first 3 + last 3 with omission marker.

    Args:
        prompts: List of prompts

    Returns:
        Concatenated string with omission marker
    """
    if len(prompts) <= 6:
        return "\n---\n".join(prompts)

    first_three = prompts[:3]
    last_three = prompts[-3:]
    omitted_count = len(prompts) - 6

    result = first_three + [f"[{omitted_count} prompts omitted]"] + last_three
    return "\n---\n".join(result)


def format_intent_markdown(intent: IntentSummary) -> str:
    """Format IntentSummary as markdown output.

    Args:
        intent: IntentSummary to format

    Returns:
        Markdown-formatted string
    """
    if not intent.primary_goal:
        return "# User Intent Summary\n\nNo intent could be determined from this session.\n"

    lines = [
        "# User Intent Summary",
        "",
        f"**Primary goal**: {intent.primary_goal}",
        "",
    ]

    if intent.sub_goals:
        lines.append("## Sub-goals")
        for sub_goal in intent.sub_goals:
            lines.append(f"- {sub_goal}")
        lines.append("")

    if intent.constraints:
        lines.append("## Constraints")
        for constraint in intent.constraints:
            lines.append(f"- {constraint}")
        lines.append("")

    lines.append(f"**Prompts analyzed**: {intent.prompt_count}")

    return "\n".join(lines) + "\n"
