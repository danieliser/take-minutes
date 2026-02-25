"""Intent extraction from user prompts in a coding session."""

from __future__ import annotations

import json
import logging

from minutes.filters import SYSTEM_REMINDER_RE, TEAMMATE_MSG_RE, is_protocol_message

# Re-export under original private name for test compatibility
_is_protocol_message = is_protocol_message
from minutes.intent_format import format_intent_markdown  # noqa: F401 — re-export
from minutes.models import IntentSummary
from minutes.reader import JsonlReader

logger = logging.getLogger(__name__)

INTENT_CHUNK_SIZE = 8000  # Max chars for concatenated prompts sent to LLM


def extract_user_prompts(file_path: str, strict: bool = False) -> list[str]:
    """Extract and clean user prompts from a JSONL transcript."""
    reader = JsonlReader(file_path, strict=strict)
    prompts: list[str] = []

    for user_message in reader.user_messages():
        content = user_message.get("content")

        if isinstance(content, str):
            text = _clean_prompt(content)
            if text:
                prompts.append(text)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")

                if block_type == "tool_result":
                    continue

                if block_type == "text":
                    text = block.get("text", "")
                    text = _clean_prompt(text)
                    if text:
                        prompts.append(text)

    return prompts


def _clean_prompt(text: str) -> str:
    """Clean a text prompt by stripping tags and filtering protocol messages."""
    text = SYSTEM_REMINDER_RE.sub("", text)
    text = TEAMMATE_MSG_RE.sub("", text)
    text = text.strip()

    if not text:
        return ""

    if is_protocol_message(text):
        return ""

    return text


def summarize_intent(backend, prompts: list[str]) -> IntentSummary:
    """Summarize user intent from a list of prompts using LLM."""
    if not prompts:
        return IntentSummary(prompt_count=0)

    concatenated = "\n---\n".join(prompts)

    if len(concatenated) > INTENT_CHUNK_SIZE:
        concatenated = _chunk_prompts(prompts)

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
        # Support both raw OpenAI client and GatewayBackend wrapper
        client = getattr(backend, 'client', backend)
        model = getattr(backend, 'model', "default")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": concatenated},
            ],
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content
        # Strip markdown code fences if present
        from minutes.extractor_chunking import extract_json_block
        json_text = extract_json_block(response_text) or response_text
        data = json.loads(json_text)

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
    """Reduce prompts to first 3 + last 3 with omission marker."""
    if len(prompts) <= 6:
        return "\n---\n".join(prompts)

    first_three = prompts[:3]
    last_three = prompts[-3:]
    omitted_count = len(prompts) - 6

    result = first_three + [f"[{omitted_count} prompts omitted]"] + last_three
    return "\n---\n".join(result)
