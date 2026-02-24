"""Parser module for meeting minutes input files.

Handles JSONL (Claude Code interaction logs) and plain text formats.
Filters infrastructure noise: tool results, system reminders, teammate
protocol messages, compaction summaries, and other non-conversation content.

All filters are configurable via FilterConfig — pass filter_config=None
to disable all filtering, or customize individual flags.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Message types that are infrastructure, not conversation
DEFAULT_SKIP_TYPES = frozenset({
    "progress",
    "system",
    "file-history-snapshot",
    "queue-operation",
})

# Content block types to drop from message content lists
DEFAULT_SKIP_CONTENT_TYPES = frozenset({
    "tool_use",
    "tool_result",
})

# Regex patterns for inline noise in text content
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL
)
_TEAMMATE_MSG_RE = re.compile(
    r"<teammate-message[^>]*>.*?</teammate-message>", re.DOTALL
)

# Teammate protocol JSON patterns (idle notifications, shutdown, etc.)
_TEAMMATE_PROTOCOL_PATTERNS = (
    '"type":"idle_notification"',
    '"type":"shutdown_approved"',
    '"type":"shutdown_request"',
    '"type":"teammate_terminated"',
    '"type": "idle_notification"',
    '"type": "shutdown_approved"',
    '"type": "shutdown_request"',
    '"type": "teammate_terminated"',
)


@dataclass
class FilterConfig:
    """Configuration for JSONL transcript noise filtering.

    All filters default to True (enabled). Set to False to keep that content.
    Pass filter_config=None to parse_jsonl() to disable ALL filtering.
    """

    skip_message_types: bool = True
    """Drop entire messages with types like progress, system, file-history-snapshot."""

    skip_content_types: bool = True
    """Drop tool_use and tool_result content blocks from messages."""

    strip_system_reminders: bool = True
    """Remove <system-reminder> tags from text content."""

    strip_teammate_tags: bool = True
    """Remove <teammate-message> tags from text content."""

    filter_protocol_messages: bool = True
    """Drop messages that are pure teammate protocol JSON (idle, shutdown)."""

    filter_compaction: bool = True
    """Drop compaction/context compression summary messages."""

    message_types_to_skip: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_TYPES)
    """Which message types to drop (only used when skip_message_types=True)."""

    content_types_to_skip: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_CONTENT_TYPES)
    """Which content block types to drop (only used when skip_content_types=True)."""


# Default config instance — all filters on
DEFAULT_FILTERS = FilterConfig()

# No-filter config — keeps everything
NO_FILTERS = FilterConfig(
    skip_message_types=False,
    skip_content_types=False,
    strip_system_reminders=False,
    strip_teammate_tags=False,
    filter_protocol_messages=False,
    filter_compaction=False,
)


def _strip_inline_noise(text: str, config: FilterConfig) -> str:
    """Remove inline noise tags from text based on config."""
    if config.strip_system_reminders:
        text = _SYSTEM_REMINDER_RE.sub("", text)
    if config.strip_teammate_tags:
        text = _TEAMMATE_MSG_RE.sub("", text)
    return text.strip()


def _is_protocol_message(text: str) -> bool:
    """Check if a message is purely teammate protocol JSON (idle, shutdown, etc.)."""
    stripped = text.strip()
    if not stripped:
        return True
    for pattern in _TEAMMATE_PROTOCOL_PATTERNS:
        if pattern in stripped:
            return True
    return False


def _is_compaction_summary(obj: dict[str, Any]) -> bool:
    """Detect compaction/context compression messages."""
    if obj.get("type") == "system" and obj.get("subtype") == "compact_boundary":
        return True

    message = obj.get("message", {})
    if not isinstance(message, dict):
        return False

    content = message.get("content")
    if isinstance(content, str):
        if "conversation that ran out of context" in content.lower():
            return True
        if "context was compressed" in content.lower():
            return True
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").lower()
                if "conversation that ran out of context" in text:
                    return True
                if "context was compressed" in text:
                    return True

    return False


def parse_jsonl(
    file_path: str,
    filter_config: FilterConfig | None = DEFAULT_FILTERS,
) -> tuple[str, dict[str, Any]]:
    """Parse a JSONL file containing Claude Code interaction logs.

    Args:
        file_path: Path to the JSONL file
        filter_config: Noise filtering configuration. Pass None to disable all
            filtering. Defaults to DEFAULT_FILTERS (all filters enabled).

    Returns:
        Tuple of (consolidated_text, metadata_dict) where metadata_dict contains:
        - "messages": count of extracted messages
        - "filtered": count of messages removed by filters
        - "skipped": count of unparseable lines
        - "format": "jsonl"
    """
    # Use no-filter config if None passed
    config = filter_config if filter_config is not None else NO_FILTERS

    messages: list[str] = []
    bad_lines = 0
    filtered = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            # Skip non-message event types
            msg_type = obj.get("type", "")
            if config.skip_message_types and msg_type in config.message_types_to_skip:
                filtered += 1
                continue

            # Skip compaction summaries
            if config.filter_compaction and _is_compaction_summary(obj):
                filtered += 1
                continue

            # Must have a message dict
            if "message" not in obj:
                filtered += 1
                continue

            message = obj["message"]
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            if role not in ("user", "assistant"):
                continue

            # Extract content - can be string or list of content blocks
            content = message.get("content")
            if content is None:
                continue

            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif config.skip_content_types and block_type in config.content_types_to_skip:
                            continue  # Skip filtered content types
                        else:
                            # Keep unrecognized block types as-is if not filtering
                            if not config.skip_content_types:
                                text_parts.append(json.dumps(block))
                text = "".join(text_parts)

            # Strip inline noise
            text = _strip_inline_noise(text, config)

            # Skip empty messages or pure protocol messages
            if not text or not text.strip():
                filtered += 1
                continue

            if config.filter_protocol_messages and _is_protocol_message(text):
                filtered += 1
                continue

            # Add role label
            label = "User:" if role == "user" else "Assistant:"
            messages.append(f"{label} {text}")

    consolidated_text = "\n\n".join(messages)
    metadata: dict[str, Any] = {
        "messages": len(messages),
        "filtered": filtered,
        "skipped": bad_lines,
        "format": "jsonl",
    }

    return consolidated_text, metadata


def parse_text(file_path: str) -> tuple[str, dict[str, Any]]:
    """Parse a plain text file.

    Args:
        file_path: Path to the text file

    Returns:
        Tuple of (file_contents, metadata_dict) where metadata_dict contains:
        - "format": "text"
        - "chars": character count
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()

    metadata: dict[str, Any] = {
        "format": "text",
        "chars": len(contents),
    }

    return contents, metadata


def parse_file(
    file_path: str,
    filter_config: FilterConfig | None = DEFAULT_FILTERS,
) -> tuple[str, dict[str, Any]]:
    """Auto-detect file format and parse accordingly.

    Supports:
    - .jsonl files (Claude Code interaction logs)
    - .txt, .md, .markdown files (plain text)
    - Unknown extensions: tries JSONL first, falls back to text

    Args:
        file_path: Path to the file
        filter_config: Noise filtering configuration for JSONL files.

    Returns:
        Tuple of (content, metadata_dict)

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return parse_jsonl(file_path, filter_config=filter_config)
    elif suffix in (".txt", ".md", ".markdown"):
        return parse_text(file_path)
    else:
        # Unknown extension: try JSONL first, fall back to text
        try:
            text, metadata = parse_jsonl(file_path, filter_config=filter_config)
            if metadata.get("messages", 0) > 0:
                return text, metadata
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return parse_text(file_path)
