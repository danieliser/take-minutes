"""Shared noise filtering patterns for JSONL transcript processing.

Provides regex patterns and detection helpers used by parser, changes,
and intent modules to strip infrastructure noise from transcripts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL
)
TEAMMATE_MSG_RE = re.compile(
    r"<teammate-message[^>]*>.*?</teammate-message>", re.DOTALL
)

# Teammate protocol JSON patterns (idle notifications, shutdown, etc.)
TEAMMATE_PROTOCOL_PATTERNS = (
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
    skip_content_types: bool = True
    strip_system_reminders: bool = True
    strip_teammate_tags: bool = True
    filter_protocol_messages: bool = True
    filter_compaction: bool = True
    message_types_to_skip: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_TYPES)
    content_types_to_skip: frozenset[str] = field(default_factory=lambda: DEFAULT_SKIP_CONTENT_TYPES)


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


def strip_inline_noise(text: str, config: FilterConfig) -> str:
    """Remove inline noise tags from text based on config."""
    if config.strip_system_reminders:
        text = SYSTEM_REMINDER_RE.sub("", text)
    if config.strip_teammate_tags:
        text = TEAMMATE_MSG_RE.sub("", text)
    return text.strip()


def is_protocol_message(text: str) -> bool:
    """Check if a message is purely teammate protocol JSON."""
    stripped = text.strip()
    if not stripped:
        return True
    for pattern in TEAMMATE_PROTOCOL_PATTERNS:
        if pattern in stripped:
            return True
    return False


def is_compaction_summary(obj: dict[str, Any]) -> bool:
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
