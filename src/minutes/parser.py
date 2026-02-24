"""Parser module for meeting minutes input files.

Handles JSONL (Claude Code interaction logs) and plain text formats.
Delegates noise filtering to the filters module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minutes.filters import (
    DEFAULT_FILTERS,
    NO_FILTERS,
    FilterConfig,
    SYSTEM_REMINDER_RE,
    TEAMMATE_MSG_RE,
    is_compaction_summary,
    is_protocol_message,
    strip_inline_noise,
)

# Re-export for backwards compatibility (changes.py, intent.py import these)
_SYSTEM_REMINDER_RE = SYSTEM_REMINDER_RE
_TEAMMATE_MSG_RE = TEAMMATE_MSG_RE


def parse_jsonl(
    file_path: str,
    filter_config: FilterConfig | None = DEFAULT_FILTERS,
) -> tuple[str, dict[str, Any]]:
    """Parse a JSONL file containing Claude Code interaction logs.

    Returns:
        Tuple of (consolidated_text, metadata_dict)
    """
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

            msg_type = obj.get("type", "")
            if config.skip_message_types and msg_type in config.message_types_to_skip:
                filtered += 1
                continue

            if config.filter_compaction and is_compaction_summary(obj):
                filtered += 1
                continue

            if "message" not in obj:
                filtered += 1
                continue

            message = obj["message"]
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            if role not in ("user", "assistant"):
                continue

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
                            continue
                        else:
                            if not config.skip_content_types:
                                text_parts.append(json.dumps(block))
                text = "".join(text_parts)

            text = strip_inline_noise(text, config)

            if not text or not text.strip():
                filtered += 1
                continue

            if config.filter_protocol_messages and is_protocol_message(text):
                filtered += 1
                continue

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
    """Parse a plain text file."""
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
    """Auto-detect file format and parse accordingly."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return parse_jsonl(file_path, filter_config=filter_config)
    elif suffix in (".txt", ".md", ".markdown"):
        return parse_text(file_path)
    else:
        try:
            text, metadata = parse_jsonl(file_path, filter_config=filter_config)
            if metadata.get("messages", 0) > 0:
                return text, metadata
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return parse_text(file_path)
