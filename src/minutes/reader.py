"""Shared JSONL reader for extraction layers.

Provides a thin abstraction over JSONL line iteration with error handling.
Used by changes.py, intent.py, and stats collection — prevents duplicated
JSONL walking logic across extraction modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

MAX_ERRORS = 100


class JsonlParseError(Exception):
    """Raised on malformed JSONL in strict mode."""

    def __init__(self, file_path: str, line_number: int, error: str):
        self.file_path = file_path
        self.line_number = line_number
        super().__init__(f"{file_path}:{line_number}: {error}")


class JsonlReader:
    """Low-level JSONL line iterator with error handling.

    Yields parsed message dicts from a JSONL transcript file.
    Handles malformed lines according to strict mode.
    """

    def __init__(self, file_path: str, strict: bool = False):
        self.file_path = file_path
        self.strict = strict
        self.errors: list[str] = []

    def messages(self) -> Iterator[dict]:
        """Yield message dicts from JSONL lines.

        On malformed JSON:
          - strict=True: raise JsonlParseError
          - strict=False: log to self.errors (capped at MAX_ERRORS), skip line
        """
        path = Path(self.file_path)
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    if self.strict:
                        raise JsonlParseError(
                            self.file_path, line_number, str(e)
                        ) from e
                    if len(self.errors) < MAX_ERRORS:
                        self.errors.append(
                            f"Line {line_number}: {e}"
                        )
                    continue

                message = obj.get("message")
                if message and isinstance(message, dict):
                    yield message

    def assistant_messages(self) -> Iterator[dict]:
        """Yield only assistant message dicts."""
        for msg in self.messages():
            if msg.get("role") == "assistant":
                yield msg

    def user_messages(self) -> Iterator[dict]:
        """Yield only user message dicts."""
        for msg in self.messages():
            if msg.get("role") == "user":
                yield msg

    def content_blocks(
        self, role: str = "assistant"
    ) -> Iterator[tuple[dict, list[dict]]]:
        """Yield (message, content_blocks) for messages of given role.

        Only yields messages where content is a list (skips string content).
        """
        for msg in self.messages():
            if msg.get("role") != role:
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                yield msg, content
