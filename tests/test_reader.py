"""Tests for the JSONL reader module."""

import json
import tempfile
from pathlib import Path

import pytest

from minutes.reader import JsonlParseError, JsonlReader, MAX_ERRORS


def _write_jsonl(path, lines):
    """Write a list of dicts as JSONL to the given path."""
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


class TestMessagesYieldsValidMessages:
    """Test that messages() yields parsed message dicts from valid JSONL."""

    def test_messages_yields_parsed_message_dicts(self, tmp_path):
        """Verify messages() yields correct message dicts from valid JSONL."""
        # Create a temp JSONL file with valid messages
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help?",
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": "What is the weather?",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "I don't have access to weather data.",
                }
            },
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        messages = list(reader.messages())

        assert len(messages) == 3
        assert messages[0] == {"role": "assistant", "content": "Hello, how can I help?"}
        assert messages[1] == {"role": "user", "content": "What is the weather?"}
        assert messages[2] == {"role": "assistant", "content": "I don't have access to weather data."}


class TestMalformedLineInLenientMode:
    """Test that malformed lines are skipped in lenient mode and logged."""

    def test_malformed_line_skipped_in_lenient_mode(self, tmp_path):
        """Verify malformed line is skipped, valid messages still yielded, error is logged."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "assistant",
                    "content": "First message",
                }
            },
            "this is not valid json at all {{{",
            {
                "message": {
                    "role": "user",
                    "content": "Second message",
                }
            },
        ]
        # Write with malformed line
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(lines[0]) + "\n")
            f.write(lines[1] + "\n")  # Write as-is, not JSON
            f.write(json.dumps(lines[2]) + "\n")

        reader = JsonlReader(str(jsonl_file), strict=False)
        messages = list(reader.messages())

        # Should have exactly 2 valid messages
        assert len(messages) == 2
        assert messages[0] == {"role": "assistant", "content": "First message"}
        assert messages[1] == {"role": "user", "content": "Second message"}

        # Should have one error
        assert len(reader.errors) == 1
        assert "Line 2:" in reader.errors[0]


class TestMalformedLineInStrictMode:
    """Test that malformed lines raise JsonlParseError in strict mode."""

    def test_malformed_line_raises_in_strict_mode(self, tmp_path):
        """Verify JsonlParseError raised with correct file path and line number."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "assistant",
                    "content": "First message",
                }
            },
            "this is not valid json at all {{{",
            {
                "message": {
                    "role": "user",
                    "content": "Second message",
                }
            },
        ]
        # Write with malformed line
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(lines[0]) + "\n")
            f.write(lines[1] + "\n")  # Write as-is, not JSON
            f.write(json.dumps(lines[2]) + "\n")

        reader = JsonlReader(str(jsonl_file), strict=True)

        with pytest.raises(JsonlParseError) as exc_info:
            list(reader.messages())

        error = exc_info.value
        assert error.file_path == str(jsonl_file)
        assert error.line_number == 2
        assert ":2:" in str(error)


class TestErrorsListCappedAtMaxErrors:
    """Test that errors list is capped at MAX_ERRORS (100)."""

    def test_errors_capped_at_max_errors(self, tmp_path):
        """Verify errors list has exactly 100 entries when >100 malformed lines present."""
        jsonl_file = tmp_path / "messages.jsonl"

        # Create JSONL with >100 malformed lines (alternating to ensure we have >100 errors)
        with open(jsonl_file, "w") as f:
            for i in range(250):
                # Write malformed line
                f.write(f"malformed line {i} {{{{\n")

        reader = JsonlReader(str(jsonl_file), strict=False)
        list(reader.messages())

        # Should have exactly MAX_ERRORS entries (100)
        assert len(reader.errors) == MAX_ERRORS
        assert MAX_ERRORS == 100


class TestAssistantMessagesFilter:
    """Test that assistant_messages() filters correctly."""

    def test_assistant_messages_filters_correctly(self, tmp_path):
        """Verify only assistant messages are yielded."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "user",
                    "content": "Hello",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Hi there",
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": "How are you?",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well",
                }
            },
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        assistant_msgs = list(reader.assistant_messages())

        assert len(assistant_msgs) == 2
        assert assistant_msgs[0] == {"role": "assistant", "content": "Hi there"}
        assert assistant_msgs[1] == {"role": "assistant", "content": "I'm doing well"}


class TestUserMessagesFilter:
    """Test that user_messages() filters correctly."""

    def test_user_messages_filters_correctly(self, tmp_path):
        """Verify only user messages are yielded."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "user",
                    "content": "Hello",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Hi there",
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": "How are you?",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well",
                }
            },
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        user_msgs = list(reader.user_messages())

        assert len(user_msgs) == 2
        assert user_msgs[0] == {"role": "user", "content": "Hello"}
        assert user_msgs[1] == {"role": "user", "content": "How are you?"}


class TestContentBlocksYieldsListContent:
    """Test that content_blocks() yields only messages with list content."""

    def test_content_blocks_yields_list_content_messages(self, tmp_path):
        """Verify only messages with list content are yielded as (message, blocks) tuples."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "assistant",
                    "content": "String content",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Block 1"},
                        {"type": "text", "text": "Block 2"},
                    ],
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": "url"},
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "1"},
                    ],
                }
            },
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        content_blocks = list(reader.content_blocks(role="assistant"))

        # Should have exactly 2 assistant messages with list content
        assert len(content_blocks) == 2

        msg1, blocks1 = content_blocks[0]
        assert msg1["role"] == "assistant"
        assert len(blocks1) == 2
        assert blocks1[0] == {"type": "text", "text": "Block 1"}

        msg2, blocks2 = content_blocks[1]
        assert msg2["role"] == "assistant"
        assert len(blocks2) == 1
        assert blocks2[0] == {"type": "tool_use", "id": "1"}


class TestContentBlocksWithDifferentRoles:
    """Test that content_blocks() filters by role correctly."""

    def test_content_blocks_filters_by_role(self, tmp_path):
        """Verify content_blocks() only yields messages with specified role."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Assistant block"}],
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "User block"}],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Another assistant block"}],
                }
            },
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        user_blocks = list(reader.content_blocks(role="user"))

        # Should have only 1 user message with list content
        assert len(user_blocks) == 1
        msg, blocks = user_blocks[0]
        assert msg["role"] == "user"
        assert blocks[0]["text"] == "User block"


class TestEmptyFile:
    """Test that empty file returns no messages."""

    def test_empty_file_returns_no_messages(self, tmp_path):
        """Verify empty file yields no messages."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        reader = JsonlReader(str(jsonl_file))
        messages = list(reader.messages())

        assert len(messages) == 0


class TestFileWithOnlyNonMessageLines:
    """Test that file with only non-message lines returns no messages."""

    def test_file_with_only_non_message_lines(self, tmp_path):
        """Verify file with only non-message JSON returns no messages."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {"other_key": "value1"},
            {"data": {"nested": "value"}},
            {"not_a_message": "this one"},
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        messages = list(reader.messages())

        assert len(messages) == 0


class TestBlankLines:
    """Test that blank lines are handled correctly."""

    def test_blank_lines_skipped(self, tmp_path):
        """Verify blank lines don't cause errors and are skipped."""
        jsonl_file = tmp_path / "messages.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"message": {"role": "user", "content": "First"}}) + "\n")
            f.write("\n")  # Blank line
            f.write("  \n")  # Whitespace-only line
            f.write(json.dumps({"message": {"role": "assistant", "content": "Second"}}) + "\n")

        reader = JsonlReader(str(jsonl_file))
        messages = list(reader.messages())

        assert len(messages) == 2
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"


class TestContentBlocksDefaultRole:
    """Test that content_blocks() defaults to 'assistant' role."""

    def test_content_blocks_defaults_to_assistant_role(self, tmp_path):
        """Verify content_blocks() defaults to 'assistant' role when not specified."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Assistant"}],
                }
            },
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "User"}],
                }
            },
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        blocks = list(reader.content_blocks())

        assert len(blocks) == 1
        msg, content = blocks[0]
        assert msg["role"] == "assistant"


class TestMessageWithoutContent:
    """Test that messages without content key are still yielded."""

    def test_message_without_content_key(self, tmp_path):
        """Verify messages without content key are still yielded."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "user",
                    "id": "123",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Has content",
                }
            },
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        messages = list(reader.messages())

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "id": "123"}
        assert messages[1] == {"role": "assistant", "content": "Has content"}


class TestContentBlocksWithEmptyContent:
    """Test content_blocks() with empty list content."""

    def test_content_blocks_with_empty_list_content(self, tmp_path):
        """Verify content_blocks() yields messages even with empty content list."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {
                "message": {
                    "role": "assistant",
                    "content": [],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Non-empty"}],
                }
            },
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))
        blocks = list(reader.content_blocks())

        assert len(blocks) == 2
        assert blocks[0][1] == []
        assert len(blocks[1][1]) == 1


class TestErrorFormatting:
    """Test that error messages are correctly formatted."""

    def test_error_message_includes_line_number(self, tmp_path):
        """Verify error messages include the line number."""
        jsonl_file = tmp_path / "messages.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"message": {"role": "user", "content": "Valid"}}) + "\n")
            f.write("{ invalid json\n")
            f.write(json.dumps({"message": {"role": "user", "content": "Another"}}) + "\n")

        reader = JsonlReader(str(jsonl_file), strict=False)
        list(reader.messages())

        assert len(reader.errors) == 1
        assert "Line 2:" in reader.errors[0]


class TestJsonlReaderMultipleIterations:
    """Test that reader can be iterated multiple times."""

    def test_can_iterate_multiple_times(self, tmp_path):
        """Verify reader can iterate over messages() multiple times."""
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            {"message": {"role": "user", "content": "First"}},
            {"message": {"role": "assistant", "content": "Second"}},
        ]
        _write_jsonl(jsonl_file, lines)

        reader = JsonlReader(str(jsonl_file))

        # First iteration
        messages1 = list(reader.messages())
        assert len(messages1) == 2

        # Second iteration
        messages2 = list(reader.messages())
        assert len(messages2) == 2
        assert messages1 == messages2
