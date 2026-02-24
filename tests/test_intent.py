"""Tests for the intent extraction module."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from minutes.intent import (
    extract_user_prompts,
    summarize_intent,
    format_intent_markdown,
    INTENT_CHUNK_SIZE,
    _is_protocol_message,
    _chunk_prompts,
)
from minutes.models import IntentSummary


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


class TestExtractUserPrompts:
    """Tests for extract_user_prompts function."""

    def test_extract_user_prompts_from_jsonl(self, fixtures_dir):
        """Verify user prompts are extracted from JSONL."""
        prompts = extract_user_prompts(str(fixtures_dir / "sample.jsonl"))

        # Should extract 2 user prompts (as strings and as content blocks)
        assert len(prompts) == 2
        assert "What architecture should we use?" in prompts
        assert "Let's go with microservices." in prompts

    def test_extract_user_prompts_skips_tool_results(self, tmp_path):
        """Verify tool_result blocks are skipped."""
        jsonl_file = tmp_path / "with_tool_results.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Do something"}]}}\n'
            '{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Read", "input": {"path": "/file"}}]}}\n'
            '{"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "result"}]}}\n'
            '{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Another prompt"}]}}\n'
        )

        prompts = extract_user_prompts(str(jsonl_file))

        # Should extract 2 prompts, skipping the tool_result block
        assert len(prompts) == 2
        assert "Do something" in prompts
        assert "Another prompt" in prompts

    def test_extract_user_prompts_filters_empty_strings(self, tmp_path):
        """Verify empty prompts are filtered out."""
        jsonl_file = tmp_path / "empty_prompts.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": ""}}\n'
            '{"type": "user", "message": {"role": "user", "content": "Real prompt"}}\n'
            '{"type": "user", "message": {"role": "user", "content": []}}\n'
        )

        prompts = extract_user_prompts(str(jsonl_file))

        assert len(prompts) == 1
        assert prompts[0] == "Real prompt"

    def test_extract_user_prompts_strips_system_reminders(self, tmp_path):
        """Verify system-reminder tags are stripped."""
        jsonl_file = tmp_path / "with_system_reminder.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": "<system-reminder>ignore this</system-reminder>Real prompt"}}\n'
        )

        prompts = extract_user_prompts(str(jsonl_file))

        assert len(prompts) == 1
        assert prompts[0] == "Real prompt"
        assert "system-reminder" not in prompts[0]

    def test_extract_user_prompts_strips_teammate_tags(self, tmp_path):
        """Verify teammate-message tags are stripped."""
        jsonl_file = tmp_path / "with_teammate_msg.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": "<teammate-message>ignore</teammate-message>User prompt"}}\n'
        )

        prompts = extract_user_prompts(str(jsonl_file))

        assert len(prompts) == 1
        assert prompts[0] == "User prompt"
        assert "teammate-message" not in prompts[0]

    def test_extract_user_prompts_filters_protocol_messages(self, tmp_path):
        """Verify protocol messages (idle notifications, etc.) are skipped."""
        jsonl_file = tmp_path / "with_protocol.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": "Real prompt"}}\n'
            '{"type": "user", "message": {"role": "user", "content": "{\\"type\\":\\"idle_notification\\"}"}}\n'
            '{"type": "user", "message": {"role": "user", "content": "{\\"type\\":\\"shutdown_request\\"}"}}\n'
        )

        prompts = extract_user_prompts(str(jsonl_file))

        assert len(prompts) == 1
        assert "Real prompt" in prompts

    def test_extract_user_prompts_handles_mixed_content(self, tmp_path):
        """Verify mixed string and block content is handled."""
        jsonl_file = tmp_path / "mixed_content.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": "String content"}}\n'
            '{"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Block content"}]}}\n'
        )

        prompts = extract_user_prompts(str(jsonl_file))

        assert len(prompts) == 2
        assert "String content" in prompts
        assert "Block content" in prompts


class TestIsProtocolMessage:
    """Tests for _is_protocol_message function."""

    def test_detects_idle_notification(self):
        """Verify idle_notification is detected."""
        assert _is_protocol_message('{"type":"idle_notification"}')
        assert _is_protocol_message('{"type": "idle_notification"}')

    def test_detects_shutdown_request(self):
        """Verify shutdown_request is detected."""
        assert _is_protocol_message('{"type":"shutdown_request"}')
        assert _is_protocol_message('{"type": "shutdown_request"}')

    def test_detects_shutdown_approved(self):
        """Verify shutdown_approved is detected."""
        assert _is_protocol_message('{"type":"shutdown_approved"}')
        assert _is_protocol_message('{"type": "shutdown_approved"}')

    def test_detects_teammate_terminated(self):
        """Verify teammate_terminated is detected."""
        assert _is_protocol_message('{"type":"teammate_terminated"}')
        assert _is_protocol_message('{"type": "teammate_terminated"}')

    def test_ignores_normal_text(self):
        """Verify normal text is not detected as protocol."""
        assert not _is_protocol_message("Normal user prompt")
        assert not _is_protocol_message('{"key": "value"}')


class TestChunkPrompts:
    """Tests for _chunk_prompts function."""

    def test_chunk_prompts_preserves_small_lists(self):
        """Verify small prompt lists are not modified."""
        prompts = ["p1", "p2", "p3"]
        result = _chunk_prompts(prompts)
        assert result == "p1\n---\np2\n---\np3"

    def test_chunk_prompts_keeps_first_and_last_three(self):
        """Verify large lists keep first 3 and last 3."""
        prompts = [f"prompt_{i}" for i in range(10)]
        result = _chunk_prompts(prompts)

        # Should have first 3, omission marker, last 3
        assert "prompt_0" in result
        assert "prompt_1" in result
        assert "prompt_2" in result
        assert "[4 prompts omitted]" in result
        assert "prompt_7" in result
        assert "prompt_8" in result
        assert "prompt_9" in result

    def test_chunk_prompts_correct_omission_count(self):
        """Verify omission count is accurate."""
        prompts = [f"p{i}" for i in range(20)]
        result = _chunk_prompts(prompts)

        # 20 prompts - 6 kept = 14 omitted
        assert "[14 prompts omitted]" in result


class TestSummarizeIntent:
    """Tests for summarize_intent function."""

    def test_summarize_intent_returns_empty_for_no_prompts(self):
        """Verify empty prompt list returns empty intent."""
        backend = Mock()
        result = summarize_intent(backend, [])

        assert result.prompt_count == 0
        assert result.primary_goal == ""
        assert result.sub_goals == []
        assert result.constraints == []
        # Backend should not be called
        backend.chat.completions.create.assert_not_called()

    def test_summarize_intent_calls_backend(self):
        """Verify backend is called for non-empty prompts."""
        backend = MagicMock()
        backend._model = "test-model"
        backend.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "primary_goal": "Build a feature",
            "sub_goals": ["Task 1", "Task 2"],
            "constraints": ["2 hours"],
        })

        prompts = ["Prompt 1", "Prompt 2"]
        result = summarize_intent(backend, prompts)

        assert result.prompt_count == 2
        assert result.primary_goal == "Build a feature"
        assert result.sub_goals == ["Task 1", "Task 2"]
        assert result.constraints == ["2 hours"]
        backend.chat.completions.create.assert_called_once()

    def test_summarize_intent_uses_model_attribute(self):
        """Verify backend._model is used if available."""
        backend = MagicMock()
        backend._model = "custom-model"
        backend.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "primary_goal": "Test",
            "sub_goals": [],
            "constraints": [],
        })

        summarize_intent(backend, ["prompt"])

        call_args = backend.chat.completions.create.call_args
        assert call_args[1]["model"] == "custom-model"

    def test_summarize_intent_uses_default_model_if_no_attribute(self):
        """Verify 'default' model is used if backend has no _model."""
        backend = MagicMock(spec=['chat'])
        # Remove _model attribute to simulate missing model
        if hasattr(backend, '_model'):
            delattr(backend, '_model')
        backend.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "primary_goal": "Test",
            "sub_goals": [],
            "constraints": [],
        })

        summarize_intent(backend, ["prompt"])

        call_args = backend.chat.completions.create.call_args
        assert call_args[1]["model"] == "default"

    def test_summarize_intent_handles_exceptions(self):
        """Verify exceptions return empty result with prompt count."""
        backend = Mock()
        backend.chat.completions.create.side_effect = Exception("LLM error")

        prompts = ["Prompt 1"]
        result = summarize_intent(backend, prompts)

        assert result.prompt_count == 1
        assert result.primary_goal == ""
        assert result.sub_goals == []
        assert result.constraints == []

    def test_summarize_intent_chunks_large_prompts(self):
        """Verify large prompt concatenation is chunked."""
        backend = MagicMock()
        backend._model = "test-model"
        backend.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "primary_goal": "Test",
            "sub_goals": [],
            "constraints": [],
        })

        # Create prompts that exceed chunk size
        large_prompts = ["x" * (INTENT_CHUNK_SIZE // 5) for _ in range(10)]
        result = summarize_intent(backend, large_prompts)

        call_args = backend.chat.completions.create.call_args
        user_content = call_args[1]["messages"][1]["content"]
        # Should contain omission marker since it was chunked
        assert "[" in user_content and "prompts omitted" in user_content


class TestFormatIntentMarkdown:
    """Tests for format_intent_markdown function."""

    def test_format_intent_markdown_full(self):
        """Verify full intent is formatted correctly."""
        intent = IntentSummary(
            primary_goal="Build a web application",
            sub_goals=["Set up database", "Create API"],
            constraints=["2 weeks", "Budget $5000"],
            prompt_count=5,
        )

        result = format_intent_markdown(intent)

        assert "# User Intent Summary" in result
        assert "**Primary goal**: Build a web application" in result
        assert "## Sub-goals" in result
        assert "- Set up database" in result
        assert "- Create API" in result
        assert "## Constraints" in result
        assert "- 2 weeks" in result
        assert "- Budget $5000" in result
        assert "**Prompts analyzed**: 5" in result

    def test_format_intent_markdown_without_subgoals(self):
        """Verify missing sub-goals section is omitted."""
        intent = IntentSummary(
            primary_goal="Build a feature",
            sub_goals=[],
            constraints=["1 week"],
            prompt_count=3,
        )

        result = format_intent_markdown(intent)

        assert "# User Intent Summary" in result
        assert "**Primary goal**: Build a feature" in result
        assert "## Sub-goals" not in result
        assert "## Constraints" in result

    def test_format_intent_markdown_without_constraints(self):
        """Verify missing constraints section is omitted."""
        intent = IntentSummary(
            primary_goal="Build a feature",
            sub_goals=["Task 1"],
            constraints=[],
            prompt_count=2,
        )

        result = format_intent_markdown(intent)

        assert "## Sub-goals" in result
        assert "## Constraints" not in result

    def test_format_intent_markdown_empty_intent(self):
        """Verify empty intent returns appropriate message."""
        intent = IntentSummary(prompt_count=0)

        result = format_intent_markdown(intent)

        assert "# User Intent Summary" in result
        assert "No intent could be determined from this session." in result
        assert "**Primary goal**:" not in result
        assert "## Sub-goals" not in result


class TestIntentChunkSize:
    """Tests for INTENT_CHUNK_SIZE constant."""

    def test_intent_chunk_size_value(self):
        """Verify INTENT_CHUNK_SIZE is 8000."""
        assert INTENT_CHUNK_SIZE == 8000
