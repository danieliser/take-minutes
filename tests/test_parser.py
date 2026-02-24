"""Tests for the parser module."""

import json
import pytest
from pathlib import Path
from minutes.parser import (
    parse_jsonl, parse_text, parse_file,
    FilterConfig, DEFAULT_FILTERS, NO_FILTERS,
)


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


class TestParseJsonl:
    """Tests for parse_jsonl function."""

    def test_parse_jsonl_extracts_user_and_assistant_messages(self, fixtures_dir):
        """Verify JSONL parser extracts user and assistant messages."""
        text, metadata = parse_jsonl(str(fixtures_dir / "sample.jsonl"))

        # Should have 4 messages: 2 user, 2 assistant
        assert metadata["messages"] == 4
        assert metadata["skipped"] == 0
        assert metadata["format"] == "jsonl"

        # Verify messages are labeled and in order
        assert "User: What architecture should we use?" in text
        assert "Assistant: I recommend a layered architecture." in text
        assert "User: Let's go with microservices." in text
        assert "Assistant: Good choice. I'll set up the project structure." in text

    def test_parse_jsonl_filters_tool_use_blocks(self, fixtures_dir):
        """Verify tool_use blocks are filtered from assistant messages."""
        text, metadata = parse_jsonl(str(fixtures_dir / "sample.jsonl"))

        # tool_use block should not appear in output
        assert "tool_use" not in text
        assert "Read" not in text
        assert "/some/file" not in text

    def test_parse_jsonl_skips_non_message_events(self, fixtures_dir):
        """Verify non-message events are skipped."""
        text, metadata = parse_jsonl(str(fixtures_dir / "sample.jsonl"))

        # file-history-snapshot and progress events should be skipped
        assert "file-history-snapshot" not in text
        assert "hook_progress" not in text

    def test_parse_jsonl_skips_tool_role_messages(self, fixtures_dir):
        """Verify tool role messages are skipped."""
        text, metadata = parse_jsonl(str(fixtures_dir / "sample.jsonl"))

        # tool_result message with role="tool" should be skipped
        assert "file contents here" not in text

    def test_parse_jsonl_with_malformed_lines(self, tmp_path):
        """Verify malformed JSON lines are skipped and counted."""
        jsonl_file = tmp_path / "malformed.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": "Valid message"}}\n'
            'not valid json at all\n'
            '{"type": "user", "message": {"role": "user", "content": "Another valid"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 2
        assert metadata["skipped"] == 1
        assert "Valid message" in text
        assert "Another valid" in text

    def test_parse_jsonl_skips_empty_messages(self, tmp_path):
        """Verify empty messages are skipped."""
        jsonl_file = tmp_path / "empty_messages.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": ""}}\n'
            '{"type": "user", "message": {"role": "user", "content": "   "}}\n'
            '{"type": "user", "message": {"role": "user", "content": []}}\n'
            '{"type": "user", "message": {"role": "user", "content": "Real message"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "Real message" in text

    def test_parse_jsonl_handles_content_as_list(self, tmp_path):
        """Verify content as list of blocks is handled correctly."""
        jsonl_file = tmp_path / "content_list.jsonl"
        jsonl_file.write_text(
            '{"type": "assistant", "message": {"role": "assistant", "content": '
            '[{"type": "text", "text": "First "}, {"type": "text", "text": "Second"}]}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "Assistant: First Second" in text

    def test_parse_jsonl_handles_content_as_string(self, tmp_path):
        """Verify content as string is handled correctly."""
        jsonl_file = tmp_path / "content_string.jsonl"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": "Plain string content"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "User: Plain string content" in text

    def test_parse_jsonl_skips_unknown_roles(self, tmp_path):
        """Verify unknown roles are skipped."""
        jsonl_file = tmp_path / "unknown_role.jsonl"
        jsonl_file.write_text(
            '{"type": "system", "message": {"role": "system", "content": "system prompt"}}\n'
            '{"type": "user", "message": {"role": "user", "content": "valid"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "system prompt" not in text
        assert "valid" in text


class TestFilters:
    """Tests for noise filtering in JSONL parser."""

    def test_filters_tool_result_content_blocks(self, tmp_path):
        """Verify tool_result blocks in user messages are filtered."""
        jsonl_file = tmp_path / "tool_result.jsonl"
        jsonl_file.write_text(json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "abc", "content": "ls output here"},
                    {"type": "text", "text": "Now fix that bug."},
                ],
            },
        }) + "\n")

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "Now fix that bug." in text
        assert "ls output here" not in text

    def test_filters_system_reminder_tags(self, tmp_path):
        """Verify <system-reminder> tags are stripped from text content."""
        jsonl_file = tmp_path / "system_reminder.jsonl"
        jsonl_file.write_text(json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text":
                    "Here's the fix.<system-reminder>Task tools haven't been used recently.</system-reminder> Let me apply it now."
                }],
            },
        }) + "\n")

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "Here's the fix." in text
        assert "Let me apply it now." in text
        assert "system-reminder" not in text
        assert "Task tools" not in text

    def test_filters_multiline_system_reminder(self, tmp_path):
        """Verify multiline system-reminder tags are stripped."""
        jsonl_file = tmp_path / "multiline_reminder.jsonl"
        jsonl_file.write_text(json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text":
                    "Starting work.<system-reminder>\nLine 1\nLine 2\nLine 3\n</system-reminder>Done."
                }],
            },
        }) + "\n")

        text, metadata = parse_jsonl(str(jsonl_file))

        assert "Starting work." in text
        assert "Done." in text
        assert "Line 1" not in text

    def test_filters_teammate_message_tags(self, tmp_path):
        """Verify <teammate-message> tags are stripped."""
        jsonl_file = tmp_path / "teammate_msg.jsonl"
        jsonl_file.write_text(json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": '<teammate-message teammate_id="reviewer" color="blue" summary="Task done">Review complete.</teammate-message>',
            },
        }) + "\n")

        text, metadata = parse_jsonl(str(jsonl_file))

        # Entire message is a teammate tag — should be empty after stripping
        assert metadata["messages"] == 0

    def test_filters_idle_notification_json(self, tmp_path):
        """Verify teammate idle notification JSON is filtered."""
        jsonl_file = tmp_path / "idle.jsonl"
        jsonl_file.write_text(json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": '{"type":"idle_notification","from":"integrator","timestamp":"2026-02-24T02:36:48.101Z","idleReason":"available"}',
            },
        }) + "\n")

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 0

    def test_filters_shutdown_protocol_json(self, tmp_path):
        """Verify shutdown protocol JSON is filtered."""
        jsonl_file = tmp_path / "shutdown.jsonl"
        jsonl_file.write_text(json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": '{"type": "shutdown_approved", "requestId": "abc", "from": "reviewer"}',
            },
        }) + "\n")

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 0

    def test_filters_progress_messages(self, tmp_path):
        """Verify progress type messages are filtered."""
        jsonl_file = tmp_path / "progress.jsonl"
        jsonl_file.write_text(
            '{"type": "progress", "data": {"type": "agent_progress"}}\n'
            '{"type": "progress", "data": {"type": "hook_progress"}}\n'
            '{"type": "user", "message": {"role": "user", "content": "real message"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert metadata["filtered"] >= 2
        assert "real message" in text

    def test_filters_file_history_snapshot(self, tmp_path):
        """Verify file-history-snapshot messages are filtered."""
        jsonl_file = tmp_path / "snapshot.jsonl"
        jsonl_file.write_text(
            '{"type": "file-history-snapshot", "snapshot": {"files": []}}\n'
            '{"type": "user", "message": {"role": "user", "content": "real"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert metadata["filtered"] >= 1

    def test_filters_queue_operation(self, tmp_path):
        """Verify queue-operation messages are filtered."""
        jsonl_file = tmp_path / "queue.jsonl"
        jsonl_file.write_text(
            '{"type": "queue-operation", "operation": "enqueue", "content": "task text"}\n'
            '{"type": "user", "message": {"role": "user", "content": "real"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "task text" not in text

    def test_filters_compaction_summary(self, tmp_path):
        """Verify compaction/context compression messages are filtered."""
        jsonl_file = tmp_path / "compaction.jsonl"
        jsonl_file.write_text(json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text":
                    "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion."
                }],
            },
        }) + "\n" + json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": "Real question after compaction.",
            },
        }) + "\n")

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "ran out of context" not in text
        assert "Real question after compaction." in text

    def test_filters_compact_boundary_system(self, tmp_path):
        """Verify compact_boundary system messages are filtered."""
        jsonl_file = tmp_path / "compact_boundary.jsonl"
        jsonl_file.write_text(
            '{"type": "system", "subtype": "compact_boundary", "tokens": {"before": 50000, "after": 20000}}\n'
            '{"type": "user", "message": {"role": "user", "content": "Continue working"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert "Continue working" in text

    def test_keeps_real_conversation_with_embedded_noise(self, tmp_path):
        """Verify real conversation is preserved when mixed with noise."""
        jsonl_file = tmp_path / "mixed.jsonl"
        lines = [
            '{"type": "progress", "data": {"type": "hook_progress"}}',
            '{"type": "file-history-snapshot", "snapshot": {}}',
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": [
                    {"type": "text", "text": "Let's use Redis for caching.<system-reminder>Remember to track tasks.</system-reminder>"},
                ]},
            }),
            json.dumps({
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": "Good idea. Redis gives us TTL support."},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "redis-cli ping"}},
                ]},
            }),
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "PONG"},
                    {"type": "text", "text": "Great, it's running."},
                ]},
            }),
            '{"type": "queue-operation", "operation": "enqueue"}',
        ]
        jsonl_file.write_text("\n".join(lines) + "\n")

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 3
        assert "Let's use Redis for caching." in text
        assert "Good idea. Redis gives us TTL support." in text
        assert "Great, it's running." in text
        # Noise is gone
        assert "system-reminder" not in text
        assert "PONG" not in text
        assert "tool_use" not in text
        assert "hook_progress" not in text

    def test_metadata_includes_filtered_count(self, tmp_path):
        """Verify metadata tracks filtered message count."""
        jsonl_file = tmp_path / "counting.jsonl"
        jsonl_file.write_text(
            '{"type": "progress", "data": {}}\n'
            '{"type": "progress", "data": {}}\n'
            '{"type": "system", "subtype": "turn_duration"}\n'
            '{"type": "file-history-snapshot", "snapshot": {}}\n'
            '{"type": "user", "message": {"role": "user", "content": "hello"}}\n'
        )

        text, metadata = parse_jsonl(str(jsonl_file))

        assert metadata["messages"] == 1
        assert metadata["filtered"] >= 4


class TestFilterConfig:
    """Tests for FilterConfig configurability."""

    def _make_noisy_jsonl(self, tmp_path):
        """Create a JSONL file with all noise types for filter testing."""
        lines = [
            # progress message
            '{"type": "progress", "data": {"type": "hook_progress"}}',
            # file-history-snapshot
            '{"type": "file-history-snapshot", "snapshot": {}}',
            # compaction summary
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": [
                    {"type": "text", "text": "This session is being continued from a previous conversation that ran out of context."},
                ]},
            }),
            # user message with tool_result + system-reminder + real text
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file contents here"},
                    {"type": "text", "text": "Fix the bug.<system-reminder>Use task tools.</system-reminder> Thanks."},
                ]},
            }),
            # assistant with tool_use + real text
            json.dumps({
                "type": "assistant",
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": "I'll fix that now."},
                    {"type": "tool_use", "id": "t2", "name": "Edit", "input": {"file": "foo.py"}},
                ]},
            }),
            # teammate protocol message
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": '{"type":"idle_notification","from":"reviewer"}'},
            }),
        ]
        f = tmp_path / "noisy.jsonl"
        f.write_text("\n".join(lines) + "\n")
        return f

    def test_default_filters_strip_all_noise(self, tmp_path):
        """DEFAULT_FILTERS removes all noise, keeps only conversation."""
        f = self._make_noisy_jsonl(tmp_path)
        text, meta = parse_jsonl(str(f), filter_config=DEFAULT_FILTERS)

        assert meta["messages"] == 2
        assert "Fix the bug." in text
        assert "Thanks." in text
        assert "I'll fix that now." in text
        # Noise is gone
        assert "file contents here" not in text
        assert "system-reminder" not in text
        assert "tool_use" not in text
        assert "idle_notification" not in text
        assert "ran out of context" not in text

    def test_no_filters_keeps_everything(self, tmp_path):
        """NO_FILTERS keeps all content including noise."""
        f = self._make_noisy_jsonl(tmp_path)
        text, meta = parse_jsonl(str(f), filter_config=NO_FILTERS)

        # Should keep tool_result content, system-reminders, protocol, compaction
        assert "file contents here" in text
        assert "system-reminder" in text or "Use task tools." in text
        assert "idle_notification" in text
        assert "ran out of context" in text
        assert meta["messages"] > 2

    def test_none_config_equals_no_filters(self, tmp_path):
        """filter_config=None disables all filtering."""
        f = self._make_noisy_jsonl(tmp_path)
        text_none, meta_none = parse_jsonl(str(f), filter_config=None)
        text_no, meta_no = parse_jsonl(str(f), filter_config=NO_FILTERS)

        assert text_none == text_no
        assert meta_none["messages"] == meta_no["messages"]

    def test_keep_tool_results_only(self, tmp_path):
        """Can keep tool_result blocks while filtering other noise."""
        f = self._make_noisy_jsonl(tmp_path)
        config = FilterConfig(skip_content_types=False)
        text, meta = parse_jsonl(str(f), filter_config=config)

        # tool_result content kept
        assert "file contents here" in text
        # Other filters still active
        assert "idle_notification" not in text
        assert "system-reminder" not in text

    def test_keep_system_reminders(self, tmp_path):
        """Can keep system-reminder tags while filtering other noise."""
        f = self._make_noisy_jsonl(tmp_path)
        config = FilterConfig(strip_system_reminders=False)
        text, meta = parse_jsonl(str(f), filter_config=config)

        assert "system-reminder" in text or "Use task tools." in text
        # tool_result still filtered
        assert "file contents here" not in text

    def test_keep_compaction_summaries(self, tmp_path):
        """Can keep compaction summaries while filtering other noise."""
        f = self._make_noisy_jsonl(tmp_path)
        config = FilterConfig(filter_compaction=False)
        text, meta = parse_jsonl(str(f), filter_config=config)

        assert "ran out of context" in text

    def test_keep_protocol_messages(self, tmp_path):
        """Can keep teammate protocol messages while filtering other noise."""
        f = self._make_noisy_jsonl(tmp_path)
        config = FilterConfig(filter_protocol_messages=False)
        text, meta = parse_jsonl(str(f), filter_config=config)

        assert "idle_notification" in text

    def test_keep_message_types(self, tmp_path):
        """Can keep progress/system messages while filtering other noise."""
        f = self._make_noisy_jsonl(tmp_path)
        config = FilterConfig(skip_message_types=False)
        text, meta = parse_jsonl(str(f), filter_config=config)

        # Progress messages don't have message.content, so they still get filtered
        # by the "must have message dict" check. But the type filter itself is off.
        assert meta["filtered"] < 6  # fewer things filtered than default

    def test_parse_file_passes_filter_config(self, tmp_path):
        """parse_file forwards filter_config to parse_jsonl."""
        f = tmp_path / "test.jsonl"
        f.write_text(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "KEPT"},
                {"type": "text", "text": "hello"},
            ]},
        }) + "\n")

        # Default: tool_result filtered
        text_default, _ = parse_file(str(f))
        assert "KEPT" not in text_default

        # NO_FILTERS: tool_result kept
        text_raw, _ = parse_file(str(f), filter_config=NO_FILTERS)
        assert "KEPT" in text_raw


class TestParseText:
    """Tests for parse_text function."""

    def test_parse_text_returns_full_content(self, fixtures_dir):
        """Verify text parser returns full file content."""
        text, metadata = parse_text(str(fixtures_dir / "sample.txt"))

        assert "Meeting Notes - Project Kickoff" in text
        assert "Decision: Use Python for the backend" in text
        assert "Action: Daniel to set up CI/CD by Friday" in text
        assert metadata["format"] == "text"
        assert metadata["chars"] == len(text)

    def test_parse_text_empty_file(self, tmp_path):
        """Verify empty file is handled correctly."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        text, metadata = parse_text(str(empty_file))

        assert text == ""
        assert metadata["format"] == "text"
        assert metadata["chars"] == 0

    def test_parse_text_preserves_formatting(self, tmp_path):
        """Verify text formatting is preserved."""
        text_file = tmp_path / "formatted.txt"
        content = "Line 1\n\nLine 2\n  Indented line\n"
        text_file.write_text(content)

        text, metadata = parse_text(str(text_file))

        assert text == content
        assert metadata["chars"] == len(content)


class TestParseFile:
    """Tests for parse_file auto-detection."""

    def test_parse_file_detects_jsonl(self, fixtures_dir):
        """Verify .jsonl extension triggers JSONL parser."""
        text, metadata = parse_file(str(fixtures_dir / "sample.jsonl"))

        assert metadata["format"] == "jsonl"
        assert metadata["messages"] == 4

    def test_parse_file_detects_txt(self, fixtures_dir):
        """Verify .txt extension triggers text parser."""
        text, metadata = parse_file(str(fixtures_dir / "sample.txt"))

        assert metadata["format"] == "text"
        assert "Meeting Notes" in text

    def test_parse_file_detects_md(self, tmp_path):
        """Verify .md extension triggers text parser."""
        md_file = tmp_path / "notes.md"
        md_file.write_text("# Header\n\nContent")

        text, metadata = parse_file(str(md_file))

        assert metadata["format"] == "text"
        assert "# Header" in text

    def test_parse_file_detects_markdown(self, tmp_path):
        """Verify .markdown extension triggers text parser."""
        md_file = tmp_path / "notes.markdown"
        md_file.write_text("# Header\n\nContent")

        text, metadata = parse_file(str(md_file))

        assert metadata["format"] == "text"
        assert "# Header" in text

    def test_parse_file_unknown_extension_tries_jsonl(self, tmp_path):
        """Verify unknown extension tries JSONL first."""
        jsonl_file = tmp_path / "data.unknown"
        jsonl_file.write_text(
            '{"type": "user", "message": {"role": "user", "content": "test"}}\n'
        )

        text, metadata = parse_file(str(jsonl_file))

        assert metadata["format"] == "jsonl"
        assert "test" in text

    def test_parse_file_unknown_extension_falls_back_to_text(self, tmp_path):
        """Verify unknown extension falls back to text if JSONL fails."""
        text_file = tmp_path / "data.unknown"
        text_file.write_text("Plain text content")

        text, metadata = parse_file(str(text_file))

        assert metadata["format"] == "text"
        assert "Plain text content" in text

    def test_parse_file_nonexistent_raises_error(self, tmp_path):
        """Verify FileNotFoundError is raised for missing files."""
        nonexistent = tmp_path / "doesnotexist.jsonl"

        with pytest.raises(FileNotFoundError):
            parse_file(str(nonexistent))


class TestIntegration:
    """Integration tests."""

    def test_full_workflow_jsonl(self, fixtures_dir):
        """Verify full workflow with JSONL fixture."""
        text, metadata = parse_file(str(fixtures_dir / "sample.jsonl"))

        # Check metadata
        assert metadata["format"] == "jsonl"
        assert metadata["messages"] == 4
        assert metadata["skipped"] == 0

        # Check content
        lines = text.split("\n\n")
        assert len(lines) == 4
        assert all(line.startswith(("User:", "Assistant:")) for line in lines)

    def test_full_workflow_text(self, fixtures_dir):
        """Verify full workflow with text fixture."""
        text, metadata = parse_file(str(fixtures_dir / "sample.txt"))

        # Check metadata
        assert metadata["format"] == "text"
        assert metadata["chars"] > 0

        # Check content is preserved
        assert "Meeting Notes" in text
        assert "Decision:" in text
        assert "Action:" in text
