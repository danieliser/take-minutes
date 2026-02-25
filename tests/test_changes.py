"""Tests for the changes module."""

import json
import pytest
from pathlib import Path

from minutes.changes import (
    parse_changes,
    collect_stats,
    format_changes_markdown,
    format_stats_markdown,
    _summarize_input,
    CHANGE_TOOLS,
)


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Create a temporary JSONL file with test data."""
    return tmp_path / "test.jsonl"


class TestParseChanges:
    """Tests for parse_changes function."""

    def test_parse_changes_extracts_write_operation(self, tmp_jsonl):
        """Verify Write tool_use creates CodeChange."""
        test_data = [
            {
                "type": "user",
                "message": {"role": "user", "content": "Create a file"},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll create a new Python file.",
                        },
                        {
                            "type": "tool_use",
                            "id": "write-1",
                            "name": "Write",
                            "input": {
                                "file_path": "/tmp/test.py",
                                "content": "def hello():\n    pass",
                            },
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        timeline = parse_changes(str(tmp_jsonl))

        assert len(timeline.changes) == 1
        assert timeline.total_writes == 1
        assert timeline.total_edits == 0
        assert timeline.changes[0].action == "write"
        assert timeline.changes[0].file_path == "/tmp/test.py"
        assert timeline.changes[0].new_content == "def hello():\n    pass"
        assert "create a new python file" in timeline.changes[0].reasoning.lower()
        assert "/tmp/test.py" in timeline.files_modified

    def test_parse_changes_extracts_edit_operation(self, tmp_jsonl):
        """Verify Edit tool_use creates CodeChange."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll improve the function.",
                        },
                        {
                            "type": "tool_use",
                            "id": "edit-1",
                            "name": "Edit",
                            "input": {
                                "file_path": "/tmp/test.py",
                                "old_string": "def hello():\n    pass",
                                "new_string": "def hello():\n    print('hello')",
                            },
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        timeline = parse_changes(str(tmp_jsonl))

        assert len(timeline.changes) == 1
        assert timeline.total_edits == 1
        assert timeline.total_writes == 0
        assert timeline.changes[0].action == "edit"
        assert timeline.changes[0].old_content == "def hello():\n    pass"
        assert timeline.changes[0].new_content == "def hello():\n    print('hello')"

    def test_parse_changes_reasoning_only_to_next_tool(self, tmp_jsonl):
        """Verify reasoning is attributed only to next tool in same message."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First thought."},
                        {
                            "type": "tool_use",
                            "id": "read-1",
                            "name": "Read",
                            "input": {"file_path": "/tmp/old.py"},
                        },
                        {"type": "text", "text": "Second thought."},
                        {
                            "type": "tool_use",
                            "id": "edit-1",
                            "name": "Edit",
                            "input": {
                                "file_path": "/tmp/test.py",
                                "old_string": "x = 1",
                                "new_string": "x = 2",
                            },
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        timeline = parse_changes(str(tmp_jsonl))

        # Only Edit is tracked (Read is not in CHANGE_TOOLS)
        assert len(timeline.changes) == 1
        # Edit should have "Second thought" as reasoning
        assert "Second thought" in timeline.changes[0].reasoning
        assert "First thought" not in timeline.changes[0].reasoning

    def test_parse_changes_reasoning_resets_at_message_boundary(self, tmp_jsonl):
        """Verify reasoning does not carry across messages."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Message 1 reasoning."},
                        {
                            "type": "tool_use",
                            "id": "write-1",
                            "name": "Write",
                            "input": {
                                "file_path": "/tmp/file1.py",
                                "content": "# file 1",
                            },
                        },
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "write-2",
                            "name": "Write",
                            "input": {
                                "file_path": "/tmp/file2.py",
                                "content": "# file 2",
                            },
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        timeline = parse_changes(str(tmp_jsonl))

        assert len(timeline.changes) == 2
        assert "Message 1 reasoning" in timeline.changes[0].reasoning
        assert "Message 1 reasoning" not in timeline.changes[1].reasoning

    def test_parse_changes_strips_system_reminder_tags(self, tmp_jsonl):
        """Verify system-reminder tags are stripped from reasoning."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Start.<system-reminder>Hidden stuff</system-reminder> End.",
                        },
                        {
                            "type": "tool_use",
                            "id": "write-1",
                            "name": "Write",
                            "input": {
                                "file_path": "/tmp/test.py",
                                "content": "code",
                            },
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        timeline = parse_changes(str(tmp_jsonl))

        assert "Start." in timeline.changes[0].reasoning
        assert "End." in timeline.changes[0].reasoning
        assert "Hidden stuff" not in timeline.changes[0].reasoning
        assert "system-reminder" not in timeline.changes[0].reasoning

    def test_parse_changes_tracks_files_in_order(self, tmp_jsonl):
        """Verify files_modified tracks order of first appearance."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "w1",
                            "name": "Write",
                            "input": {
                                "file_path": "/tmp/file_c.py",
                                "content": "c",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "w2",
                            "name": "Write",
                            "input": {
                                "file_path": "/tmp/file_a.py",
                                "content": "a",
                            },
                        },
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "e1",
                            "name": "Edit",
                            "input": {
                                "file_path": "/tmp/file_c.py",
                                "old_string": "c",
                                "new_string": "c2",
                            },
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        timeline = parse_changes(str(tmp_jsonl))

        # Files in order of first appearance: c, then a
        assert timeline.files_modified == ["/tmp/file_c.py", "/tmp/file_a.py"]
        assert len(timeline.changes) == 3
        assert timeline.total_writes == 2
        assert timeline.total_edits == 1

    def test_parse_changes_sequence_numbering(self, tmp_jsonl):
        """Verify changes are numbered sequentially."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "w1",
                            "name": "Write",
                            "input": {
                                "file_path": "/tmp/f1.py",
                                "content": "1",
                            },
                        },
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "w2",
                            "name": "Write",
                            "input": {
                                "file_path": "/tmp/f2.py",
                                "content": "2",
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "e1",
                            "name": "Edit",
                            "input": {
                                "file_path": "/tmp/f3.py",
                                "old_string": "old",
                                "new_string": "new",
                            },
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        timeline = parse_changes(str(tmp_jsonl))

        assert len(timeline.changes) == 3
        assert timeline.changes[0].sequence == 1
        assert timeline.changes[1].sequence == 2
        assert timeline.changes[2].sequence == 3


class TestCollectStats:
    """Tests for collect_stats function."""

    def test_collect_stats_counts_tools(self, tmp_jsonl):
        """Verify tool counts are aggregated correctly."""
        test_data = [
            {
                "type": "user",
                "message": {"role": "user", "content": "Do stuff"},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "e1",
                            "name": "Edit",
                            "input": {"file_path": "/tmp/f.py", "old_string": "a", "new_string": "b"},
                        },
                        {
                            "type": "tool_use",
                            "id": "r1",
                            "name": "Read",
                            "input": {"file_path": "/tmp/f2.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "b1",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "e2",
                            "name": "Edit",
                            "input": {"file_path": "/tmp/f.py", "old_string": "c", "new_string": "d"},
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        stats = collect_stats(str(tmp_jsonl))

        assert stats.total_calls == 4
        assert stats.edit_count == 2
        assert stats.read_count == 1
        assert stats.bash_count == 1
        assert stats.by_tool["Edit"] == 2
        assert stats.by_tool["Read"] == 1
        assert stats.by_tool["Bash"] == 1

    def test_collect_stats_counts_messages(self, tmp_jsonl):
        """Verify user and assistant message counts."""
        test_data = [
            {"type": "user", "message": {"role": "user", "content": "msg1"}},
            {"type": "assistant", "message": {"role": "assistant", "content": []}},
            {"type": "user", "message": {"role": "user", "content": "msg2"}},
            {"type": "assistant", "message": {"role": "assistant", "content": []}},
            {"type": "user", "message": {"role": "user", "content": "msg3"}},
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        stats = collect_stats(str(tmp_jsonl))

        assert stats.user_prompt_count == 3
        assert stats.assistant_turn_count == 2
        assert stats.message_count == 5

    def test_collect_stats_tracks_by_file(self, tmp_jsonl):
        """Verify changes per file are tracked."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "w1",
                            "name": "Write",
                            "input": {"file_path": "/tmp/a.py", "content": "code"},
                        },
                        {
                            "type": "tool_use",
                            "id": "e1",
                            "name": "Edit",
                            "input": {"file_path": "/tmp/a.py", "old_string": "x", "new_string": "y"},
                        },
                        {
                            "type": "tool_use",
                            "id": "e2",
                            "name": "Edit",
                            "input": {"file_path": "/tmp/b.py", "old_string": "x", "new_string": "y"},
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        stats = collect_stats(str(tmp_jsonl))

        assert stats.by_file["/tmp/a.py"] == 2
        assert stats.by_file["/tmp/b.py"] == 1

    def test_collect_stats_search_tools(self, tmp_jsonl):
        """Verify Glob, Grep, WebSearch are counted as search."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "g1",
                            "name": "Glob",
                            "input": {"pattern": "*.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "gr1",
                            "name": "Grep",
                            "input": {"pattern": "foo"},
                        },
                        {
                            "type": "tool_use",
                            "id": "ws1",
                            "name": "WebSearch",
                            "input": {"query": "foo"},
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        stats = collect_stats(str(tmp_jsonl))

        assert stats.search_count == 3
        assert stats.by_tool["Glob"] == 1
        assert stats.by_tool["Grep"] == 1
        assert stats.by_tool["WebSearch"] == 1

    def test_collect_stats_detail_mode(self, tmp_jsonl):
        """Verify detail mode builds ToolCall list."""
        test_data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Reasoning text."},
                        {
                            "type": "tool_use",
                            "id": "e1",
                            "name": "Edit",
                            "input": {"file_path": "/tmp/test.py", "old_string": "a", "new_string": "b"},
                        },
                    ],
                },
            },
        ]
        tmp_jsonl.write_text(
            "\n".join(json.dumps(item) for item in test_data)
        )

        stats = collect_stats(str(tmp_jsonl), detail=True)

        assert len(stats.calls) == 1
        assert stats.calls[0].tool_name == "Edit"
        assert stats.calls[0].input_summary == "test.py"
        assert "Reasoning text" in stats.calls[0].reasoning


class TestSummarizeInput:
    """Tests for _summarize_input function."""

    def test_summarize_input_edit_write_read(self):
        """Verify file-based tools return basename."""
        result = _summarize_input("Edit", {"file_path": "/path/to/file.py"})
        assert result == "file.py"

        result = _summarize_input("Write", {"file_path": "/tmp/test.txt"})
        assert result == "test.txt"

        result = _summarize_input("Read", {"file_path": "/root/config.ini"})
        assert result == "config.ini"

    def test_summarize_input_bash(self):
        """Verify Bash returns first 120 chars of command."""
        long_cmd = "echo " + "x" * 150
        result = _summarize_input("Bash", {"command": long_cmd})
        assert len(result) == 120

        short_cmd = "ls -la"
        result = _summarize_input("Bash", {"command": short_cmd})
        assert result == short_cmd

    def test_summarize_input_grep_glob(self):
        """Verify Grep/Glob return pattern and path."""
        result = _summarize_input("Grep", {"pattern": "foo.*bar", "path": "/src"})
        assert "foo.*bar in /src" == result

        result = _summarize_input("Glob", {"pattern": "**/*.py", "path": "."})
        assert "**/*.py in ." == result

    def test_summarize_input_web_tools(self):
        """Verify WebSearch and WebFetch return query/url."""
        result = _summarize_input("WebSearch", {"query": "python async patterns"})
        assert result == "python async patterns"

        result = _summarize_input("WebFetch", {"url": "https://example.com"})
        assert result == "https://example.com"

    def test_summarize_input_task(self):
        """Verify Task returns subagent_type and truncated prompt."""
        result = _summarize_input(
            "Task",
            {"subagent_type": "reviewer", "prompt": "x" * 200}
        )
        assert "reviewer" in result
        assert len(result) < 200

    def test_summarize_input_empty(self):
        """Verify empty input returns empty string."""
        result = _summarize_input("Edit", {})
        assert result == ""


class TestFormatChangesMarkdown:
    """Tests for format_changes_markdown function."""

    def test_format_changes_empty_timeline(self):
        """Verify empty timeline shows appropriate message."""
        from minutes.models import ChangeTimeline

        timeline = ChangeTimeline()
        result = format_changes_markdown(timeline, "test-session")

        assert "No code changes found" in result

    def test_format_changes_header(self):
        """Verify header includes session name and counts."""
        from minutes.models import ChangeTimeline, CodeChange

        timeline = ChangeTimeline()
        timeline.changes = [
            CodeChange(
                sequence=1,
                file_path="/tmp/a.py",
                action="write",
                new_content="code",
            ),
            CodeChange(
                sequence=2,
                file_path="/tmp/b.py",
                action="edit",
                old_content="old",
                new_content="new",
            ),
        ]
        timeline.files_modified = ["/tmp/a.py", "/tmp/b.py"]
        timeline.total_writes = 1
        timeline.total_edits = 1

        result = format_changes_markdown(timeline, "my-session")

        assert "# Code Changes Timeline" in result
        assert "**Session**: my-session" in result
        assert "**Files modified**: 2" in result
        assert "1 edits, 1 writes" in result

    def test_format_changes_groups_by_file(self):
        """Verify changes are grouped by file."""
        from minutes.models import ChangeTimeline, CodeChange

        timeline = ChangeTimeline()
        timeline.changes = [
            CodeChange(
                sequence=1,
                file_path="/tmp/file.py",
                action="write",
                new_content="line1\nline2",
            ),
            CodeChange(
                sequence=2,
                file_path="/tmp/file.py",
                action="edit",
                old_content="line2",
                new_content="line2_modified",
            ),
        ]
        timeline.files_modified = ["/tmp/file.py"]
        timeline.total_writes = 1
        timeline.total_edits = 1

        result = format_changes_markdown(timeline, "test")

        assert "## 1. file.py (2 changes)" in result
        assert "### Change 1 (write)" in result
        assert "### Change 2 (edit)" in result

    def test_format_changes_includes_reasoning(self):
        """Verify reasoning is included in output."""
        from minutes.models import ChangeTimeline, CodeChange

        timeline = ChangeTimeline()
        timeline.changes = [
            CodeChange(
                sequence=1,
                file_path="/tmp/test.py",
                action="write",
                new_content="code",
                reasoning="Create initial file structure",
            ),
        ]
        timeline.files_modified = ["/tmp/test.py"]
        timeline.total_writes = 1

        result = format_changes_markdown(timeline, "test")

        assert "**Reasoning**: Create initial file structure" in result

    def test_format_changes_truncation_disabled(self):
        """Verify full=True disables truncation."""
        from minutes.models import ChangeTimeline, CodeChange

        # Create edit with many lines
        old_lines = [f"line {i}" for i in range(50)]
        new_lines = [f"line {i} modified" for i in range(50)]

        timeline = ChangeTimeline()
        timeline.changes = [
            CodeChange(
                sequence=1,
                file_path="/tmp/large.py",
                action="edit",
                old_content="\n".join(old_lines),
                new_content="\n".join(new_lines),
            ),
        ]
        timeline.files_modified = ["/tmp/large.py"]
        timeline.total_edits = 1

        result_full = format_changes_markdown(timeline, "test", full=True)
        result_truncated = format_changes_markdown(timeline, "test", full=False)

        assert "... (" not in result_full
        assert "... (" in result_truncated


class TestFormatStatsMarkdown:
    """Tests for format_stats_markdown function."""

    def test_format_stats_header(self):
        """Verify header includes session name."""
        from minutes.models import ToolStats

        stats = ToolStats()
        result = format_stats_markdown(stats, "my-session")

        assert "# Session Statistics" in result
        assert "**Session**: my-session" in result

    def test_format_stats_metrics_table(self):
        """Verify metrics table is included."""
        from minutes.models import ToolStats

        stats = ToolStats(
            user_prompt_count=3,
            assistant_turn_count=2,
            total_calls=5,
            edit_count=1,
            write_count=2,
            read_count=1,
            bash_count=1,
            search_count=0,
        )

        result = format_stats_markdown(stats, "test")

        assert "## Metrics" in result
        assert "| User Prompts | 3 |" in result
        assert "| Assistant Turns | 2 |" in result
        assert "| Edit | 1 |" in result
        assert "| Write | 2 |" in result

    def test_format_stats_by_file_table(self):
        """Verify by_file table is included and sorted."""
        from minutes.models import ToolStats

        stats = ToolStats(
            by_file={
                "/tmp/b.py": 1,
                "/tmp/a.py": 3,
                "/tmp/c.py": 2,
            }
        )

        result = format_stats_markdown(stats, "test")

        assert "## Effort by File" in result
        # Should be sorted by count descending
        lines = result.split("\n")
        a_idx = next(i for i, line in enumerate(lines) if "a.py" in line)
        c_idx = next(i for i, line in enumerate(lines) if "c.py" in line)
        b_idx = next(i for i, line in enumerate(lines) if "b.py" in line)
        assert a_idx < c_idx < b_idx

    def test_format_stats_tool_breakdown_table(self):
        """Verify tool breakdown table is included and sorted."""
        from minutes.models import ToolStats

        stats = ToolStats(
            by_tool={"Edit": 5, "Read": 2, "Bash": 8}
        )

        result = format_stats_markdown(stats, "test")

        assert "## Tool Breakdown" in result
        # Bash (8) should appear before Edit (5) should appear before Read (2)
        lines = result.split("\n")
        bash_idx = next(i for i, line in enumerate(lines) if "Bash" in line and "8" in line)
        edit_idx = next(i for i, line in enumerate(lines) if "Edit" in line and "5" in line)
        read_idx = next(i for i, line in enumerate(lines) if "Read" in line and "2" in line)
        assert bash_idx < edit_idx < read_idx

    def test_format_stats_detail_mode(self):
        """Verify detail=True includes tool call log."""
        from minutes.models import ToolStats, ToolCall

        stats = ToolStats(
            calls=[
                ToolCall(
                    sequence=1,
                    tool_name="Edit",
                    input_summary="test.py",
                    reasoning="Fix bug",
                ),
                ToolCall(
                    sequence=2,
                    tool_name="Bash",
                    input_summary="ls -la",
                    reasoning="Check files",
                ),
            ]
        )

        result_with_detail = format_stats_markdown(stats, "test", detail=True)
        result_without_detail = format_stats_markdown(stats, "test", detail=False)

        assert "## Tool Call Log" in result_with_detail
        assert "## Tool Call Log" not in result_without_detail
        assert "| # | Tool | Target | Reasoning" in result_with_detail

    def test_format_stats_reasoning_truncation(self):
        """Verify reasoning is truncated to 80 chars in detail log."""
        from minutes.models import ToolStats, ToolCall

        long_reasoning = "x" * 200

        stats = ToolStats(
            calls=[
                ToolCall(
                    sequence=1,
                    tool_name="Edit",
                    input_summary="test.py",
                    reasoning=long_reasoning,
                ),
            ]
        )

        result = format_stats_markdown(stats, "test", detail=True)

        assert "..." in result
        # Check that we don't have full 200 char string
        assert long_reasoning not in result


class TestConstant:
    """Tests for module constants."""

    def test_change_tools_contains_edit_and_write(self):
        """Verify CHANGE_TOOLS has Edit and Write."""
        assert "Edit" in CHANGE_TOOLS
        assert "Write" in CHANGE_TOOLS
        assert len(CHANGE_TOOLS) == 2
