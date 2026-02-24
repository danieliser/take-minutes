"""Extract code changes and tool statistics from Claude Code JSONL transcripts."""

from __future__ import annotations

from pathlib import Path

from minutes.filters import SYSTEM_REMINDER_RE
from minutes.models import CodeChange, ChangeTimeline, ToolCall, ToolStats
from minutes.reader import JsonlReader


CHANGE_TOOLS = {"Edit", "Write"}


def parse_changes(file_path: str, strict: bool = False) -> ChangeTimeline:
    """Extract code changes from a JSONL transcript.

    Walks assistant messages, accumulating text blocks as reasoning,
    and extracting CodeChange objects from Edit/Write tool_use blocks.

    Reasoning from text blocks is attributed to the next tool use in the
    same message. Reasoning does not carry across messages.
    """
    reader = JsonlReader(file_path, strict=strict)
    timeline = ChangeTimeline()
    files_seen = {}

    sequence = 0

    for message, content_blocks in reader.content_blocks("assistant"):
        pending_reasoning: list[str] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "")
                text = SYSTEM_REMINDER_RE.sub("", text)
                text = text.strip()
                if text:
                    pending_reasoning.append(text)

            elif block_type == "tool_use":
                tool_name = block.get("name", "")

                if tool_name in CHANGE_TOOLS:
                    inp = block.get("input", {})
                    sequence += 1

                    if tool_name == "Edit":
                        change = CodeChange(
                            sequence=sequence,
                            file_path=inp.get("file_path", ""),
                            action="edit",
                            old_content=inp.get("old_string", ""),
                            new_content=inp.get("new_string", ""),
                            reasoning="\n".join(pending_reasoning),
                            tool_use_id=block.get("id", ""),
                        )
                    else:  # Write
                        change = CodeChange(
                            sequence=sequence,
                            file_path=inp.get("file_path", ""),
                            action="write",
                            old_content="",
                            new_content=inp.get("content", ""),
                            reasoning="\n".join(pending_reasoning),
                            tool_use_id=block.get("id", ""),
                        )

                    timeline.changes.append(change)

                    if change.file_path not in files_seen:
                        files_seen[change.file_path] = True
                        timeline.files_modified.append(change.file_path)

                    if tool_name == "Edit":
                        timeline.total_edits += 1
                    else:
                        timeline.total_writes += 1

                pending_reasoning = []
            else:
                if block_type == "tool_use":
                    pending_reasoning = []

    return timeline


def collect_stats(
    file_path: str, detail: bool = False, strict: bool = False
) -> ToolStats:
    """Collect tool usage statistics from a JSONL transcript."""
    reader = JsonlReader(file_path, strict=strict)
    stats = ToolStats()

    sequence = 0

    for message in reader.messages():
        role = message.get("role")
        if role == "user":
            stats.user_prompt_count += 1
        elif role == "assistant":
            stats.assistant_turn_count += 1

        content = message.get("content")
        if not isinstance(content, list):
            continue

        pending_reasoning: list[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text", "")
                text = SYSTEM_REMINDER_RE.sub("", text)
                text = text.strip()
                if text:
                    pending_reasoning.append(text)

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                sequence += 1

                stats.by_tool[tool_name] = stats.by_tool.get(tool_name, 0) + 1
                stats.total_calls += 1

                if tool_name == "Edit":
                    stats.edit_count += 1
                    inp = block.get("input", {})
                    file_path_val = inp.get("file_path", "")
                    if file_path_val:
                        stats.by_file[file_path_val] = (
                            stats.by_file.get(file_path_val, 0) + 1
                        )
                elif tool_name == "Write":
                    stats.write_count += 1
                    inp = block.get("input", {})
                    file_path_val = inp.get("file_path", "")
                    if file_path_val:
                        stats.by_file[file_path_val] = (
                            stats.by_file.get(file_path_val, 0) + 1
                        )
                elif tool_name == "Read":
                    stats.read_count += 1
                elif tool_name == "Bash":
                    stats.bash_count += 1
                elif tool_name in ("Glob", "Grep", "WebSearch"):
                    stats.search_count += 1

                if detail:
                    input_summary = _summarize_input(tool_name, block.get("input", {}))
                    call = ToolCall(
                        sequence=sequence,
                        tool_name=tool_name,
                        tool_use_id=block.get("id", ""),
                        input_summary=input_summary,
                        reasoning="\n".join(pending_reasoning),
                    )
                    stats.calls.append(call)

                pending_reasoning = []
            else:
                if block_type != "text":
                    pending_reasoning = []

    stats.message_count = stats.user_prompt_count + stats.assistant_turn_count
    return stats


def _summarize_input(tool_name: str, inp: dict) -> str:
    """Summarize tool input for logging."""
    if not inp:
        return ""

    if tool_name in ("Edit", "Write", "Read"):
        file_path_val = inp.get("file_path", "")
        return Path(file_path_val).name if file_path_val else ""

    if tool_name == "Bash":
        command = inp.get("command", "")
        return command[:120] if command else ""

    if tool_name in ("Grep", "Glob"):
        pattern = inp.get("pattern", "")
        path = inp.get("path", ".")
        return f"{pattern} in {path}"

    if tool_name == "WebSearch":
        return inp.get("query", "")

    if tool_name == "WebFetch":
        return inp.get("url", "")

    if tool_name == "Task":
        subagent_type = inp.get("subagent_type", "")
        prompt = inp.get("prompt", "")[:80]
        return f"{subagent_type} — {prompt}"

    for key, value in inp.items():
        val_str = str(value)
        if len(val_str) > 80:
            val_str = val_str[:80]
        return f"{key}={val_str}"

    return ""
