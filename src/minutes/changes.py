"""Extract and analyze code changes from Claude Code JSONL transcripts.

Parses tool_use blocks for Edit/Write operations, tracks file modifications,
and generates change timelines and statistics for session analysis.
"""

from __future__ import annotations

from pathlib import Path

from minutes.models import CodeChange, ChangeTimeline, ToolCall, ToolStats
from minutes.parser import _SYSTEM_REMINDER_RE
from minutes.reader import JsonlReader


CHANGE_TOOLS = {"Edit", "Write"}


def parse_changes(file_path: str, strict: bool = False) -> ChangeTimeline:
    """Extract code changes from a JSONL transcript.

    Walks assistant messages, accumulating text blocks as reasoning,
    and extracting CodeChange objects from Edit/Write tool_use blocks.

    Reasoning from text blocks is attributed to the next tool use in the
    same message. Reasoning does not carry across messages.

    Args:
        file_path: Path to JSONL transcript file
        strict: If True, raise on malformed JSON; otherwise skip with warning

    Returns:
        ChangeTimeline with ordered changes and file statistics
    """
    reader = JsonlReader(file_path, strict=strict)
    timeline = ChangeTimeline()
    files_seen = {}  # Track first appearance order

    sequence = 0

    for message, content_blocks in reader.content_blocks("assistant"):
        pending_reasoning: list[str] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                # Accumulate text as reasoning for next tool
                text = block.get("text", "")
                # Strip system-reminder tags
                text = _SYSTEM_REMINDER_RE.sub("", text)
                text = text.strip()
                if text:
                    pending_reasoning.append(text)

            elif block_type == "tool_use":
                tool_name = block.get("name", "")

                if tool_name in CHANGE_TOOLS:
                    # Create CodeChange from this tool_use
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

                    # Track unique files in order of first appearance
                    if change.file_path not in files_seen:
                        files_seen[change.file_path] = True
                        timeline.files_modified.append(change.file_path)

                    # Track edit/write counts
                    if tool_name == "Edit":
                        timeline.total_edits += 1
                    else:
                        timeline.total_writes += 1

                # Reset reasoning for non-CHANGE_TOOLS or after processing change
                pending_reasoning = []
            else:
                # Other tool_use types: reset pending reasoning
                if block_type == "tool_use":
                    pending_reasoning = []

    return timeline


def collect_stats(
    file_path: str, detail: bool = False, strict: bool = False
) -> ToolStats:
    """Collect tool usage statistics from a JSONL transcript.

    Walks all messages (user and assistant), counting tool invocations by type,
    tracking changes per file, and optionally building a detailed call log.

    Args:
        file_path: Path to JSONL transcript file
        detail: If True, build ToolCall list with input summaries and reasoning
        strict: If True, raise on malformed JSON; otherwise skip with warning

    Returns:
        ToolStats with aggregate counts and optional detailed log
    """
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
                text = _SYSTEM_REMINDER_RE.sub("", text)
                text = text.strip()
                if text:
                    pending_reasoning.append(text)

            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                sequence += 1

                # Count this tool
                stats.by_tool[tool_name] = stats.by_tool.get(tool_name, 0) + 1
                stats.total_calls += 1

                # Track specific counts
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

                # Build detail log if requested
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

                # Reset reasoning
                pending_reasoning = []
            else:
                # Other block types: reset reasoning on non-text
                if block_type != "text":
                    pending_reasoning = []

    stats.message_count = stats.user_prompt_count + stats.assistant_turn_count
    return stats


def _summarize_input(tool_name: str, inp: dict) -> str:
    """Summarize tool input for logging.

    Args:
        tool_name: Name of the tool
        inp: Input dict from tool_use block

    Returns:
        Summarized input string (max 120 chars for most tools)
    """
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
        query = inp.get("query", "")
        return query

    if tool_name == "WebFetch":
        url = inp.get("url", "")
        return url

    if tool_name == "Task":
        subagent_type = inp.get("subagent_type", "")
        prompt = inp.get("prompt", "")[:80]
        return f"{subagent_type} — {prompt}"

    # Fallback: first key=value pair
    for key, value in inp.items():
        val_str = str(value)
        if len(val_str) > 80:
            val_str = val_str[:80]
        return f"{key}={val_str}"

    return ""


def format_changes_markdown(
    timeline: ChangeTimeline, session_name: str, full: bool = False
) -> str:
    """Format a ChangeTimeline as markdown.

    Groups changes by file, shows diff for each change, and optionally
    truncates large diffs when full=False.

    Args:
        timeline: The ChangeTimeline to format
        session_name: Name/identifier for the session (for header)
        full: If False, truncate large diffs (>20 lines); if True, show all

    Returns:
        Formatted markdown string
    """
    if not timeline.changes:
        return "No code changes found in this session."

    lines = [
        "# Code Changes Timeline\n",
        f"**Session**: {session_name}",
        f"**Files modified**: {len(timeline.files_modified)}",
        f"**Total changes**: {timeline.total_edits} edits, {timeline.total_writes} writes\n",
    ]

    # Group changes by file (in order of first appearance)
    file_groups = {}
    for file_path in timeline.files_modified:
        file_groups[file_path] = []

    for change in timeline.changes:
        file_groups[change.file_path].append(change)

    # Format each file group
    file_num = 1
    for file_path in timeline.files_modified:
        changes = file_groups[file_path]
        rel_path = Path(file_path).name  # Use basename as relative path
        lines.append(f"## {file_num}. {rel_path} ({len(changes)} changes)")
        file_num += 1

        # Format each change
        change_num = 1
        for change in changes:
            lines.append(
                f"### Change {change_num} ({change.action}) — seq #{change.sequence}"
            )
            if change.reasoning:
                lines.append(f"**Reasoning**: {change.reasoning}")
            lines.append("")

            # Build diff
            if change.action == "edit":
                diff_lines = []
                old_lines = change.old_content.splitlines()
                new_lines = change.new_content.splitlines()

                for line in old_lines:
                    diff_lines.append(f"- {line}")
                for line in new_lines:
                    diff_lines.append(f"+ {line}")

                diff_content = "\n".join(diff_lines)

                # Truncate if needed
                if not full and len(diff_lines) > 20:
                    first_10 = diff_lines[:10]
                    last_5 = diff_lines[-5:]
                    omitted = len(diff_lines) - 15
                    diff_content = (
                        "\n".join(first_10)
                        + f"\n... ({omitted} lines omitted)\n"
                        + "\n".join(last_5)
                    )
            else:  # write
                diff_content = change.new_content
                if not full and len(diff_content.splitlines()) > 30:
                    lines_list = diff_content.splitlines()
                    diff_content = (
                        "\n".join(lines_list[:30]) + f"\n... (truncated, {len(lines_list)} total lines)"
                    )

            lines.append("```diff")
            lines.append(diff_content)
            lines.append("```")
            lines.append("")

            change_num += 1

    return "\n".join(lines)


def format_stats_markdown(
    stats: ToolStats, session_name: str, detail: bool = False
) -> str:
    """Format ToolStats as markdown.

    Shows aggregate statistics in tables, with optional detailed call log.

    Args:
        stats: The ToolStats to format
        session_name: Name/identifier for the session (for header)
        detail: If True, append detailed tool call log

    Returns:
        Formatted markdown string
    """
    lines = [
        "# Session Statistics\n",
        f"**Session**: {session_name}\n",
    ]

    # Metrics table
    lines.append("## Metrics\n")
    lines.append("| Metric | Count |")
    lines.append("| --- | --- |")
    lines.append(f"| User Prompts | {stats.user_prompt_count} |")
    lines.append(f"| Assistant Turns | {stats.assistant_turn_count} |")
    lines.append(f"| Total Tool Calls | {stats.total_calls} |")
    lines.append(f"| Edit | {stats.edit_count} |")
    lines.append(f"| Write | {stats.write_count} |")
    lines.append(f"| Read | {stats.read_count} |")
    lines.append(f"| Bash | {stats.bash_count} |")
    lines.append(f"| Search (Glob+Grep+WebSearch) | {stats.search_count} |")
    lines.append("")

    # Effort by File table
    if stats.by_file:
        lines.append("## Effort by File\n")
        lines.append("| File | Changes |")
        lines.append("| --- | --- |")

        sorted_files = sorted(
            stats.by_file.items(), key=lambda x: x[1], reverse=True
        )
        for file_path, count in sorted_files:
            lines.append(f"| {Path(file_path).name} | {count} |")
        lines.append("")

    # Tool Breakdown table
    if stats.by_tool:
        lines.append("## Tool Breakdown\n")
        lines.append("| Tool | Count |")
        lines.append("| --- | --- |")

        sorted_tools = sorted(
            stats.by_tool.items(), key=lambda x: x[1], reverse=True
        )
        for tool_name, count in sorted_tools:
            lines.append(f"| {tool_name} | {count} |")
        lines.append("")

    # Detail log if requested
    if detail and stats.calls:
        lines.append("## Tool Call Log\n")
        lines.append(
            "| # | Tool | Target | Reasoning (truncated) |"
        )
        lines.append("| --- | --- | --- | --- |")

        for call in stats.calls:
            # Truncate reasoning to 80 chars
            reasoning = call.reasoning
            if reasoning and len(reasoning) > 80:
                reasoning = reasoning[:77] + "..."
            # Escape pipes in reasoning
            reasoning = reasoning.replace("|", "\\|")

            lines.append(
                f"| {call.sequence} | {call.tool_name} | {call.input_summary} | {reasoning} |"
            )
        lines.append("")

    return "\n".join(lines)
