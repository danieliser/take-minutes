"""Markdown formatting for code changes and tool statistics."""

from __future__ import annotations

from pathlib import Path

from minutes.models import ChangeTimeline, ToolStats


def format_changes_markdown(
    timeline: ChangeTimeline, session_name: str, full: bool = False
) -> str:
    """Format a ChangeTimeline as markdown.

    Groups changes by file, shows diff for each change, and optionally
    truncates large diffs when full=False.
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
    file_groups: dict[str, list] = {}
    for file_path in timeline.files_modified:
        file_groups[file_path] = []

    for change in timeline.changes:
        file_groups[change.file_path].append(change)

    file_num = 1
    for file_path in timeline.files_modified:
        changes = file_groups[file_path]
        rel_path = Path(file_path).name
        lines.append(f"## {file_num}. {rel_path} ({len(changes)} changes)")
        file_num += 1

        change_num = 1
        for change in changes:
            lines.append(
                f"### Change {change_num} ({change.action}) — seq #{change.sequence}"
            )
            if change.reasoning:
                lines.append(f"**Reasoning**: {change.reasoning}")
            lines.append("")

            if change.action == "edit":
                diff_lines = []
                old_lines = change.old_content.splitlines()
                new_lines = change.new_content.splitlines()

                for line in old_lines:
                    diff_lines.append(f"- {line}")
                for line in new_lines:
                    diff_lines.append(f"+ {line}")

                diff_content = "\n".join(diff_lines)

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
    """Format ToolStats as markdown."""
    lines = [
        "# Session Statistics\n",
        f"**Session**: {session_name}\n",
    ]

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

    if detail and stats.calls:
        lines.append("## Tool Call Log\n")
        lines.append(
            "| # | Tool | Target | Reasoning (truncated) |"
        )
        lines.append("| --- | --- | --- | --- |")

        for call in stats.calls:
            reasoning = call.reasoning
            if reasoning and len(reasoning) > 80:
                reasoning = reasoning[:77] + "..."
            reasoning = reasoning.replace("|", "\\|")

            lines.append(
                f"| {call.sequence} | {call.tool_name} | {call.input_summary} | {reasoning} |"
            )
        lines.append("")

    return "\n".join(lines)
