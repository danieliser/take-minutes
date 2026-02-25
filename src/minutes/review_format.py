"""Markdown formatting for review gap analysis."""

from __future__ import annotations

from minutes.models import ReviewResult


def format_review_markdown(result: ReviewResult, filename: str) -> str:
    """Format a ReviewResult as markdown."""
    lines = [f"# Session Review: {filename}", ""]

    if not result.summary and not result.covered and not result.gaps:
        lines.append("No review could be generated for this session.")
        if result.intent_prompt_count == 0:
            lines.append("No user prompts found (likely a team/subagent session).")
        if result.changes_count == 0:
            lines.append("No code changes found.")
        lines.extend(["", f"**Prompts analyzed**: {result.intent_prompt_count} | **Code changes**: {result.changes_count}"])
        return "\n".join(lines)

    lines.append(f"**Alignment Score**: {result.alignment_score:.2f} / 1.0")
    lines.append(f"**Prompts analyzed**: {result.intent_prompt_count} | **Code changes**: {result.changes_count}")
    lines.append("")

    if result.summary:
        lines.extend(["## Summary", "", result.summary, ""])

    if result.covered:
        lines.append("## Covered")
        lines.append("")
        for item in result.covered:
            line = f"- {item.description}"
            if item.evidence:
                line += f" — *{item.evidence}*"
            lines.append(line)
        lines.append("")

    if result.gaps:
        lines.append("## Gaps (requested but not done)")
        lines.append("")
        for item in result.gaps:
            line = f"- {item.description}"
            if item.evidence:
                line += f" — *{item.evidence}*"
            lines.append(line)
        lines.append("")

    if result.unasked:
        lines.append("## Unasked Work (done but not requested)")
        lines.append("")
        for item in result.unasked:
            line = f"- {item.description}"
            if item.evidence:
                line += f" — *{item.evidence}*"
            lines.append(line)
        lines.append("")

    return "\n".join(lines)
