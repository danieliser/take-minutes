"""Markdown formatting for intent extraction results."""

from __future__ import annotations

from minutes.models import IntentSummary


def format_intent_markdown(intent: IntentSummary) -> str:
    """Format IntentSummary as markdown output."""
    if not intent.primary_goal:
        return "# User Intent Summary\n\nNo intent could be determined from this session.\n"

    lines = [
        "# User Intent Summary",
        "",
        f"**Primary goal**: {intent.primary_goal}",
        "",
    ]

    if intent.sub_goals:
        lines.append("## Sub-goals")
        for sub_goal in intent.sub_goals:
            lines.append(f"- {sub_goal}")
        lines.append("")

    if intent.constraints:
        lines.append("## Constraints")
        for constraint in intent.constraints:
            lines.append(f"- {constraint}")
        lines.append("")

    lines.append(f"**Prompts analyzed**: {intent.prompt_count}")

    return "\n".join(lines) + "\n"
