"""Markdown generation for session outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from minutes.models import ExtractionResult


def write_session_markdown(
    result: ExtractionResult,
    metadata: dict[str, Any],
    output_dir: str,
    file_hash: str,
    input_file: str,
    backend_name: str,
) -> str:  # noqa: D103
    """
    Generate and write a markdown file with extracted session content.

    Args:
        result: ExtractionResult containing all extracted data
        metadata: Metadata dict with 'content_metric' and 'format' keys
        output_dir: Directory to write markdown file to
        file_hash: Hash of input file
        input_file: Name of input file
        backend_name: Name of backend used (e.g., "claude-haiku")

    Returns:
        Path to the generated markdown file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp and filename
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{timestamp_str}.md"
    filepath = output_path / filename

    # Build markdown content
    lines = []

    # Header
    readable_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Session Notes — {readable_timestamp}")
    lines.append("")

    # Metadata line
    chunk_count = metadata.get("format", "text")
    if "msgs" in metadata.get("content_metric", ""):
        chunk_info = metadata.get("content_metric", "")
    else:
        chunk_info = metadata.get("content_metric", "0 msgs")

    hash_short = file_hash[:12] if file_hash else "unknown"
    lines.append(f"**Input:** `{input_file}` ({backend_name}, {chunk_info})")
    lines.append(f"**Hash:** `{hash_short}...` (new)")
    lines.append("")

    # TLDR section (always include if present)
    if result.tldr:
        lines.append("## TLDR")
        lines.append(result.tldr)
        lines.append("")

    # Decisions section
    if result.decisions:
        lines.append("## Decisions")
        for i, decision in enumerate(result.decisions, 1):
            reason_text = f"reason: {decision.rationale}" if decision.rationale else ""
            owner_text = f", owner: {decision.owner}" if decision.owner else ""
            extra = f"({reason_text}{owner_text})"
            lines.append(f"{i}. {decision.summary} {extra}".rstrip())
        lines.append("")

    # Ideas section
    if result.ideas:
        lines.append("## Ideas")
        for i, idea in enumerate(result.ideas, 1):
            lines.append(f"{i}. **{idea.title}** — {idea.category}: {idea.description}")
        lines.append("")

    # Questions section
    if result.questions:
        lines.append("## Questions")
        for i, question in enumerate(result.questions, 1):
            context_text = f"(context: {question.context})" if question.context else ""
            lines.append(f"{i}. {question.text} {context_text}".rstrip())
        lines.append("")

    # Action Items section
    if result.action_items:
        lines.append("## Action Items")
        for action in result.action_items:
            owner_text = f"Owner: {action.owner}" if action.owner else "Owner: Unassigned"
            due_text = f", Due: {action.deadline}" if action.deadline else ""
            lines.append(f"- [ ] {action.description} — {owner_text}{due_text}")
        lines.append("")

    # Concepts section
    if result.concepts:
        lines.append("## Concepts")
        for concept in result.concepts:
            lines.append(f"- **{concept.name}:** {concept.definition}")
        lines.append("")

    # Terminology section
    if result.terms:
        lines.append("## Terminology")
        for term in result.terms:
            context_text = f" ({term.context})" if term.context else ""
            lines.append(f"- **{term.term}:** {term.definition}{context_text}")
        lines.append("")

    # Write to file
    content = "\n".join(lines).rstrip() + "\n"
    filepath.write_text(content)

    return str(filepath)
