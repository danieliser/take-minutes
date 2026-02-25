"""Logging and indexing for session outputs."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from minutes.models import ExtractionResult

logger = logging.getLogger(__name__)


def append_session_log(
    output_dir: str,
    input_file: str,
    metadata: dict[str, Any],
    result: ExtractionResult,
    file_hash: str,
    is_cached: bool,
) -> None:  # noqa: D103
    """
    Append a session entry to the session.log file.

    Args:
        output_dir: Directory containing session.log
        input_file: Name of input file
        metadata: Metadata dict with content_metric
        result: ExtractionResult with extracted counts
        file_hash: Hash of input file
        is_cached: Whether this result was cached
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "session.log"

    # Get current ISO timestamp
    timestamp = datetime.now().isoformat()

    # Get content metric from metadata
    content_metric = metadata.get("content_metric", "0 msgs")

    # Count extracted items
    decisions_count = len(result.decisions)
    ideas_count = len(result.ideas)
    questions_count = len(result.questions)
    actions_count = len(result.action_items)

    # Hash first 12 characters
    hash_12 = file_hash[:12] if file_hash else "unknown"

    # Status
    status = "cached" if is_cached else "new"

    # Build tab-separated line
    fields = [
        timestamp,
        input_file,
        content_metric,
        str(decisions_count),
        str(ideas_count),
        str(questions_count),
        str(actions_count),
        hash_12,
        status,
    ]

    line = "\t".join(fields)

    # Append to log file
    with open(log_file, "a") as f:
        f.write(line + "\n")


def update_index(
    output_dir: str,
    input_file: str,
    result: ExtractionResult,
    file_hash: str,
    output_file: str,
    glossary_matches: int = 0,
    glossary_unknown: int = 0,
) -> None:  # noqa: D103
    """
    Update or create the index.json file with session metadata and stats.

    Args:
        output_dir: Directory containing index.json
        input_file: Name of input file
        result: ExtractionResult with extracted data
        file_hash: Hash of input file
        output_file: Name of output markdown file
        glossary_matches: Number of glossary matches
        glossary_unknown: Number of unknown glossary terms
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_file = output_path / "index.json"

    # Load existing index or create new
    if index_file.exists():
        data = json.loads(index_file.read_text())
    else:
        data = {
            "version": "1.0",
            "generated": datetime.now().isoformat(),
            "total_sessions": 0,
            "stats": {
                "decisions": 0,
                "ideas": 0,
                "questions": 0,
                "action_items": 0,
                "concepts": 0,
                "terms": 0,
            },
            "sessions": [],
        }

    # Update aggregate stats
    counts = {
        "decisions": len(result.decisions),
        "ideas": len(result.ideas),
        "questions": len(result.questions),
        "action_items": len(result.action_items),
        "concepts": len(result.concepts),
        "terms": len(result.terms),
    }

    for key in counts:
        data["stats"][key] += counts[key]

    # Add session entry
    session_entry = {
        "date": datetime.now().isoformat(),
        "file": input_file,
        "hash": file_hash[:12],
        "output_file": output_file,
        "counts": counts,
        "glossary_matches": glossary_matches,
        "glossary_unknown": glossary_unknown,
    }

    data["sessions"].append(session_entry)
    data["total_sessions"] = len(data["sessions"])
    data["generated"] = datetime.now().isoformat()

    # Write atomically (write to temp file, then rename)
    temp_file = index_file.with_suffix(".json.tmp")
    temp_file.write_text(json.dumps(data, indent=2))
    temp_file.replace(index_file)


def add_glossary_section(
    markdown_path: str,
    matches: list[dict[str, Any]],
    unknown: list[dict[str, Any]],
) -> None:  # noqa: D103
    """
    Append a Glossary Cross-Reference section to an existing markdown file.

    Args:
        markdown_path: Path to markdown file
        matches: List of matched glossary terms (dicts with 'term' key)
        unknown: List of unknown terms (dicts with 'term' key)
    """
    filepath = Path(markdown_path)

    # Build glossary section
    lines = []
    lines.append("## Glossary Cross-Reference")
    lines.append("")

    # Add matched terms
    for item in matches:
        term = item.get("term", "")
        lines.append(f"- ✓ **{term}** — matches known concept")

    # Add unknown terms
    for item in unknown:
        term = item.get("term", "")
        lines.append(f"- ? **{term}** — unknown term (not in glossary)")

    section = "\n".join(lines)

    # Append to file
    current_content = filepath.read_text()
    if not current_content.endswith("\n"):
        current_content += "\n"

    new_content = current_content + "\n" + section + "\n"
    filepath.write_text(new_content)
