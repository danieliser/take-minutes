"""Shared CLI utility functions."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path


def parse_since(since: str) -> datetime:
    """Parse --since value: ISO date or relative like '2w', '7d', '30d'."""
    m = re.match(r'^(\d+)([dwm])$', since)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {'d': timedelta(days=n), 'w': timedelta(weeks=n),
                 'm': timedelta(days=n * 30)}[unit]
        return datetime.now() - delta
    return datetime.fromisoformat(since)


def parse_size(size_str: str) -> int:
    """Parse a human size string like '10KB', '1.5MB', '2GB' -> bytes."""
    size_match = re.match(r'^([\d.]+)\s*(KB|MB|GB|B)?$', size_str, re.IGNORECASE)
    if size_match:
        n = float(size_match.group(1))
        unit = (size_match.group(2) or 'B').upper()
        return int(n * {'B': 1, 'KB': 1024, 'MB': 1024 * 1024, 'GB': 1024 ** 3}[unit])
    return 10240


# Backward compat alias
parse_min_size = parse_size


def find_main_sessions(
    projects_dir: Path,
    since: datetime | None = None,
    min_size: int = 10240,
    max_size: int | None = None,
    project_filter: str | None = None,
    sort: str = "date",
) -> list[tuple[str, Path]]:
    """Find main-thread session JSONL files (skip subagents/)."""
    results = []
    if not projects_dir.exists():
        return results

    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        project_key = project_dir.name

        if project_filter and project_filter not in project_key:
            continue

        for f in sorted(project_dir.glob("*.jsonl")):
            if "subagents" in f.parts:
                continue
            try:
                stat = f.stat()
            except OSError:
                continue
            if stat.st_size < min_size:
                continue
            if max_size and stat.st_size > max_size:
                continue
            if since and datetime.fromtimestamp(stat.st_mtime) < since:
                continue
            results.append((project_key, f))

    if sort == "date":
        results.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    elif sort == "date-asc":
        results.sort(key=lambda x: x[1].stat().st_mtime)
    elif sort == "size":
        results.sort(key=lambda x: x[1].stat().st_size, reverse=True)
    elif sort == "size-asc":
        results.sort(key=lambda x: x[1].stat().st_size)
    elif sort == "project":
        results.sort(key=lambda x: (x[0], x[1].name))

    return results
