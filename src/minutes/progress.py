"""Rich progress bars for batch processing."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class BatchProgress:
    """Dual-bar progress tracker: file-level + chunk-level.

    File bar: overall session progress with accurate ETA.
    Chunk bar: per-file chunk progress (only shown for multi-chunk files).
    """

    def __init__(self, total_files: int) -> None:
        self.enabled = sys.stderr.isatty()
        self.total_files = total_files
        if not self.enabled:
            return

        self.console = Console(stderr=True)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description:<45}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.file_task = self.progress.add_task("Files", total=total_files)
        self.chunk_task_id: int | None = None

    def start_file(self, filename: str, total_chunks: int, completed_chunks: int = 0) -> None:
        """Begin tracking a new file's chunk extraction."""
        if not self.enabled:
            return
        # Update file bar description to show current file
        short = filename[:40] if len(filename) > 40 else filename
        self.progress.update(self.file_task, description=f"Files  ({short})")

        # Only show chunk bar for multi-chunk files
        if total_chunks > 1:
            if self.chunk_task_id is not None:
                self.progress.remove_task(self.chunk_task_id)
            self.chunk_task_id = self.progress.add_task(
                f"  Chunks", total=total_chunks, completed=completed_chunks,
            )
        else:
            # Single chunk — no chunk bar needed
            if self.chunk_task_id is not None:
                self.progress.remove_task(self.chunk_task_id)
                self.chunk_task_id = None

    def advance_chunk(self) -> None:
        """Mark one chunk as complete."""
        if not self.enabled or self.chunk_task_id is None:
            return
        self.progress.advance(self.chunk_task_id)

    def finish_file(self) -> None:
        """Mark current file as complete."""
        if not self.enabled:
            return
        if self.chunk_task_id is not None:
            self.progress.remove_task(self.chunk_task_id)
            self.chunk_task_id = None
        self.progress.update(self.file_task, description="Files")
        self.progress.advance(self.file_task)

    def log(self, message: str) -> None:
        """Print a message above the progress bars without breaking the display."""
        if self.enabled:
            self.progress.print(message)
        else:
            import click
            click.echo(message)

    def __enter__(self) -> BatchProgress:
        if self.enabled:
            _suppress_noisy_loggers()
            self.progress.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self.enabled:
            self.progress.stop()


def _suppress_noisy_loggers() -> None:
    """Silence loggers that spam during LLM requests."""
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
