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
    """Progress tracker with chunk-based ETA.

    Main bar: global chunk progress across all files (drives the ETA).
    Per-file bar: chunk progress within current file (transient).
    Total adjusts as real chunk counts replace estimates.
    """

    def __init__(self, total_files: int, estimated_chunks: int = 0) -> None:
        self.enabled = sys.stderr.isatty()
        self.total_files = total_files
        self.files_done = 0
        self._running_total = estimated_chunks
        self._current_file_estimate = 0
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
        self.main_task = self.progress.add_task(
            self._main_desc(),
            total=max(estimated_chunks, 1),
        )
        self.chunk_task_id: int | None = None

    def _main_desc(self, filename: str | None = None) -> str:
        label = f"File {self.files_done + 1}/{self.total_files}"
        if filename:
            short = filename[:30] if len(filename) > 30 else filename
            return f"{label}  {short}"
        return label

    def set_file_estimate(self, estimated: int) -> None:
        """Set the estimated chunk count for the upcoming file.

        Called before start_file so we can compute the delta when the
        real count arrives.
        """
        self._current_file_estimate = estimated

    def start_file(
        self, filename: str, total_chunks: int, completed_chunks: int = 0,
    ) -> None:
        """Begin tracking a new file's chunk extraction."""
        if not self.enabled:
            return

        # Adjust main bar total: replace estimate with actual
        delta = total_chunks - self._current_file_estimate
        if delta != 0:
            self._running_total += delta
            self.progress.update(self.main_task, total=self._running_total)
        self._current_file_estimate = 0

        self.progress.update(
            self.main_task, description=self._main_desc(filename),
        )

        if total_chunks > 1:
            if self.chunk_task_id is not None:
                self.progress.remove_task(self.chunk_task_id)
            self.chunk_task_id = self.progress.add_task(
                "  Chunks",
                total=total_chunks, completed=completed_chunks,
            )
        else:
            if self.chunk_task_id is not None:
                self.progress.remove_task(self.chunk_task_id)
                self.chunk_task_id = None

    def advance_chunk(self) -> None:
        """Mark one chunk as complete (advances both main and per-file bars)."""
        if not self.enabled:
            return
        self.progress.advance(self.main_task)
        if self.chunk_task_id is not None:
            self.progress.advance(self.chunk_task_id)

    def finish_file(self) -> None:
        """Mark current file as complete."""
        if not self.enabled:
            return
        self.files_done += 1
        if self.chunk_task_id is not None:
            self.progress.remove_task(self.chunk_task_id)
            self.chunk_task_id = None
        self.progress.update(
            self.main_task, description=self._main_desc(),
        )

    def log(self, message: str) -> None:
        """Print a message above the progress bars."""
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


def estimate_chunks(file_size: int, chunk_size: int) -> int:
    """Estimate number of chunks from raw JSONL file size.

    JSONL overhead is roughly 2x vs extracted text content (JSON structure,
    tool call metadata, system messages filtered out). Using 2x is slightly
    conservative — better to overestimate chunks than underestimate.
    """
    transcript_estimate = file_size // 2
    if transcript_estimate <= chunk_size:
        return 1
    return max(1, -(-transcript_estimate // chunk_size))  # ceil division


def _suppress_noisy_loggers() -> None:
    """Silence loggers that spam during LLM requests."""
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
