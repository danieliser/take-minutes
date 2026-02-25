"""Watch command — monitor directory for new transcripts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from minutes.config import load_config
from minutes.dedup import DedupStore
from minutes.extractor import get_backend, process_transcript
from minutes.output import write_session_markdown
from minutes.parser import parse_file


def handle_watch(directory: str, output: str | None, interval: int) -> None:
    """Watch a directory for new transcripts and process them."""
    try:
        dir_path = Path(directory)

        if not dir_path.exists():
            click.secho(f"Error: Directory not found: {directory}", fg='red', err=True)
            sys.exit(1)

        if not dir_path.is_dir():
            click.secho(f"Error: Not a directory: {directory}", fg='red', err=True)
            sys.exit(1)

        config = load_config()
        if output:
            config.output_dir = output

        processed_hashes: set[str] = set()

        click.secho(f"Watching {directory} for new files...", fg='green')
        click.secho(f"Interval: {interval}s | Output: {config.output_dir}", fg='cyan')

        try:
            while True:
                supported_extensions = {'.txt', '.md', '.markdown', '.jsonl'}
                files_to_check = []

                try:
                    for item in dir_path.iterdir():
                        if item.name.startswith('.') or item.name.startswith('~'):
                            continue
                        if not item.is_file():
                            continue
                        if item.suffix.lower() in supported_extensions:
                            files_to_check.append(item)
                except OSError as e:
                    click.secho(f"Warning: Error scanning directory: {e}", fg='yellow', err=True)
                    time.sleep(interval)
                    continue

                for file_path in files_to_check:
                    try:
                        file_hash = DedupStore(config.output_dir).compute_hash(str(file_path))

                        if file_hash in processed_hashes:
                            continue

                        processed_hashes.add(file_hash)
                        click.secho(f"Processing: {file_path.name}", fg='cyan')

                        try:
                            text, metadata = parse_file(str(file_path))
                            backend, backend_name = get_backend(config)
                            result = process_transcript(backend, config, text)

                            dedup_store = DedupStore(config.output_dir)
                            markdown_path = write_session_markdown(
                                result=result,
                                metadata=metadata,
                                output_dir=config.output_dir,
                                file_hash=file_hash,
                                input_file=file_path.name,
                                backend_name=backend_name,
                            )
                            dedup_store.record(file_hash, markdown_path)
                            click.secho(f"  ✓ {file_path.name}", fg='green')

                        except Exception as e:
                            click.secho(f"  Error processing {file_path.name}: {e}", fg='red', err=True)

                    except Exception as e:
                        click.secho(f"  Warning: {e}", fg='yellow', err=True)

                time.sleep(interval)

        except KeyboardInterrupt:
            click.secho("\nShutdown gracefully.", fg='cyan')
            sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"Watch error: {e}", fg='red', err=True)
        sys.exit(1)
