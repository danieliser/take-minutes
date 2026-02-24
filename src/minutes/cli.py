"""Click CLI entry point for minutes tool."""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click

from minutes.config import load_config
from minutes.dedup import DedupStore
from minutes.extractor import get_backend, process_transcript
from minutes.output import (
    add_glossary_section,
    append_session_log,
    update_index,
    write_session_markdown,
)
from minutes.glossary import load_glossary, match_terms
from minutes.parser import parse_file, NO_FILTERS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def main() -> None:  # noqa: D103
    """minutes — Extract structured knowledge from conversation transcripts."""
    pass


@main.command()
@click.argument('file', type=click.Path(exists=False))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--no-dedup', is_flag=True, help='Skip deduplication check')
@click.option('--raw', is_flag=True, help='Disable noise filters (keep tool results, system reminders, etc.)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def process(file: str, output: str | None, no_dedup: bool, raw: bool, verbose: bool) -> None:  # noqa: D103
    """Process a transcript and extract structured knowledge.

    \b
    Examples:
        minutes process session.jsonl
        minutes process meeting.txt -o ./output
        minutes process recording.srt --no-dedup
    """
    try:
        # Load configuration
        config = load_config()

        # Override config with CLI flags
        if output:
            config.output_dir = output
        if verbose:
            config.verbose = verbose

        # Ensure output directory exists
        output_path = Path(config.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            click.secho(f"Error: Cannot write to output directory: {e}", fg='red', err=True)
            sys.exit(5)

        # Parse input file
        filter_config = NO_FILTERS if raw else None
        try:
            text, metadata = parse_file(file, filter_config=filter_config)
        except FileNotFoundError as e:
            click.secho(f"Error: {e}", fg='red', err=True)
            sys.exit(1)
        except Exception as e:
            click.secho(f"Error: Malformed input file: {e}", fg='red', err=True)
            sys.exit(3)

        # Compute file hash for deduplication
        dedup_store = DedupStore(config.output_dir)
        file_hash = dedup_store.compute_hash(file)

        # Check if already processed (unless --no-dedup)
        is_cached = False
        if not no_dedup:
            existing_output = dedup_store.is_processed(file_hash)
            if existing_output:
                click.secho(f"✓ Already processed: {existing_output}", fg='green')
                is_cached = True
                # Load the existing result to append to logs/index
                # For simplicity, we'll reconstruct a minimal result
                # In production, you might want to store the result JSON
                from minutes.models import ExtractionResult
                result = ExtractionResult()
            else:
                result = None
        else:
            result = None

        # Get LLM backend
        if result is None:
            try:
                backend, backend_name = get_backend(config)
            except RuntimeError as e:
                click.secho(f"Error: {e}", fg='red', err=True)
                sys.exit(2)

            # Extract structured knowledge
            try:
                result = process_transcript(backend, config, text)
            except Exception as e:
                click.secho(f"Error: Extraction failed: {e}", fg='red', err=True)
                if config.verbose:
                    import traceback
                    traceback.print_exc()
                sys.exit(4)
        else:
            backend_name = "cached"

        # Prepare metadata for output functions
        if 'messages' in metadata:
            content_metric = f"{metadata['messages']} messages"
        elif 'chars' in metadata:
            content_metric = f"{metadata['chars']} chars"
        else:
            content_metric = "0 items"

        # Write session markdown
        try:
            markdown_path = write_session_markdown(
                result=result,
                metadata={'content_metric': content_metric, **metadata},
                output_dir=config.output_dir,
                file_hash=file_hash,
                input_file=Path(file).name,
                backend_name=backend_name,
            )
        except (OSError, PermissionError) as e:
            click.secho(f"Error: Cannot write markdown file: {e}", fg='red', err=True)
            sys.exit(5)

        # Load and match glossary
        glossary = load_glossary(config.glossary_path)
        glossary_matches, glossary_unknown = match_terms(result, glossary)

        # Add glossary section to markdown
        if glossary and (glossary_matches or glossary_unknown):
            try:
                add_glossary_section(markdown_path, glossary_matches, glossary_unknown)
            except (OSError, PermissionError) as e:
                if config.verbose:
                    click.secho(f"Warning: Could not add glossary section: {e}", fg='yellow', err=True)

        # Append session log
        try:
            append_session_log(
                output_dir=config.output_dir,
                input_file=Path(file).name,
                metadata={'content_metric': content_metric},
                result=result,
                file_hash=file_hash,
                is_cached=is_cached,
            )
        except (OSError, PermissionError) as e:
            if config.verbose:
                click.secho(f"Warning: Could not append session log: {e}", fg='yellow', err=True)

        # Update index
        try:
            update_index(
                output_dir=config.output_dir,
                input_file=Path(file).name,
                result=result,
                file_hash=file_hash,
                output_file=Path(markdown_path).name,
                glossary_matches=len(glossary_matches),
                glossary_unknown=len(glossary_unknown),
            )
        except (OSError, PermissionError) as e:
            if config.verbose:
                click.secho(f"Warning: Could not update index: {e}", fg='yellow', err=True)

        # Record in dedup store (unless cached)
        if not is_cached:
            dedup_store.record(file_hash, markdown_path, input_file=file)

        # Index in SQLite store
        try:
            from minutes.store import MinutesStore
            db_path = Path(config.output_dir) / "minutes.db"
            store = MinutesStore(db_path)
            # Derive session_id from input filename (strip extension)
            session_id = Path(file).stem
            # Derive project_key from output_dir path
            project_key = Path(config.output_dir).name or "unknown"
            file_stat = Path(file).stat() if Path(file).exists() else None
            store.upsert_session(
                session_id=session_id,
                project_key=project_key,
                input_file=str(Path(file).resolve()),
                result=result,
                output_file=markdown_path,
                file_hash=file_hash,
                file_size=file_stat.st_size if file_stat else 0,
                message_count=metadata.get('messages', 0),
                transcript_chars=metadata.get('chars', 0),
            )
            store.close()
        except Exception as e:
            if config.verbose:
                click.secho(f"Warning: Could not index in store: {e}", fg='yellow', err=True)

        # Print summary
        click.secho("✓ Processing complete", fg='green')
        click.echo(f"  Decisions: {len(result.decisions)}")
        click.echo(f"  Ideas: {len(result.ideas)}")
        click.echo(f"  Questions: {len(result.questions)}")
        click.echo(f"  Action Items: {len(result.action_items)}")
        click.echo(f"  Concepts: {len(result.concepts)}")
        click.echo(f"  Terms: {len(result.terms)}")
        if glossary_matches or glossary_unknown:
            click.echo(f"  Glossary Matches: {len(glossary_matches)} | Unknown: {len(glossary_unknown)}")
        click.echo(f"  Output: {markdown_path}")

    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"Unexpected error: {e}", fg='red', err=True)
        if config.verbose if 'config' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(4)


@main.command()
@click.argument('directory', type=click.Path(exists=False))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--interval', type=int, default=5, help='Poll interval in seconds')
def watch(directory: str, output: str | None, interval: int) -> None:  # noqa: D103
    """Watch a directory for new transcripts and process them.

    DIRECTORY: Directory to watch for .txt, .md, .jsonl files
    """
    try:
        dir_path = Path(directory)

        # Check if directory exists
        if not dir_path.exists():
            click.secho(f"Error: Directory not found: {directory}", fg='red', err=True)
            sys.exit(1)

        if not dir_path.is_dir():
            click.secho(f"Error: Not a directory: {directory}", fg='red', err=True)
            sys.exit(1)

        # Load configuration
        config = load_config()
        if output:
            config.output_dir = output

        # Get list of files to process initially
        processed_hashes = set()

        click.secho(f"Watching {directory} for new files...", fg='green')
        click.secho(f"Interval: {interval}s | Output: {config.output_dir}", fg='cyan')

        try:
            while True:
                # Scan directory for supported files (flat, no recursion)
                supported_extensions = {'.txt', '.md', '.markdown', '.jsonl'}
                files_to_check = []

                try:
                    for item in dir_path.iterdir():
                        # Skip hidden files and files starting with ~
                        if item.name.startswith('.') or item.name.startswith('~'):
                            continue

                        # Only process files (not directories)
                        if not item.is_file():
                            continue

                        # Check if supported extension
                        if item.suffix.lower() in supported_extensions:
                            files_to_check.append(item)
                except OSError as e:
                    click.secho(f"Warning: Error scanning directory: {e}", fg='yellow', err=True)
                    time.sleep(interval)
                    continue

                # Process each file
                for file_path in files_to_check:
                    try:
                        file_hash = DedupStore(config.output_dir).compute_hash(str(file_path))

                        # Skip if already processed
                        if file_hash in processed_hashes:
                            continue

                        # Mark as processed
                        processed_hashes.add(file_hash)

                        # Process the file (silently unless verbose)
                        click.secho(f"Processing: {file_path.name}", fg='cyan')

                        try:
                            text, metadata = parse_file(str(file_path))

                            # Get backend
                            backend, backend_name = get_backend(config)

                            # Extract
                            result = process_transcript(backend, config, text)

                            # Write output
                            dedup_store = DedupStore(config.output_dir)
                            markdown_path = write_session_markdown(
                                result=result,
                                metadata=metadata,
                                output_dir=config.output_dir,
                                file_hash=file_hash,
                                input_file=file_path.name,
                                backend_name=backend_name,
                            )

                            # Record in dedup
                            dedup_store.record(file_hash, markdown_path)

                            click.secho(f"  ✓ {file_path.name}", fg='green')

                        except Exception as e:
                            click.secho(f"  Error processing {file_path.name}: {e}", fg='red', err=True)

                    except Exception as e:
                        click.secho(f"  Warning: {e}", fg='yellow', err=True)

                # Sleep before next scan
                time.sleep(interval)

        except KeyboardInterrupt:
            click.secho("\nShutdown gracefully.", fg='cyan')
            sys.exit(0)

    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"Watch error: {e}", fg='red', err=True)
        sys.exit(1)


@main.command()
@click.option('--env', is_flag=True, help='Show .env file path')
def config(env: bool) -> None:  # noqa: D103
    """Display active configuration values.

    \b
    Examples:
        minutes config
        minutes config --env
    """
    try:
        cfg = load_config()

        if env:
            # Show .env file info
            env_file = Path('.env')
            click.echo(f".env file: {env_file.resolve()}")
            click.echo(f"Exists: {env_file.exists()}")
            click.echo("")

        # Display config values
        click.echo("Active Configuration:")
        click.echo(f"  Gateway Model: {cfg.gateway_model}")
        click.echo(f"  Gateway URL: {cfg.gateway_url}")
        click.echo(f"  Output Directory: {cfg.output_dir}")
        click.echo(f"  Glossary Path: {cfg.glossary_path}")
        click.echo(f"  Max Chunk Size: {cfg.max_chunk_size}")
        click.echo(f"  Chunk Overlap: {cfg.chunk_overlap}")
        click.echo(f"  Max Retries: {cfg.max_retries}")
        click.echo(f"  Verbose: {cfg.verbose}")

    except Exception as e:
        click.secho(f"Error loading configuration: {e}", fg='red', err=True)
        sys.exit(1)


@main.command()
def setup() -> None:  # noqa: D103
    """Set up minutes tool: check dependencies and cache models.

    \b
    Examples:
        minutes setup
    """
    try:
        # Check if sentence-transformers is installed
        try:
            import sentence_transformers
        except ImportError:
            click.secho("Search extras not installed. Run: pip install 'take-minutes[search]'", fg='yellow')
            return

        # Load the all-mpnet-base-v2 model to cache
        click.echo("Loading embedding model to cache...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-mpnet-base-v2')
            click.secho("✓ Model cached successfully", fg='green')
        except Exception as e:
            click.secho(f"Error loading model: {e}", fg='red', err=True)
            sys.exit(1)

        # Check gateway connectivity
        click.echo("Checking gateway connectivity...")
        try:
            import urllib.request
            urllib.request.urlopen('http://localhost:8800', timeout=2)
            click.secho("✓ Gateway is reachable at http://localhost:8800", fg='green')
        except Exception as e:
            click.secho(f"⚠ Gateway not reachable: {e}", fg='yellow')

        # Print next steps
        click.echo("")
        click.secho("Setup complete! Next steps:", fg='cyan')
        click.echo("  1. Configure .env with your API keys and gateway settings")
        click.echo("  2. Run 'minutes process <file>' to extract knowledge from a transcript")
        click.echo("  3. Run 'minutes batch' to process historical sessions")
        click.echo("  4. Run 'minutes search <query>' to find knowledge across sessions")

    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"Setup error: {e}", fg='red', err=True)
        sys.exit(1)


def _parse_since(since: str) -> datetime:  # noqa: D103
    """Parse --since value: ISO date or relative like '2w', '7d', '30d'."""
    m = re.match(r'^(\d+)([dwm])$', since)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {'d': timedelta(days=n), 'w': timedelta(weeks=n),
                 'm': timedelta(days=n * 30)}[unit]
        return datetime.now() - delta
    return datetime.fromisoformat(since)


def _find_main_sessions(
    projects_dir: Path,
    since: datetime | None = None,
    min_size: int = 10240,
    project_filter: str | None = None,
    sort: str = "date",
) -> list[tuple[str, Path]]:  # noqa: D103
    """Find main-thread session JSONL files (skip subagents/).

    Args:
        sort: 'date' (newest first), 'date-asc' (oldest first),
              'size' (largest first), 'size-asc' (smallest first),
              'project' (alphabetical by project key)
    """
    results = []
    if not projects_dir.exists():
        return results

    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        project_key = project_dir.name

        # Substring match: --project "persistence" matches "-Users-danieliser-Toolkit-persistence"
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
            if since and datetime.fromtimestamp(stat.st_mtime) < since:
                continue
            results.append((project_key, f))

    # Sort results
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


@main.command()
@click.option('--project', type=str, default=None, help='Filter by project key')
@click.option('--since', type=str, default=None, help='Only files modified after (ISO date or 2w/7d/30d)')
@click.option('--min-size', type=str, default='10KB', help='Skip files smaller than this')
@click.option('--output', '-o', type=click.Path(), default=None, help='Output base directory')
@click.option('--dry-run', is_flag=True, help='Show what would be processed')
@click.option('--no-embed', is_flag=True, help='Skip embedding generation')
@click.option('--sort', type=click.Choice(['date', 'date-asc', 'size', 'size-asc', 'project']), default='date', help='Sort order for sessions')
@click.option('--raw', is_flag=True, help='Disable noise filters (keep tool results, system reminders, etc.)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def batch(project: str | None, since: str | None, min_size: str, output: str | None, dry_run: bool, no_embed: bool, sort: str, raw: bool, verbose: bool) -> None:  # noqa: D103
    """Batch process historical session transcripts.

    Scans ~/.claude/projects/ for main-thread JSONL files, extracts structured
    knowledge, indexes in SQLite, and generates embeddings for semantic search.

    \b
    Examples:
        minutes batch --since 2w
        minutes batch --project my-project --dry-run
        minutes batch --sort size --min-size 50KB

    \b
    --since formats:
        7d, 2w, 1m (relative) or 2026-02-01 (ISO date)
    """
    config = load_config()
    if verbose:
        config.verbose = True

    # Parse min-size (e.g., "10KB" -> 10240)
    size_match = re.match(r'^(\d+)\s*(KB|MB|B)?$', min_size, re.IGNORECASE)
    if size_match:
        n = int(size_match.group(1))
        unit = (size_match.group(2) or 'B').upper()
        min_bytes = n * {'B': 1, 'KB': 1024, 'MB': 1024 * 1024}[unit]
    else:
        min_bytes = 10240

    # Parse since
    since_dt = _parse_since(since) if since else None

    # Find sessions
    projects_dir = Path.home() / ".claude" / "projects"
    sessions = _find_main_sessions(projects_dir, since=since_dt, min_size=min_bytes,
                                   project_filter=project, sort=sort)

    if not sessions:
        click.secho("No matching sessions found.", fg='yellow')
        return

    click.secho(f"Found {len(sessions)} session(s) to process", fg='cyan')

    if dry_run:
        for project_key, f in sessions:
            stat = f.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            click.echo(f"  {project_key}  {f.name}  {size_kb:.0f}KB  {mtime}")
        return

    # Determine output base dir
    output_base = Path(output) if output else Path.home() / ".claude" / "minutes"

    # Get backend once for all extractions
    try:
        backend, backend_name = get_backend(config)
    except RuntimeError as e:
        click.secho(f"Error: {e}", fg='red', err=True)
        sys.exit(2)

    processed = 0
    skipped = 0
    errors = 0

    for project_key, session_file in sessions:
        output_dir = output_base / project_key
        output_dir.mkdir(parents=True, exist_ok=True)

        db_path = output_dir / "minutes.db"
        session_id = session_file.stem

        # Check if already indexed with same hash
        from minutes.store import MinutesStore
        store = MinutesStore(db_path)
        dedup = DedupStore(str(output_dir))

        try:
            file_hash = dedup.compute_hash(str(session_file))
        except OSError as e:
            click.secho(f"  Skip {session_file.name}: {e}", fg='yellow', err=True)
            store.close()
            skipped += 1
            continue

        if store.is_indexed(session_id, file_hash):
            if verbose:
                click.echo(f"  Skip (indexed): {session_file.name}")
            store.close()
            skipped += 1
            continue

        click.echo(f"  Processing: {session_file.name} ({session_file.stat().st_size / 1024:.0f}KB)")

        try:
            batch_filter = NO_FILTERS if raw else None
            text, metadata = parse_file(str(session_file), filter_config=batch_filter)
            result = process_transcript(backend, config, text)

            # Write markdown
            if 'messages' in metadata:
                content_metric = f"{metadata['messages']} messages"
            elif 'chars' in metadata:
                content_metric = f"{metadata['chars']} chars"
            else:
                content_metric = "0 items"

            markdown_path = write_session_markdown(
                result=result,
                metadata={'content_metric': content_metric, **metadata},
                output_dir=str(output_dir),
                file_hash=file_hash,
                input_file=session_file.name,
                backend_name=backend_name,
            )

            # Index in store
            store.upsert_session(
                session_id=session_id,
                project_key=project_key,
                input_file=str(session_file.resolve()),
                result=result,
                output_file=markdown_path,
                file_hash=file_hash,
                file_size=session_file.stat().st_size,
                message_count=metadata.get('messages', 0),
                transcript_chars=metadata.get('chars', 0),
            )

            # Record in dedup
            dedup.record(file_hash, markdown_path, input_file=str(session_file))

            counts = (f"{len(result.decisions)}d {len(result.ideas)}i "
                      f"{len(result.questions)}q {len(result.action_items)}a "
                      f"{len(result.concepts)}c {len(result.terms)}t")
            click.secho(f"    ✓ {counts}", fg='green')
            processed += 1

        except Exception as e:
            click.secho(f"    Error: {e}", fg='red', err=True)
            errors += 1

        store.close()

    # Embed all unembedded items across all project stores
    if not no_embed and processed > 0:
        from minutes import embeddings as emb
        from minutes.embeddings import DEFAULT_MODEL
        hf_id = emb.MODELS[DEFAULT_MODEL][0]
        dims = emb.MODELS[DEFAULT_MODEL][1]
        click.echo(f"\nGenerating embeddings ({hf_id}, {dims}d)...")

        try:
            for project_dir in output_base.iterdir():
                if not project_dir.is_dir():
                    continue
                db_file = project_dir / "minutes.db"
                if not db_file.exists():
                    continue

                store = MinutesStore(db_file)
                unembedded = store.get_unembedded_items(model=hf_id)
                if not unembedded:
                    store.close()
                    continue

                texts = [f"{item['content']} {item.get('detail') or ''}".strip()
                         for item in unembedded]
                item_ids = [item['id'] for item in unembedded]

                vectors = emb.embed(texts)
                store.store_embeddings(item_ids, vectors, model=hf_id)
                click.secho(f"  ✓ {project_dir.name}: {len(item_ids)} items", fg='green')
                store.close()

        except ImportError:
            click.secho("  Warning: sentence-transformers not installed, skipping embeddings. Install: pip install 'take-minutes[search]'", fg='yellow', err=True)
        except Exception as e:
            click.secho(f"  Embedding error: {e}", fg='red', err=True)

    # Summary
    click.echo(f"\nBatch complete: {processed} processed, {skipped} skipped, {errors} errors")


@main.command()
@click.argument('query')
@click.option('--project', type=str, default=None, help='Filter by project key')
@click.option('--category', type=str, default=None, help='Filter by category (decision, idea, question, action_item, concept, term)')
@click.option('--limit', type=int, default=10, help='Max results')
@click.option('--mode', type=click.Choice(['keyword', 'vector', 'hybrid']), default='hybrid', help='Search mode')
def search(query: str, project: str | None, category: str | None, limit: int, mode: str) -> None:  # noqa: D103
    """Search across all indexed session extractions.

    \b
    Examples:
        minutes search "authentication decision"
        minutes search "budget" --category decision
        minutes search "chunk size" --mode keyword --limit 5
    """
    output_base = Path.home() / ".claude" / "minutes"
    if not output_base.exists():
        click.secho("No minutes index found. Run 'batch' first.", fg='yellow')
        return

    from minutes import embeddings as emb
    from minutes.embeddings import DEFAULT_MODEL
    from minutes.store import MinutesStore
    hf_id = emb.MODELS[DEFAULT_MODEL][0]
    all_results = []

    # Get query embedding if needed
    query_embedding = None
    if mode in ('vector', 'hybrid'):
        try:
            query_embedding = emb.embed_one(query)
        except ImportError:
            if mode == 'vector':
                click.secho("Error: Install search extras: pip install 'take-minutes[search]'", fg='red', err=True)
                sys.exit(2)
            mode = 'keyword'
        except Exception as e:
            if mode == 'vector':
                click.secho(f"Error: {e}", fg='red', err=True)
                sys.exit(2)
            mode = 'keyword'

    for project_dir in sorted(output_base.iterdir()):
        if not project_dir.is_dir():
            continue
        if project and project_dir.name != project:
            continue
        db_file = project_dir / "minutes.db"
        if not db_file.exists():
            continue

        store = MinutesStore(db_file)

        try:
            if mode == 'keyword':
                results = store.search_keyword(query, category=category, limit=limit)
            elif mode == 'vector':
                results = store.search_vector(query_embedding, category=category, limit=limit, model=hf_id)
            else:
                results = store.search_hybrid(query, query_embedding, category=category, limit=limit, model=hf_id)

            for r in results:
                r['project_key'] = project_dir.name
            all_results.extend(results)
        except Exception as e:
            if 'verbose' in dir() and verbose:
                click.secho(f"  Warning searching {project_dir.name}: {e}", fg='yellow', err=True)

        store.close()

    if not all_results:
        click.secho("No results found.", fg='yellow')
        return

    # Sort by score descending, take top N
    score_key = 'rrf_score' if mode == 'hybrid' else 'score'
    all_results.sort(key=lambda r: r.get(score_key, r.get('score', 0)), reverse=True)
    all_results = all_results[:limit]

    # Display results
    for i, r in enumerate(all_results, 1):
        cat = r.get('category', '?')
        content = r.get('content', '')
        detail = r.get('detail', '')
        score = r.get(score_key, r.get('score', 0))
        proj = r.get('project_key', '?')
        session = r.get('session_id', '?')

        click.secho(f"\n{i}. [{cat}] ", fg='cyan', nl=False)
        click.echo(content)
        if detail:
            click.echo(f"   {detail}")
        click.secho(f"   score={score:.4f}  project={proj}  session={session}", fg='bright_black')


if __name__ == '__main__':
    main()
