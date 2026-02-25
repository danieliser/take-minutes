"""Batch command — process multiple historical session transcripts."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import click

from minutes.cli_utils import find_main_sessions, parse_min_size, parse_size, parse_since
from minutes.config import load_config
from minutes.dedup import DedupStore
from minutes.extractor import get_backend, process_transcript
from minutes.output import write_session_markdown
from minutes.parser import NO_FILTERS, parse_file


def handle_batch(
    project: str | None,
    since: str | None,
    min_size: str,
    max_size: str | None,
    output: str | None,
    dry_run: bool,
    no_embed: bool,
    sort: str,
    raw: bool,
    verbose: bool,
    mode: str,
    detail: bool,
    full: bool,
    strict: bool,
    chunk_size: int | None = None,
) -> None:
    """Batch process historical session transcripts."""
    config = load_config()
    if verbose:
        config.verbose = True
    if chunk_size:
        config.max_chunk_size = chunk_size
        config.chunk_size_override = True

    min_bytes = parse_min_size(min_size)
    max_bytes = parse_size(max_size) if max_size else None
    since_dt = parse_since(since) if since else None

    projects_dir = Path.home() / ".claude" / "projects"
    sessions = find_main_sessions(projects_dir, since=since_dt, min_size=min_bytes,
                                  max_size=max_bytes, project_filter=project, sort=sort)

    if not sessions:
        click.secho("No matching sessions found.", fg='yellow')
        return

    output_base = Path(output) if output else Path.home() / ".claude" / "minutes"

    from minutes.store import MinutesStore

    # Pre-scan: separate already-indexed from work-to-do
    pending: list[tuple[str, Path, str]] = []  # (project_key, session_file, file_hash)
    skipped = 0

    for project_key, session_file in sessions:
        output_dir = output_base / project_key
        output_dir.mkdir(parents=True, exist_ok=True)
        dedup = DedupStore(str(output_dir))

        try:
            file_hash = dedup.compute_hash(str(session_file))
        except OSError:
            skipped += 1
            continue

        db_path = output_dir / "minutes.db"
        store = MinutesStore(db_path)
        indexed = store.is_indexed(session_file.stem, file_hash)
        store.close()

        if indexed:
            skipped += 1
        else:
            pending.append((project_key, session_file, file_hash))

    if not pending:
        click.secho(f"All {len(sessions)} sessions already indexed, nothing to do.", fg='green')
        return

    click.secho(f"{len(pending)} to process, {skipped} already indexed", fg='cyan')

    if dry_run:
        for project_key, f, _hash in pending:
            stat = f.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            click.echo(f"  {project_key}  {f.name}  {size_kb:.0f}KB  {mtime}")
        return

    # Init backend based on mode
    if mode in ('changes', 'stats'):
        backend = None
        backend_name = "structural"
    elif mode in ('intent', 'review'):
        try:
            backend, backend_name = get_backend(config)
        except RuntimeError as e:
            click.secho(f"Error: {e}", fg='red', err=True)
            sys.exit(2)
    else:  # extract
        try:
            backend, backend_name = get_backend(config)
        except RuntimeError as e:
            click.secho(f"Error: {e}", fg='red', err=True)
            sys.exit(2)

    processed = 0
    errors = 0

    with BatchProgress(len(pending)) as bp:
        for project_key, session_file, file_hash in pending:
            output_dir = output_base / project_key
            db_path = output_dir / "minutes.db"
            session_id = session_file.stem

            store = MinutesStore(db_path)
            dedup = DedupStore(str(output_dir))

            try:
                if mode == 'changes':
                    _batch_changes(session_file, output_dir, full, strict)
                    processed += 1
                elif mode == 'stats':
                    _batch_stats(session_file, output_dir, detail, strict)
                    processed += 1
                elif mode == 'intent':
                    result = _batch_intent(session_file, output_dir, backend, strict)
                    if result == 'skipped':
                        skipped += 1
                    else:
                        processed += 1
                elif mode == 'review':
                    _batch_review(session_file, output_dir, backend, strict)
                    processed += 1
                else:  # extract
                    def _on_chunks_ready(total: int, done: int) -> None:
                        bp.start_file(session_file.name, total, done)

                    _batch_extract(
                        session_file, output_dir, store, dedup, backend, backend_name,
                        config, raw, session_id, project_key, file_hash,
                        on_chunk_done=bp.advance_chunk,
                        on_chunks_ready=_on_chunks_ready,
                        log=bp.log,
                    )
                    processed += 1
            except Exception as e:
                bp.log(f"  [red]Error: {e}[/red]")
                errors += 1

            bp.finish_file()
            store.close()

    # Embed unembedded items (extract mode only)
    if not no_embed and processed > 0 and mode == 'extract':
        _generate_embeddings(output_base)

    click.echo(f"\nBatch complete: {processed} processed, {skipped} skipped, {errors} errors")


def _batch_changes(session_file: Path, output_dir: Path, full: bool, strict: bool) -> None:
    from minutes.changes import parse_changes, format_changes_markdown
    timeline = parse_changes(str(session_file), strict=strict)
    markdown = format_changes_markdown(timeline, session_file.name, full=full)
    out_file = output_dir / f"{session_file.stem}-changes.md"
    out_file.write_text(markdown)
    click.secho(f"    ✓ {timeline.total_edits}e {timeline.total_writes}w", fg='green')


def _batch_stats(session_file: Path, output_dir: Path, detail: bool, strict: bool) -> None:
    from minutes.changes import collect_stats, format_stats_markdown
    stats_result = collect_stats(str(session_file), detail=detail, strict=strict)
    markdown = format_stats_markdown(stats_result, session_file.name, detail=detail)
    out_file = output_dir / f"{session_file.stem}-stats.md"
    out_file.write_text(markdown)
    click.secho(f"    ✓ {stats_result.total_calls} calls", fg='green')


def _batch_intent(session_file: Path, output_dir: Path, backend, strict: bool) -> str:
    from minutes.intent import extract_user_prompts, summarize_intent, format_intent_markdown
    prompts = extract_user_prompts(str(session_file), strict=strict)
    if not prompts:
        click.secho(f"    Skip (no prompts)", fg='yellow')
        return 'skipped'
    try:
        intent = summarize_intent(backend, prompts)
        markdown = format_intent_markdown(intent)
        out_file = output_dir / f"{session_file.stem}-intent.md"
        out_file.write_text(markdown)
        click.secho(f"    ✓ {intent.primary_goal[:60] if intent.primary_goal else 'no goal'}", fg='green')
        return 'processed'
    except Exception as e:
        click.secho(f"    Warning: LLM failed for {session_file.name}: {e}", fg='yellow', err=True)
        return 'skipped'


def _batch_review(session_file: Path, output_dir: Path, backend, strict: bool) -> None:
    from minutes.review import run_review
    from minutes.review_format import format_review_markdown
    result = run_review(backend, str(session_file), strict=strict)
    markdown = format_review_markdown(result, session_file.name)
    out_file = output_dir / f"{session_file.stem}-review.md"
    out_file.write_text(markdown)
    click.secho(f"    ✓ alignment: {result.alignment_score:.2f} | covered: {len(result.covered)} | gaps: {len(result.gaps)}", fg='green')


def _batch_extract(
    session_file: Path,
    output_dir: Path,
    store,
    dedup,
    backend,
    backend_name: str,
    config,
    raw: bool,
    session_id: str,
    project_key: str,
    file_hash: str,
    on_chunk_done=None,
    on_chunks_ready=None,
    log=None,
) -> None:
    batch_filter = NO_FILTERS if raw else None
    text, metadata = parse_file(str(session_file), filter_config=batch_filter)
    result = process_transcript(
        backend, config, text,
        file_size=session_file.stat().st_size,
        session_id=session_id,
        file_hash=file_hash,
        store=store,
        on_chunk_done=on_chunk_done,
        on_chunks_ready=on_chunks_ready,
    )

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

    dedup.record(file_hash, markdown_path, input_file=str(session_file))

    counts = (f"{len(result.decisions)}d {len(result.ideas)}i "
              f"{len(result.questions)}q {len(result.action_items)}a "
              f"{len(result.concepts)}c {len(result.terms)}t")
    _log = log or click.echo
    _log(f"    ✓ {counts}")


def _generate_embeddings(output_base: Path) -> None:
    from minutes import embeddings as emb
    from minutes.embeddings import DEFAULT_MODEL
    from minutes.store import MinutesStore

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
