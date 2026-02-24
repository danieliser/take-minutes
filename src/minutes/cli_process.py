"""Process command — extract structured knowledge from a single transcript."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from minutes.config import load_config
from minutes.dedup import DedupStore
from minutes.extractor import get_backend, process_transcript
from minutes.glossary import load_glossary, match_terms
from minutes.output import (
    add_glossary_section,
    append_session_log,
    update_index,
    write_session_markdown,
)
from minutes.parser import NO_FILTERS, parse_file


def handle_process(
    file: str,
    output: str | None,
    no_dedup: bool,
    raw: bool,
    verbose: bool,
    mode: str,
    detail: bool,
    full: bool,
    strict: bool,
) -> None:
    """Process a single transcript file."""
    try:
        config = load_config()

        if output:
            config.output_dir = output
        if verbose:
            config.verbose = verbose

        # Handle v0.4.0 extraction modes
        if mode != 'extract':
            _handle_mode(file, output, config, mode, detail, full, strict)
            return

        # Extract mode — original behavior
        output_path = Path(config.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            click.secho(f"Error: Cannot write to output directory: {e}", fg='red', err=True)
            sys.exit(5)

        filter_config = NO_FILTERS if raw else None
        try:
            text, metadata = parse_file(file, filter_config=filter_config)
        except FileNotFoundError as e:
            click.secho(f"Error: {e}", fg='red', err=True)
            sys.exit(1)
        except Exception as e:
            click.secho(f"Error: Malformed input file: {e}", fg='red', err=True)
            sys.exit(3)

        dedup_store = DedupStore(config.output_dir)
        file_hash = dedup_store.compute_hash(file)

        is_cached = False
        result = None
        if not no_dedup:
            existing_output = dedup_store.is_processed(file_hash)
            if existing_output:
                click.secho(f"✓ Already processed: {existing_output}", fg='green')
                is_cached = True
                from minutes.models import ExtractionResult
                result = ExtractionResult()

        if result is None:
            try:
                backend, backend_name = get_backend(config)
            except RuntimeError as e:
                click.secho(f"Error: {e}", fg='red', err=True)
                sys.exit(2)

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

        if 'messages' in metadata:
            content_metric = f"{metadata['messages']} messages"
        elif 'chars' in metadata:
            content_metric = f"{metadata['chars']} chars"
        else:
            content_metric = "0 items"

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

        glossary = load_glossary(config.glossary_path)
        glossary_matches, glossary_unknown = match_terms(result, glossary)

        if glossary and (glossary_matches or glossary_unknown):
            try:
                add_glossary_section(markdown_path, glossary_matches, glossary_unknown)
            except (OSError, PermissionError) as e:
                if config.verbose:
                    click.secho(f"Warning: Could not add glossary section: {e}", fg='yellow', err=True)

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

        if not is_cached:
            dedup_store.record(file_hash, markdown_path, input_file=file)

        try:
            from minutes.store import MinutesStore
            db_path = Path(config.output_dir) / "minutes.db"
            store = MinutesStore(db_path)
            session_id = Path(file).stem
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


def _handle_mode(
    file: str,
    output: str | None,
    config,
    mode: str,
    detail: bool,
    full: bool,
    strict: bool,
) -> None:
    """Handle non-extract modes (changes, stats, intent)."""
    if not file.endswith('.jsonl'):
        click.secho(f"Error: --mode {mode} requires a JSONL file. Got: {Path(file).name}", fg='red', err=True)
        sys.exit(1)

    if not Path(file).exists():
        click.secho(f"Error: File not found: {file}", fg='red', err=True)
        sys.exit(1)

    if detail and mode != 'stats':
        click.secho("Warning: --detail has no effect without --mode stats. Ignored.", fg='yellow', err=True)

    output_dir = Path(output) if output else Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file).stem

    if mode == 'changes':
        from minutes.changes import parse_changes, format_changes_markdown
        timeline = parse_changes(file, strict=strict)
        markdown = format_changes_markdown(timeline, Path(file).name, full=full)
        out_file = output_dir / f"{stem}-changes.md"
        out_file.write_text(markdown)
        click.secho(f"✓ Changes extracted: {timeline.total_edits} edits, {timeline.total_writes} writes", fg='green')
        click.echo(f"  Files modified: {len(timeline.files_modified)}")
        click.echo(f"  Output: {out_file}")

    elif mode == 'stats':
        from minutes.changes import collect_stats, format_stats_markdown
        stats = collect_stats(file, detail=detail, strict=strict)
        markdown = format_stats_markdown(stats, Path(file).name, detail=detail)
        out_file = output_dir / f"{stem}-stats.md"
        out_file.write_text(markdown)
        click.secho(f"✓ Stats collected: {stats.total_calls} tool calls", fg='green')
        click.echo(f"  Messages: {stats.user_prompt_count} user, {stats.assistant_turn_count} assistant")
        click.echo(f"  Output: {out_file}")

    elif mode == 'intent':
        from minutes.intent import extract_user_prompts, summarize_intent, format_intent_markdown
        prompts = extract_user_prompts(file, strict=strict)

        if not prompts:
            click.secho("No user prompts found in session.", fg='yellow')
            return

        try:
            backend, backend_name = get_backend(config)
        except RuntimeError as e:
            click.secho(f"Error: {e}", fg='red', err=True)
            sys.exit(2)

        intent = summarize_intent(backend, prompts)
        markdown = format_intent_markdown(intent)
        out_file = output_dir / f"{stem}-intent.md"
        out_file.write_text(markdown)

        if intent.primary_goal:
            click.secho(f"✓ Intent summarized: {intent.primary_goal[:80]}", fg='green')
        else:
            click.secho("Warning: LLM summarization failed. Showing extracted prompts only.", fg='yellow', err=True)
        click.echo(f"  Prompts analyzed: {len(prompts)}")
        click.echo(f"  Output: {out_file}")
