"""Click CLI entry point for minutes tool."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from minutes.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@click.group()
def main() -> None:
    """minutes — Extract structured knowledge from conversation transcripts."""
    pass


@main.command()
@click.argument('file', type=click.Path(exists=False))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--no-dedup', is_flag=True, help='Skip deduplication check')
@click.option('--raw', is_flag=True, help='Disable noise filters')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--mode', type=click.Choice(['extract', 'changes', 'intent', 'stats', 'review']), default='extract', help='Extraction mode')
@click.option('--detail', is_flag=True, help='Include full tool call log (--mode stats only)')
@click.option('--full', is_flag=True, help='Disable output truncation (--mode changes only)')
@click.option('--strict', is_flag=True, help='Fail on malformed JSONL')
def process(file, output, no_dedup, raw, verbose, mode, detail, full, strict):
    """Process a transcript and extract structured knowledge."""
    from minutes.cli_process import handle_process
    handle_process(file, output, no_dedup, raw, verbose, mode, detail, full, strict)


@main.command()
@click.argument('directory', type=click.Path(exists=False))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--interval', type=int, default=5, help='Poll interval in seconds')
def watch(directory, output, interval):
    """Watch a directory for new transcripts and process them."""
    from minutes.cli_watch import handle_watch
    handle_watch(directory, output, interval)


@main.command()
@click.option('--env', is_flag=True, help='Show .env file path')
def config(env):
    """Display active configuration values."""
    try:
        cfg = load_config()

        if env:
            env_file = Path('.env')
            click.echo(f".env file: {env_file.resolve()}")
            click.echo(f"Exists: {env_file.exists()}")
            click.echo("")

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
def setup():
    """Set up minutes tool: check dependencies and cache models."""
    try:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            click.secho("Search extras not installed. Run: pip install 'take-minutes[search]'", fg='yellow')
            return

        click.echo("Loading embedding model to cache...")
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer('all-mpnet-base-v2')
            click.secho("✓ Model cached successfully", fg='green')
        except Exception as e:
            click.secho(f"Error loading model: {e}", fg='red', err=True)
            sys.exit(1)

        click.echo("Checking gateway connectivity...")
        try:
            import urllib.request
            urllib.request.urlopen('http://localhost:8800', timeout=2)
            click.secho("✓ Gateway is reachable at http://localhost:8800", fg='green')
        except Exception as e:
            click.secho(f"⚠ Gateway not reachable: {e}", fg='yellow')

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


@main.command()
@click.option('--project', type=str, default=None, help='Filter by project key')
@click.option('--since', type=str, default=None, help='Only files modified after (ISO date or 2w/7d/30d)')
@click.option('--min-size', type=str, default='10KB', help='Skip files smaller than this')
@click.option('--output', '-o', type=click.Path(), default=None, help='Output base directory')
@click.option('--dry-run', is_flag=True, help='Show what would be processed')
@click.option('--no-embed', is_flag=True, help='Skip embedding generation')
@click.option('--sort', type=click.Choice(['date', 'date-asc', 'size', 'size-asc', 'project']), default='date', help='Sort order')
@click.option('--raw', is_flag=True, help='Disable noise filters')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--mode', type=click.Choice(['extract', 'changes', 'intent', 'stats', 'review']), default='extract', help='Extraction mode')
@click.option('--detail', is_flag=True, help='Include full tool call log (--mode stats only)')
@click.option('--full', is_flag=True, help='Disable output truncation (--mode changes only)')
@click.option('--strict', is_flag=True, help='Fail on malformed JSONL')
def batch(project, since, min_size, output, dry_run, no_embed, sort, raw, verbose, mode, detail, full, strict):
    """Batch process historical session transcripts."""
    from minutes.cli_batch import handle_batch
    handle_batch(project, since, min_size, output, dry_run, no_embed, sort, raw, verbose, mode, detail, full, strict)


@main.command()
@click.argument('query')
@click.option('--project', type=str, default=None, help='Filter by project key')
@click.option('--category', type=str, default=None, help='Filter by category')
@click.option('--limit', type=int, default=10, help='Max results')
@click.option('--mode', type=click.Choice(['keyword', 'vector', 'hybrid']), default='hybrid', help='Search mode')
def search(query, project, category, limit, mode):
    """Search across all indexed session extractions."""
    from minutes.cli_search import handle_search
    handle_search(query, project, category, limit, mode)


if __name__ == '__main__':
    main()
