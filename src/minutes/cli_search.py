"""Search command — query across indexed session extractions."""

from __future__ import annotations

import sys
from pathlib import Path

import click


def handle_search(
    query: str,
    project: str | None,
    category: str | None,
    limit: int,
    mode: str,
) -> None:
    """Search across all indexed session extractions."""
    output_base = Path.home() / ".claude" / "minutes"
    if not output_base.exists():
        click.secho("No minutes index found. Run 'batch' first.", fg='yellow')
        return

    from minutes import embeddings as emb
    from minutes.embeddings import DEFAULT_MODEL
    from minutes.store import MinutesStore

    hf_id = emb.MODELS[DEFAULT_MODEL][0]
    all_results = []

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
        except Exception:
            pass

        store.close()

    if not all_results:
        click.secho("No results found.", fg='yellow')
        return

    score_key = 'rrf_score' if mode == 'hybrid' else 'score'
    all_results.sort(key=lambda r: r.get(score_key, r.get('score', 0)), reverse=True)
    all_results = all_results[:limit]

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
