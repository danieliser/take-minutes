"""SQLite index + FAISS vector search for minutes extractions."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

from minutes.models import ExtractionResult
from minutes.store_schema import init_pragmas, init_schema, migrate
from minutes.store_search import _rrf_merge, get_all_embeddings, search_keyword, search_vector


_CATEGORY_FIELDS = {
    "decision": ("summary", "rationale", "owner"),
    "idea": ("title", "description", None),
    "question": ("text", "context", "owner"),
    "action_item": ("description", None, "owner"),
    "concept": ("name", "definition", None),
    "term": ("term", "definition", None),
}


class MinutesStore:
    """SQLite-backed index with FTS5 keyword search and FAISS vector search."""

    def __init__(self, db_path: str | Path) -> None:  # noqa: D102
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        init_pragmas(self.conn)
        migrate(self.conn)
        init_schema(self.conn)

    def upsert_session(
        self,
        session_id: str,
        project_key: str,
        input_file: str,
        result: ExtractionResult,
        output_file: str = "",
        file_hash: str = "",
        file_size: int = 0,
        message_count: int = 0,
        transcript_chars: int = 0,
    ) -> None:  # noqa: D102
        """Insert or replace session metadata and all extracted items."""
        # Get old item IDs for FTS cleanup
        old_ids = [row[0] for row in self.conn.execute(
            "SELECT id FROM items WHERE session_id = ?", (session_id,)
        ).fetchall()]

        # Delete old FTS entries for this session's items
        for old_id in old_ids:
            self.conn.execute("DELETE FROM items_fts WHERE rowid = ?", (old_id,))

        # Delete old items (cascade handles embeddings)
        self.conn.execute("DELETE FROM items WHERE session_id = ?", (session_id,))

        # Upsert session
        self.conn.execute("""
            INSERT OR REPLACE INTO sessions
            (id, project_key, input_file, output_file, file_hash, extracted_at,
             file_size, message_count, transcript_chars,
             decisions, ideas, questions, action_items, concepts, terms, tldr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, project_key, input_file, output_file, file_hash,
            datetime.now().isoformat(),
            file_size, message_count, transcript_chars,
            len(result.decisions), len(result.ideas), len(result.questions),
            len(result.action_items), len(result.concepts), len(result.terms),
            result.tldr,
        ))

        # Insert items and their FTS entries
        items_to_insert = []
        for decision in result.decisions:
            items_to_insert.append(("decision", decision.summary, decision.rationale, decision.owner, decision.date))
        for idea in result.ideas:
            items_to_insert.append(("idea", idea.title, idea.description, None, None))
        for question in result.questions:
            items_to_insert.append(("question", question.text, question.context, question.owner, None))
        for action in result.action_items:
            items_to_insert.append(("action_item", action.description, None, action.owner, action.deadline))
        for concept in result.concepts:
            items_to_insert.append(("concept", concept.name, concept.definition, None, None))
        for term in result.terms:
            items_to_insert.append(("term", term.term, term.definition, None, None))

        for category, content, detail, owner, date in items_to_insert:
            cursor = self.conn.execute("""
                INSERT OR IGNORE INTO items (session_id, category, content, detail, owner, date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, category, content, detail, owner, date))
            if cursor.lastrowid:
                self.conn.execute(
                    "INSERT INTO items_fts(rowid, content, detail) VALUES (?, ?, ?)",
                    (cursor.lastrowid, content, detail or ""),
                )

        self.conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:  # noqa: D102
        """Get session metadata by ID."""
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def is_indexed(self, session_id: str, file_hash: str = "") -> bool:  # noqa: D102
        """Check if a session is already indexed (optionally with matching hash)."""
        row = self.conn.execute(
            "SELECT file_hash FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return False
        if file_hash and row["file_hash"] != file_hash:
            return False
        return True

    def list_sessions(
        self,
        project_key: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:  # noqa: D102
        """List sessions, optionally filtered by project and date."""
        query = "SELECT * FROM sessions WHERE 1=1"
        params: list = []

        if project_key:
            query += " AND project_key = ?"
            params.append(project_key)
        if since:
            query += " AND extracted_at >= ?"
            params.append(since)

        query += " ORDER BY extracted_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def search_keyword(self, query: str, category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:  # noqa: D102
        """Full-text search via FTS5."""
        return search_keyword(self.conn, query, category=category, limit=limit)

    def get_all_embeddings(self, model: str = "all-mpnet-base-v2") -> tuple[list[int], Any]:  # noqa: D102
        """Load all item embeddings for a given model into a numpy matrix."""
        return get_all_embeddings(self.conn, model=model)

    def store_embeddings(self, item_ids: list[int], embeddings: Any, model: str = "all-mpnet-base-v2") -> None:  # noqa: D102
        """Store embeddings as float32 BLOBs."""
        for item_id, emb in zip(item_ids, embeddings):
            self.conn.execute(
                "INSERT OR REPLACE INTO item_embeddings (item_id, model, embedding) VALUES (?, ?, ?)",
                (item_id, model, emb.astype(np.float32).tobytes()),
            )
        self.conn.commit()

    def get_unembedded_items(self, model: str = "all-mpnet-base-v2") -> list[dict[str, Any]]:  # noqa: D102
        """Get items that don't have embeddings for the given model."""
        rows = self.conn.execute("""
            SELECT i.id, i.content, i.detail
            FROM items i
            LEFT JOIN item_embeddings e ON e.item_id = i.id AND e.model = ?
            WHERE e.item_id IS NULL
        """, (model,)).fetchall()
        return [dict(r) for r in rows]

    def get_item(self, item_id: int) -> dict[str, Any] | None:  # noqa: D102
        """Get a single item by ID with session info."""
        row = self.conn.execute("""
            SELECT i.*, s.project_key, s.input_file AS session_file, s.extracted_at
            FROM items i
            JOIN sessions s ON s.id = i.session_id
            WHERE i.id = ?
        """, (item_id,)).fetchone()
        return dict(row) if row else None

    def search_vector(
        self,
        query_embedding: Any,
        category: str | None = None,
        limit: int = 10,
        model: str = "all-mpnet-base-v2",
    ) -> list[dict[str, Any]]:  # noqa: D102
        """Vector similarity search using FAISS IndexFlatIP."""
        return search_vector(
            self.conn,
            query_embedding,
            category=category,
            limit=limit,
            model=model,
            get_item_func=self.get_item,
        )

    def search_hybrid(
        self,
        query: str,
        query_embedding: Any | None = None,
        category: str | None = None,
        limit: int = 10,
        model: str = "all-mpnet-base-v2",
    ) -> list[dict[str, Any]]:  # noqa: D102
        """Hybrid keyword + vector search merged via RRF."""
        ranked_lists = []

        # Keyword search
        try:
            kw_results = self.search_keyword(query, category=category, limit=limit)
            if kw_results:
                ranked_lists.append(kw_results)
        except Exception:
            pass

        # Vector search
        if query_embedding is not None:
            try:
                vec_results = self.search_vector(query_embedding, category=category, limit=limit, model=model)
                if vec_results:
                    ranked_lists.append(vec_results)
            except Exception:
                pass

        if not ranked_lists:
            return []

        merged = _rrf_merge(ranked_lists)

        # Enrich any items missing session info
        results = []
        for item in merged[:limit]:
            if "session_file" not in item:
                full = self.get_item(item["id"])
                if full:
                    full["rrf_score"] = item.get("rrf_score", item.get("score", 0))
                    results.append(full)
                else:
                    results.append(item)
            else:
                results.append(item)

        return results

    def stats(self) -> dict[str, Any]:  # noqa: D102
        """Get aggregate stats across all sessions."""
        row = self.conn.execute("""
            SELECT COUNT(*) as session_count,
                   SUM(decisions) as total_decisions,
                   SUM(ideas) as total_ideas,
                   SUM(questions) as total_questions,
                   SUM(action_items) as total_action_items,
                   SUM(concepts) as total_concepts,
                   SUM(terms) as total_terms
            FROM sessions
        """).fetchone()
        return dict(row) if row else {}

    # --- Chunk progress for resume support ---

    def get_chunk_progress(self, session_id: str, file_hash: str) -> list[dict[str, Any]]:
        """Get completed chunk indices + results for a partial extraction."""
        rows = self.conn.execute(
            "SELECT chunk_index, chunk_size, total_chunks, result_json FROM chunk_progress "
            "WHERE session_id = ? AND file_hash = ? ORDER BY chunk_index",
            (session_id, file_hash),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_chunk_result(
        self, session_id: str, file_hash: str,
        chunk_index: int, chunk_size: int, total_chunks: int,
        result: ExtractionResult,
    ) -> None:
        """Save a single chunk's extraction result for resume."""
        import json
        self.conn.execute(
            "INSERT OR REPLACE INTO chunk_progress "
            "(session_id, file_hash, chunk_index, chunk_size, total_chunks, result_json, extracted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, file_hash, chunk_index, chunk_size, total_chunks,
             json.dumps(result.model_dump()), datetime.now().isoformat()),
        )
        self.conn.commit()

    def clear_chunk_progress(self, session_id: str, file_hash: str) -> None:
        """Remove chunk progress after successful full extraction."""
        self.conn.execute(
            "DELETE FROM chunk_progress WHERE session_id = ? AND file_hash = ?",
            (session_id, file_hash),
        )
        self.conn.commit()

    def has_partial_progress(self, session_id: str, file_hash: str) -> bool:
        """Check if there's any partial chunk progress for this session."""
        row = self.conn.execute(
            "SELECT 1 FROM chunk_progress WHERE session_id = ? AND file_hash = ? LIMIT 1",
            (session_id, file_hash),
        ).fetchone()
        return row is not None

    def close(self) -> None:  # noqa: D102
        self.conn.close()
