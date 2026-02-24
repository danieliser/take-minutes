"""SQLite index + FAISS vector search for minutes extractions."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # numpy only needed for vector search features

from minutes.models import ExtractionResult


# Category → (content_field, detail_field, owner_field)
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

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self._init_schema()

    def _migrate(self):
        """Migrate old schema if needed. Never drops data — uses rename-copy-drop."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='item_embeddings'"
        )
        if cursor.fetchone() is not None:
            cols = [r[1] for r in self.conn.execute("PRAGMA table_info(item_embeddings)").fetchall()]
            if "model" not in cols:
                # Old schema (no model column) → migrate data to new composite PK
                self.conn.executescript("""
                    ALTER TABLE item_embeddings RENAME TO item_embeddings_old;

                    CREATE TABLE item_embeddings (
                        item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                        model TEXT NOT NULL DEFAULT 'all-mpnet-base-v2',
                        embedding BLOB NOT NULL,
                        PRIMARY KEY (item_id, model)
                    );

                    INSERT INTO item_embeddings (item_id, model, embedding)
                    SELECT item_id, 'all-MiniLM-L6-v2', embedding
                    FROM item_embeddings_old;

                    DROP TABLE item_embeddings_old;
                """)

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_key TEXT NOT NULL,
                input_file TEXT NOT NULL,
                output_file TEXT,
                file_hash TEXT,
                extracted_at TEXT,
                file_size INTEGER,
                message_count INTEGER,
                transcript_chars INTEGER,
                decisions INTEGER DEFAULT 0,
                ideas INTEGER DEFAULT 0,
                questions INTEGER DEFAULT 0,
                action_items INTEGER DEFAULT 0,
                concepts INTEGER DEFAULT 0,
                terms INTEGER DEFAULT 0,
                tldr TEXT
            );

            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                detail TEXT,
                owner TEXT,
                date TEXT,
                UNIQUE(session_id, category, content)
            );

            CREATE TABLE IF NOT EXISTS item_embeddings (
                item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                model TEXT NOT NULL DEFAULT 'all-mpnet-base-v2',
                embedding BLOB NOT NULL,
                PRIMARY KEY (item_id, model)
            );

            CREATE INDEX IF NOT EXISTS idx_items_session ON items(session_id);
            CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_key);
        """)

        # FTS5 virtual table — standalone (not content-synced) for simplicity
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items_fts'"
        )
        if cursor.fetchone() is None:
            self.conn.execute("""
                CREATE VIRTUAL TABLE items_fts USING fts5(
                    content, detail
                )
            """)

        self.conn.commit()

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
    ):
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

    def get_session(self, session_id: str) -> dict | None:
        """Get session metadata by ID."""
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def is_indexed(self, session_id: str, file_hash: str = "") -> bool:
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
    ) -> list[dict]:
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

    def search_keyword(self, query: str, category: str | None = None, limit: int = 10) -> list[dict]:
        """Full-text search via FTS5."""
        sql = """
            SELECT i.id, i.session_id, i.category, i.content, i.detail, i.owner,
                   f.rank AS score
            FROM items_fts f
            JOIN items i ON i.id = f.rowid
            WHERE items_fts MATCH ?
        """
        params: list = [query]

        if category:
            sql += " AND i.category = ?"
            params.append(category)

        sql += " ORDER BY f.rank LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_embeddings(self, model: str = "all-mpnet-base-v2") -> tuple[list[int], np.ndarray]:
        """Load all item embeddings for a given model into a numpy matrix."""
        cursor = self.conn.execute(
            "SELECT item_id, embedding FROM item_embeddings WHERE model = ? ORDER BY item_id",
            (model,),
        )
        rows = cursor.fetchall()
        if not rows:
            return [], np.array([], dtype=np.float32)

        item_ids = []
        embeddings = []
        for item_id, embedding_bytes in rows:
            item_ids.append(item_id)
            embeddings.append(np.frombuffer(embedding_bytes, dtype=np.float32))

        return item_ids, np.stack(embeddings, axis=0)

    def store_embeddings(self, item_ids: list[int], embeddings: np.ndarray, model: str = "all-mpnet-base-v2"):
        """Store embeddings as float32 BLOBs."""
        for item_id, emb in zip(item_ids, embeddings):
            self.conn.execute(
                "INSERT OR REPLACE INTO item_embeddings (item_id, model, embedding) VALUES (?, ?, ?)",
                (item_id, model, emb.astype(np.float32).tobytes()),
            )
        self.conn.commit()

    def get_unembedded_items(self, model: str = "all-mpnet-base-v2") -> list[dict]:
        """Get items that don't have embeddings for the given model."""
        rows = self.conn.execute("""
            SELECT i.id, i.content, i.detail
            FROM items i
            LEFT JOIN item_embeddings e ON e.item_id = i.id AND e.model = ?
            WHERE e.item_id IS NULL
        """, (model,)).fetchall()
        return [dict(r) for r in rows]

    def get_item(self, item_id: int) -> dict | None:
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
        query_embedding: np.ndarray,
        category: str | None = None,
        limit: int = 10,
        model: str = "all-mpnet-base-v2",
    ) -> list[dict]:
        """Vector similarity search using FAISS IndexFlatIP."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for vector search. "
                "Install search extras: pip install 'take-minutes[search]'"
            )

        item_ids, embeddings = self.get_all_embeddings(model=model)
        if len(embeddings) == 0:
            return []

        # Filter by category if needed
        if category:
            rows = self.conn.execute(
                "SELECT e.item_id FROM item_embeddings e "
                "JOIN items i ON i.id = e.item_id WHERE i.category = ?",
                (category,),
            ).fetchall()
            valid_ids = {r["item_id"] for r in rows}
            mask = [i for i, iid in enumerate(item_ids) if iid in valid_ids]
            if not mask:
                return []
            item_ids = [item_ids[i] for i in mask]
            embeddings = embeddings[mask]

        # Normalize
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        q = (query_embedding / query_norm).reshape(1, -1).astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = (embeddings / norms).astype(np.float32)

        # Build ephemeral FAISS index
        d = normed.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(np.ascontiguousarray(normed))

        k = min(limit, len(item_ids))
        scores, indices = index.search(q, k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx >= 0:
                item = self.get_item(item_ids[idx])
                if item:
                    item["score"] = float(scores[0][i])
                    results.append(item)

        return results

    def search_hybrid(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        category: str | None = None,
        limit: int = 10,
        model: str = "all-mpnet-base-v2",
    ) -> list[dict]:
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

    def stats(self) -> dict:
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

    def close(self):
        self.conn.close()


def _rrf_merge(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion — merge multiple ranked lists."""
    score_map: dict[int, float] = {}
    item_map: dict[int, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            reciprocal = 1.0 / (k + rank)

            if item_id not in score_map:
                score_map[item_id] = 0.0
                item_map[item_id] = item
            else:
                item_map[item_id].update(item)

            score_map[item_id] += reciprocal

    sorted_results = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    output = []
    for item_id, score in sorted_results:
        result = item_map[item_id].copy()
        result["rrf_score"] = score
        output.append(result)

    return output
