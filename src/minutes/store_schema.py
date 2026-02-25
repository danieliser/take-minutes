"""SQLite schema initialization, migrations, and pragma configuration."""

from __future__ import annotations

import sqlite3


def init_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")


def migrate(conn: sqlite3.Connection) -> None:
    """Migrate old schema if needed. Never drops data — uses rename-copy-drop."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='item_embeddings'"
    )
    if cursor.fetchone() is not None:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(item_embeddings)").fetchall()]
        if "model" not in cols:
            # Old schema (no model column) → migrate data to new composite PK
            conn.executescript("""
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


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
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

        CREATE TABLE IF NOT EXISTS chunk_progress (
            session_id TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL,
            total_chunks INTEGER NOT NULL,
            result_json TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            PRIMARY KEY (session_id, file_hash, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_items_session ON items(session_id);
        CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
        CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_key);
    """)

    # FTS5 virtual table — standalone (not content-synced) for simplicity
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='items_fts'"
    )
    if cursor.fetchone() is None:
        conn.execute("""
            CREATE VIRTUAL TABLE items_fts USING fts5(
                content, detail
            )
        """)

    conn.commit()
