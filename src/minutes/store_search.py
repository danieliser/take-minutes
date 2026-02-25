"""Search operations: keyword, vector, and hybrid search."""

from __future__ import annotations

import sqlite3
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None


def search_keyword(conn: sqlite3.Connection, query: str, category: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
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

    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_all_embeddings(conn: sqlite3.Connection, model: str = "all-mpnet-base-v2") -> tuple[list[int], Any]:
    """Load all item embeddings for a given model into a numpy matrix."""
    cursor = conn.execute(
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


def search_vector(
    conn: sqlite3.Connection,
    query_embedding: Any,
    category: str | None = None,
    limit: int = 10,
    model: str = "all-mpnet-base-v2",
    get_item_func: Any = None,
) -> list[dict[str, Any]]:
    """Vector similarity search using FAISS IndexFlatIP."""
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is required for vector search. "
            "Install search extras: pip install 'take-minutes[search]'"
        )

    item_ids, embeddings = get_all_embeddings(conn, model=model)
    if len(embeddings) == 0:
        return []

    # Filter by category if needed
    if category:
        rows = conn.execute(
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
            item = get_item_func(item_ids[idx])
            if item:
                item["score"] = float(scores[0][i])
                results.append(item)

    return results


def _rrf_merge(ranked_lists: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion — merge multiple ranked lists."""
    score_map: dict[int, float] = {}
    item_map: dict[int, dict[str, Any]] = {}

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
