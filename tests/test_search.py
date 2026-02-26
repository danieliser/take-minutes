"""Tests for search operations: keyword, vector, and hybrid (RRF merge)."""

from __future__ import annotations

import sqlite3

import pytest

from minutes.store_search import _rrf_merge, search_keyword


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def search_db(tmp_path):
    """In-memory DB with FTS5 and sample items."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            category TEXT,
            content TEXT,
            detail TEXT,
            owner TEXT
        )
    """)
    db.execute("""
        CREATE VIRTUAL TABLE items_fts USING fts5(
            content, detail, content=items, content_rowid=id
        )
    """)
    items = [
        (1, "s1", "decision", "Replace mpnet with mxbai model", "Better MTEB scores", None),
        (2, "s1", "term", "mxbai-embed-large-v1", "SOTA retrieval model", None),
        (3, "s1", "concept", "Vector Search", "Querying for similar embeddings", None),
        (4, "s2", "action_item", "Add pgvector schema", "For embedding storage", None),
        (5, "s2", "idea", "Task-based model selection", "Rank models by task", None),
    ]
    db.executemany(
        "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?)", items
    )
    for item_id, _, _, content, detail, _ in items:
        db.execute(
            "INSERT INTO items_fts(rowid, content, detail) VALUES (?, ?, ?)",
            (item_id, content, detail or ""),
        )
    db.commit()
    return db


# ── Keyword search ────────────────────────────────────────────────────────


def test_keyword_finds_exact_term(search_db):
    results = search_keyword(search_db, "mxbai")
    assert len(results) >= 1
    ids = [r["id"] for r in results]
    assert 2 in ids  # "mxbai-embed-large-v1"


def test_keyword_respects_category_filter(search_db):
    results = search_keyword(search_db, "model", category="decision")
    assert all(r["category"] == "decision" for r in results)


def test_keyword_respects_limit(search_db):
    results = search_keyword(search_db, "model", limit=2)
    assert len(results) <= 2


def test_keyword_no_match(search_db):
    results = search_keyword(search_db, "nonexistent_xyz_query")
    assert results == []


# ── RRF merge ──────────────────────────────────────────────────────────────


def _make_items(ids_scores):
    """Helper: list of dicts with id and score."""
    return [{"id": i, "score": s, "content": f"item {i}"} for i, s in ids_scores]


class TestRRFMerge:
    def test_single_list_preserves_native_scores(self):
        """When only one source has results, native scores pass through."""
        items = _make_items([(1, 0.95), (2, 0.80), (3, 0.65)])
        merged = _rrf_merge([items])

        assert len(merged) == 3
        assert merged[0]["rrf_score"] == 0.95
        assert merged[1]["rrf_score"] == 0.80
        assert merged[2]["rrf_score"] == 0.65

    def test_single_list_preserves_order(self):
        """Order is preserved for single-source results."""
        items = _make_items([(10, 0.9), (20, 0.5), (30, 0.1)])
        merged = _rrf_merge([items])
        assert [m["id"] for m in merged] == [10, 20, 30]

    def test_two_lists_uses_reciprocal_ranking(self):
        """With two sources, RRF produces positional reciprocal scores."""
        list_a = _make_items([(1, 0.9), (2, 0.5)])
        list_b = _make_items([(2, 0.8), (3, 0.6)])

        merged = _rrf_merge([list_a, list_b])

        # Item 2 appears in both lists → highest RRF score
        ids = [m["id"] for m in merged]
        assert ids[0] == 2  # boosted by appearing in both

        # RRF score for item 2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        expected_2 = (1 / 62) + (1 / 61)
        assert abs(merged[0]["rrf_score"] - expected_2) < 1e-6

    def test_two_lists_items_unique_to_one_list(self):
        """Items appearing in only one list get single reciprocal score."""
        list_a = _make_items([(1, 0.9)])
        list_b = _make_items([(2, 0.8)])

        merged = _rrf_merge([list_a, list_b])
        assert len(merged) == 2

        # Both at rank 1 in their respective lists → same score
        assert abs(merged[0]["rrf_score"] - merged[1]["rrf_score"]) < 1e-6
        assert abs(merged[0]["rrf_score"] - 1 / 61) < 1e-6

    def test_empty_lists(self):
        """Empty input returns empty output."""
        assert _rrf_merge([]) == []

    def test_single_empty_list(self):
        """Single empty list returns empty output."""
        assert _rrf_merge([[]]) == []

    def test_k_parameter_affects_scores(self):
        """Custom k parameter changes the score magnitudes."""
        items = _make_items([(1, 0.9), (2, 0.5)])
        merged_default = _rrf_merge([items, items], k=60)
        merged_small_k = _rrf_merge([items, items], k=10)

        # Smaller k = larger scores (1/(10+rank) > 1/(60+rank))
        assert merged_small_k[0]["rrf_score"] > merged_default[0]["rrf_score"]
