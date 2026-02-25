"""Deduplication: _deduplicate_by_similarity, _cross_category_dedup, _deduplicate_by_exact_attr, merge_results."""

from __future__ import annotations

from difflib import SequenceMatcher

from minutes.models import ExtractionResult


def merge_results(results: list[ExtractionResult]) -> ExtractionResult:  # noqa: D103
    """Merge multiple extraction results, deduplicating and selecting best items.

    Deduplication:
    - Decisions/Ideas/Questions/ActionItems: >80% text similarity
    - Concepts/Terms: exact name/term match
    - TLDR: keep longest

    Args:
        results: List of extraction results to merge

    Returns:
        Single merged ExtractionResult
    """
    if not results:
        return ExtractionResult()

    merged = ExtractionResult()

    # Concatenate all lists
    all_decisions = [d for r in results for d in r.decisions]
    all_ideas = [i for r in results for i in r.ideas]
    all_questions = [q for r in results for q in r.questions]
    all_action_items = [a for r in results for a in r.action_items]
    all_concepts = [c for r in results for c in r.concepts]
    all_terms = [t for r in results for t in r.terms]

    # Deduplicate decisions by >80% similarity
    merged.decisions = _deduplicate_by_similarity(all_decisions, attr="summary")

    # Deduplicate ideas by >80% similarity
    merged.ideas = _deduplicate_by_similarity(all_ideas, attr="title")

    # Deduplicate questions by >80% similarity
    merged.questions = _deduplicate_by_similarity(all_questions, attr="text")

    # Deduplicate action_items by >80% similarity
    merged.action_items = _deduplicate_by_similarity(
        all_action_items, attr="description"
    )

    # Cross-category dedup: action_items that restate a decision
    merged.action_items = _cross_category_dedup(
        merged.action_items, "description", merged.decisions, "summary"
    )

    # Cross-category dedup: ideas that restate a decision
    merged.ideas = _cross_category_dedup(
        merged.ideas, "title", merged.decisions, "summary"
    )

    # Deduplicate concepts by exact name match
    merged.concepts = _deduplicate_by_exact_attr(all_concepts, attr="name")

    # Deduplicate terms by exact term match
    merged.terms = _deduplicate_by_exact_attr(all_terms, attr="term")

    # Keep longest TLDR
    tldrs = [r.tldr for r in results if r.tldr]
    merged.tldr = max(tldrs, key=len) if tldrs else ""

    return merged


def _deduplicate_by_similarity(
    items: list[object], attr: str, threshold: float = 0.8
) -> list[object]:  # noqa: D103
    """Deduplicate items by >threshold text similarity on an attribute."""
    if not items:
        return []

    kept = []
    for item in items:
        item_text = getattr(item, attr, "")

        is_duplicate = False
        for kept_item in kept:
            kept_text = getattr(kept_item, attr, "")
            similarity = SequenceMatcher(None, item_text, kept_text).ratio()
            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(item)

    return kept


def _cross_category_dedup(
    items: list[object], items_attr: str,
    reference: list[object], reference_attr: str,
    threshold: float = 0.8,
) -> list[object]:  # noqa: D103
    """Remove items that duplicate entries in a reference category."""
    if not items or not reference:
        return items
    ref_texts = [getattr(r, reference_attr, "").lower() for r in reference]
    kept = []
    for item in items:
        item_text = getattr(item, items_attr, "").lower()
        is_dup = any(
            SequenceMatcher(None, item_text, ref).ratio() >= threshold
            for ref in ref_texts
        )
        if not is_dup:
            kept.append(item)
    return kept


def _deduplicate_by_exact_attr(items: list[object], attr: str) -> list[object]:  # noqa: D103
    """Deduplicate items by exact attribute match, keeping first occurrence."""
    if not items:
        return []

    seen = set()
    kept = []
    for item in items:
        item_val = getattr(item, attr, "")
        if item_val not in seen:
            seen.add(item_val)
            kept.append(item)

    return kept
