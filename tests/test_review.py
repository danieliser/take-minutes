"""Tests for the review gap analysis module."""

import json
import pytest
from unittest.mock import MagicMock, patch
from minutes.review import run_review, REVIEW_SYSTEM_PROMPT
from minutes.review_format import format_review_markdown
from minutes.models import ReviewResult, ReviewItem


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.model = "test-model"
    return backend


@pytest.fixture
def review_response():
    return json.dumps({
        "alignment_score": 0.75,
        "summary": "Most goals were addressed but database migration was skipped.",
        "covered": [
            {"description": "Add user auth", "evidence": "auth.py created"},
            {"description": "Add API endpoints", "evidence": "routes.py modified"},
        ],
        "gaps": [
            {"description": "Database migration", "evidence": "No migration files found"},
        ],
        "unasked": [
            {"description": "Added logging", "evidence": "logger.py created"},
        ],
    })


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a JSONL with user prompts and assistant tool uses."""
    jsonl_file = tmp_path / "session.jsonl"
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": "Add user authentication"}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "I'll create the auth module."},
            {"type": "tool_use", "id": "1", "name": "Write", "input": {"file_path": "/app/auth.py", "content": "# auth"}},
        ]}}),
        json.dumps({"type": "user", "message": {"role": "user", "content": "Also add API endpoints"}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "2", "name": "Edit", "input": {"file_path": "/app/routes.py", "old_string": "old", "new_string": "new"}},
        ]}}),
    ]
    jsonl_file.write_text("\n".join(lines) + "\n")
    return str(jsonl_file)


class TestRunReview:

    def test_returns_empty_for_empty_file(self, mock_backend, tmp_path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        result = run_review(mock_backend, str(empty_file))
        assert result.alignment_score == 0.0
        assert result.covered == []
        assert result.gaps == []
        mock_backend.client.chat.completions.create.assert_not_called()

    def test_calls_backend_with_intent_and_changes(self, mock_backend, sample_jsonl, review_response):
        mock_backend.client.chat.completions.create.return_value.choices[0].message.content = review_response
        # Mock summarize_intent to avoid a second LLM call
        with patch("minutes.review.summarize_intent") as mock_intent:
            from minutes.models import IntentSummary
            mock_intent.return_value = IntentSummary(
                primary_goal="Add auth and API",
                sub_goals=["Add user auth", "Add API endpoints"],
                prompt_count=2,
            )
            result = run_review(mock_backend, sample_jsonl)

        assert result.alignment_score == 0.75
        assert len(result.covered) == 2
        assert len(result.gaps) == 1
        assert len(result.unasked) == 1
        assert result.intent_prompt_count == 2
        assert result.changes_count == 2
        mock_backend.client.chat.completions.create.assert_called_once()

    def test_handles_backend_exception(self, mock_backend, sample_jsonl):
        mock_backend.client.chat.completions.create.side_effect = Exception("LLM down")
        with patch("minutes.review.summarize_intent") as mock_intent:
            from minutes.models import IntentSummary
            mock_intent.return_value = IntentSummary(primary_goal="Test", prompt_count=2)
            result = run_review(mock_backend, sample_jsonl)

        assert result.alignment_score == 0.0
        assert result.intent_prompt_count == 2
        assert result.changes_count == 2

    def test_handles_no_prompts_with_changes(self, mock_backend, tmp_path, review_response):
        """Session with changes but no user prompts (team session)."""
        jsonl_file = tmp_path / "team.jsonl"
        lines = [
            json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "1", "name": "Write", "input": {"file_path": "/f.py", "content": "x"}},
            ]}}),
        ]
        jsonl_file.write_text("\n".join(lines) + "\n")
        mock_backend.client.chat.completions.create.return_value.choices[0].message.content = review_response

        result = run_review(mock_backend, str(jsonl_file))
        assert result.changes_count == 1
        assert result.intent_prompt_count == 0


class TestFormatReviewMarkdown:

    def test_full_result(self):
        result = ReviewResult(
            alignment_score=0.85,
            covered=[ReviewItem(description="Auth added", evidence="auth.py created")],
            gaps=[ReviewItem(description="DB migration", evidence="No files")],
            unasked=[ReviewItem(description="Logging", evidence="logger.py")],
            summary="Good alignment overall.",
            intent_prompt_count=5,
            changes_count=12,
        )
        md = format_review_markdown(result, "session.jsonl")

        assert "# Session Review: session.jsonl" in md
        assert "**Alignment Score**: 0.85 / 1.0" in md
        assert "**Prompts analyzed**: 5 | **Code changes**: 12" in md
        assert "## Summary" in md
        assert "Good alignment overall." in md
        assert "## Covered" in md
        assert "- Auth added — *auth.py created*" in md
        assert "## Gaps (requested but not done)" in md
        assert "- DB migration — *No files*" in md
        assert "## Unasked Work (done but not requested)" in md
        assert "- Logging — *logger.py*" in md

    def test_empty_result(self):
        result = ReviewResult()
        md = format_review_markdown(result, "empty.jsonl")

        assert "No review could be generated" in md
        assert "No user prompts found" in md
        assert "No code changes found" in md

    def test_no_gaps(self):
        result = ReviewResult(
            alignment_score=1.0,
            covered=[ReviewItem(description="Everything done")],
            gaps=[],
            unasked=[],
            summary="Perfect session.",
            intent_prompt_count=3,
            changes_count=5,
        )
        md = format_review_markdown(result, "perfect.jsonl")

        assert "## Covered" in md
        assert "## Gaps" not in md
        assert "## Unasked" not in md

    def test_no_evidence(self):
        result = ReviewResult(
            alignment_score=0.5,
            covered=[ReviewItem(description="Something done", evidence="")],
            summary="Partial.",
            intent_prompt_count=1,
            changes_count=1,
        )
        md = format_review_markdown(result, "test.jsonl")
        assert "- Something done\n" in md
        assert "—" not in md.split("## Covered")[1].split("##")[0] if "## Covered" in md else True
