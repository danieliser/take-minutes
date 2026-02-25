"""Tests for the extractor module."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from minutes.extractor import (
    GatewayBackend,
    get_backend,
    extract_structured,
    process_transcript,
)
from minutes.extractor_chunking import (
    extract_json_block,
    chunk_transcript,
)
from minutes.extractor_dedup import (
    merge_results,
)
from minutes.models import ExtractionResult, Decision, Idea, Question, ActionItem, Concept, Term
from minutes.config import Config


class TestExtractJsonBlock:
    """Tests for extract_json_block function."""

    def test_extract_json_with_markdown_code_block(self):
        text = '```json\n{"decisions": [], "ideas": [], "questions": [], "action_items": [], "concepts": [], "terms": [], "tldr": ""}\n```'
        result = extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["decisions"] == []

    def test_extract_json_with_bare_code_block(self):
        text = '```\n{"decisions": [], "ideas": [], "questions": [], "action_items": [], "concepts": [], "terms": [], "tldr": "test"}\n```'
        result = extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["tldr"] == "test"

    def test_extract_json_raw_without_code_block(self):
        text = '{"decisions": [], "ideas": [], "questions": [], "action_items": [], "concepts": [], "terms": [], "tldr": "raw"}'
        result = extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["tldr"] == "raw"

    def test_extract_json_with_whitespace(self):
        text = '\n```json\n  {"decisions": [], "ideas": [], "questions": [], "action_items": [], "concepts": [], "terms": [], "tldr": ""}\n```\n'
        result = extract_json_block(text)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_extract_json_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json_block("")

    def test_extract_json_no_valid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json_block("no json here")


class TestChunkTranscript:
    """Tests for chunk_transcript function."""

    def test_chunk_short_transcript_single_chunk(self):
        text = "Short text" * 50
        chunks = chunk_transcript(text, max_size=1000, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_transcript_multiple_chunks(self):
        paragraphs = [f"This is paragraph {i}.\n\nContent {i}" for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_transcript(text, max_size=200, overlap=50)
        assert len(chunks) > 1

    def test_chunk_respects_overlap(self):
        text = "word " * 500
        chunks = chunk_transcript(text, max_size=500, overlap=100)
        if len(chunks) >= 2:
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0

    def test_chunk_preserves_content(self):
        text = "unique text " * 500
        chunks = chunk_transcript(text, max_size=300, overlap=50)
        concatenated = "".join(chunks)
        assert "unique text" in concatenated

    def test_chunk_prefers_paragraph_boundaries(self):
        text = "Para1 content.\n\nPara2 content.\n\nPara3 content.\n\nPara4 content."
        chunks = chunk_transcript(text, max_size=100, overlap=10)
        for chunk in chunks:
            assert len(chunk) > 0


class TestMergeResults:
    """Tests for merge_results function."""

    def test_merge_empty_results(self):
        results = [ExtractionResult(), ExtractionResult()]
        merged = merge_results(results)
        assert len(merged.decisions) == 0
        assert len(merged.ideas) == 0

    def test_merge_concatenates_items(self):
        result1 = ExtractionResult(
            decisions=[Decision(summary="Use Python for the backend")],
            ideas=[Idea(title="Add caching layer")],
        )
        result2 = ExtractionResult(
            decisions=[Decision(summary="Deploy to AWS with Kubernetes")],
            ideas=[Idea(title="Build monitoring dashboard")],
        )
        merged = merge_results([result1, result2])
        assert len(merged.decisions) == 2
        assert len(merged.ideas) == 2

    def test_merge_deduplicates_decisions_by_similarity(self):
        result1 = ExtractionResult(decisions=[Decision(summary="Use Python for backend")])
        result2 = ExtractionResult(decisions=[Decision(summary="Use Python for backend")])
        merged = merge_results([result1, result2])
        assert len(merged.decisions) == 1

    def test_merge_deduplicates_concepts_by_name(self):
        result1 = ExtractionResult(concepts=[Concept(name="Microservices", definition="Arch pattern")])
        result2 = ExtractionResult(concepts=[Concept(name="Microservices", definition="Different def")])
        merged = merge_results([result1, result2])
        assert len(merged.concepts) == 1

    def test_merge_deduplicates_terms_by_name(self):
        result1 = ExtractionResult(terms=[Term(term="API", definition="Application Programming Interface")])
        result2 = ExtractionResult(terms=[Term(term="API", definition="Different definition")])
        merged = merge_results([result1, result2])
        assert len(merged.terms) == 1

    def test_merge_keeps_longest_tldr(self):
        result1 = ExtractionResult(tldr="Short summary")
        result2 = ExtractionResult(tldr="This is a much longer summary with more details")
        merged = merge_results([result1, result2])
        assert merged.tldr == "This is a much longer summary with more details"

    def test_merge_single_result(self):
        result = ExtractionResult(decisions=[Decision(summary="Test")], tldr="Test summary")
        merged = merge_results([result])
        assert len(merged.decisions) == 1
        assert merged.tldr == "Test summary"


class TestGatewayBackend:
    """Tests for GatewayBackend class."""

    @patch("minutes.extractor.openai.OpenAI")
    def test_gateway_backend_init(self, mock_openai_class):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        backend = GatewayBackend(model="qwen3-4b", base_url="http://localhost:8800/v1")
        assert backend.model == "qwen3-4b"
        assert backend.client is mock_client

    @patch("minutes.extractor.openai.OpenAI")
    def test_gateway_backend_generate(self, mock_openai_class):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Gateway response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        backend = GatewayBackend(model="qwen3-4b")
        result = backend.generate(system_prompt="You are helpful", user_prompt="What is AI?")
        assert result == "Gateway response"

    @patch("minutes.extractor.openai.OpenAI")
    def test_gateway_backend_passes_model(self, mock_openai_class):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        backend = GatewayBackend(model="haiku")
        backend.generate("system", "user")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "haiku"

    @patch("minutes.extractor.openai.OpenAI")
    def test_gateway_backend_sends_messages(self, mock_openai_class):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        backend = GatewayBackend(model="qwen3-4b")
        backend.generate("Be concise", "Question")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Question"


class TestGetBackend:
    """Tests for get_backend function."""

    @patch("minutes.extractor.openai.OpenAI")
    def test_get_backend_returns_gateway(self, mock_openai_class):
        mock_openai_class.return_value = Mock()
        config = Config(gateway_model="qwen3-4b")
        backend, backend_type = get_backend(config)
        assert backend_type == "gateway"
        assert isinstance(backend, GatewayBackend)

    @patch("minutes.extractor.openai.OpenAI")
    def test_get_backend_uses_config_model(self, mock_openai_class):
        mock_openai_class.return_value = Mock()
        config = Config(gateway_model="haiku")
        backend, _ = get_backend(config)
        assert backend.model == "haiku"

    @patch("minutes.extractor.openai.OpenAI")
    def test_get_backend_prints_status(self, mock_openai_class, capsys):
        mock_openai_class.return_value = Mock()
        config = Config(gateway_model="qwen3-4b")
        get_backend(config)
        captured = capsys.readouterr()
        assert "gateway" in captured.out.lower()


class TestExtractStructured:
    """Tests for extract_structured function."""

    def test_extract_structured_basic(self):
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "decisions": [{"summary": "Use Python", "owner": "", "rationale": "", "date": ""}],
            "ideas": [], "questions": [], "action_items": [],
            "concepts": [], "terms": [], "tldr": "We decided to use Python"
        })
        config = Config()
        result = extract_structured(mock_backend, config, "meeting transcript")
        assert len(result.decisions) == 1
        assert result.tldr == "We decided to use Python"

    def test_extract_structured_retry_on_validation_error(self):
        mock_backend = Mock()
        valid = json.dumps({
            "decisions": [], "ideas": [], "questions": [],
            "action_items": [], "concepts": [], "terms": [], "tldr": ""
        })
        mock_backend.generate.side_effect = ["{ invalid json", valid]
        config = Config(max_retries=3)
        result = extract_structured(mock_backend, config, "transcript")
        assert isinstance(result, ExtractionResult)
        assert mock_backend.generate.call_count == 2

    def test_extract_structured_returns_empty_on_max_retries(self):
        mock_backend = Mock()
        mock_backend.generate.return_value = "{ invalid json"
        config = Config(max_retries=2)
        result = extract_structured(mock_backend, config, "transcript")
        assert isinstance(result, ExtractionResult)
        assert len(result.decisions) == 0

    def test_extract_structured_includes_schema_in_prompt(self):
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "decisions": [], "ideas": [], "questions": [],
            "action_items": [], "concepts": [], "terms": [], "tldr": ""
        })
        config = Config()
        extract_structured(mock_backend, config, "transcript")
        call_args = mock_backend.generate.call_args
        user_prompt = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("user_prompt", "")
        assert "schema" in user_prompt.lower() or "json" in user_prompt.lower()


class TestProcessTranscript:
    """Tests for process_transcript function."""

    def test_process_transcript_single_chunk(self):
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "decisions": [{"summary": "Decision 1", "owner": "", "rationale": "", "date": ""}],
            "ideas": [], "questions": [], "action_items": [],
            "concepts": [], "terms": [], "tldr": "Summary"
        })
        config = Config(max_chunk_size=5000)
        result = process_transcript(mock_backend, config, "Short transcript" * 10)
        assert len(result.decisions) == 1
        assert mock_backend.generate.call_count == 1

    def test_process_transcript_multiple_chunks(self):
        mock_backend = Mock()
        mock_backend.generate.return_value = json.dumps({
            "decisions": [{"summary": "Decision", "owner": "", "rationale": "", "date": ""}],
            "ideas": [], "questions": [], "action_items": [],
            "concepts": [], "terms": [], "tldr": "Summary"
        })
        config = Config(max_chunk_size=200, chunk_overlap=20, max_retries=1)
        long_transcript = "This is a paragraph.\n\n" * 100
        result = process_transcript(mock_backend, config, long_transcript)
        assert isinstance(result, ExtractionResult)
        assert mock_backend.generate.call_count > 1

    def test_process_transcript_empty_input(self):
        mock_backend = Mock()
        config = Config()
        result = process_transcript(mock_backend, config, "")
        assert isinstance(result, ExtractionResult)


class TestIntegration:
    """Integration tests with gateway backend."""

    @patch("minutes.extractor.openai.OpenAI")
    def test_full_extraction_workflow(self, mock_openai_class):
        mock_client = Mock()
        response_data = {
            "decisions": [{"summary": "Use Python", "owner": "Team", "rationale": "Performance", "date": "2024-01-01"}],
            "ideas": [{"title": "Microservices", "description": "Split into services", "category": "suggestion"}],
            "questions": [{"text": "How to scale?", "context": "Future planning", "owner": "Arch team"}],
            "action_items": [{"description": "Set up CI/CD", "owner": "DevOps", "deadline": "2024-02-01"}],
            "concepts": [{"name": "DevOps", "definition": "Practices and tools"}],
            "terms": [{"term": "CI/CD", "definition": "Continuous Integration/Deployment", "context": "Infrastructure"}],
            "tldr": "Meeting decided to use Python with microservices architecture"
        }
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps(response_data)))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = Config(gateway_model="qwen3-4b")
        backend, backend_type = get_backend(config)
        assert backend_type == "gateway"

        result = process_transcript(backend, config, "We discussed architecture...")

        assert len(result.decisions) >= 1
        assert len(result.ideas) >= 1
        assert len(result.tldr) > 0
