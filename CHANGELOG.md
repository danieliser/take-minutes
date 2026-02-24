# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-24

### Added
- `FilterConfig` dataclass for individually toggling each JSONL noise filter
- `--raw` flag on `process` and `batch` commands to disable all filters
- `NO_FILTERS` constant for programmatic use (keeps tool results, system reminders, etc.)
- 9 new tests for filter configurability

### Changed
- `parse_jsonl()` and `parse_file()` now accept `filter_config` parameter
- Passing `filter_config=None` disables all filtering (equivalent to `NO_FILTERS`)

## [0.2.0] - 2026-02-24

### Added
- Comprehensive PEP 604 type hints across all 11 modules
- SQLite performance pragmas: `synchronous=NORMAL`, `cache_size=10000`, `temp_store=MEMORY`
- PyPI badges, uv-first install instructions in README

### Changed
- Migrated build backend from setuptools to hatchling
- Single-source versioning via `[tool.hatch.version]`
- PEP 639 license metadata (`license = {text = "MIT"}`)

### Fixed
- Dev dependencies moved to `[dependency-groups]` for `uv sync --dev` compatibility
- License classifier conflict with PEP 639

## [0.1.1] - 2026-02-24

### Fixed
- License classifier conflict with PEP 639 metadata
- Missing readme, classifiers, and project URLs in pyproject.toml
- Replaced hardcoded model-gateway reference with generic LLM provider docs

## [0.1.0] - 2026-02-24

### Added
- `process` command — extract structured knowledge from a single transcript
- `batch` command — process multiple sessions from `~/.claude/projects/`
- `search` command — keyword, vector, and hybrid search across extractions
- `watch` command — monitor a directory for new transcripts
- `config` and `setup` commands
- JSONL parser with noise filtering (tool results, system reminders, protocol messages, compaction summaries)
- Plain text and markdown file support
- LLM extraction via configurable gateway (Ollama, LM Studio, vLLM, etc.)
- SQLite index with FTS5 keyword search
- FAISS vector search with sentence-transformers embeddings
- Hybrid search with Reciprocal Rank Fusion (RRF)
- Deduplication via file hash tracking
- Configurable glossary matching
- Markdown output with session logs and index
- `on-session-end.sh` hook for auto-extraction on Claude Code session end
