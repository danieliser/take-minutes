# minutes

> Distill any conversation into structured knowledge.

Local-first CLI for extracting decisions, ideas, questions, action items, concepts, and key terms from any conversation transcript. Works with Claude Code sessions, meeting transcripts, plain text, and markdown. Runs entirely offline with a local LLM—nothing leaves your machine.

## Quick Start

```bash
pip install take-minutes
minutes process my-session.jsonl
```

See output in `./output/`:
- Markdown file with structured knowledge
- SQLite index for future searches
- Optionally: semantic embeddings for cross-session discovery

## What It Extracts

| Category | Description | Example |
|----------|-------------|---------|
| **Decisions** | What was decided and why | "Use SQLite instead of PostgreSQL for MVP (reason: no schema migrations needed)" |
| **Ideas** | Concepts, suggestions, opportunities | "Implement quiet hours to prevent 2am notifications" |
| **Questions** | Open issues needing resolution | "What's the deployment target—Raspberry Pi or cloud?" |
| **Action Items** | Tasks assigned with owners | "Write health endpoint monitor (owner: ops-pragmatist, deadline: Phase 1)" |
| **Concepts** | Key technical or business ideas | "3-tiered autonomy model: tier 1 (audit), tier 2 (approval), tier 3 (confirmation)" |
| **Terms** | Abbreviations, jargon, domain terms | "EDA = Event-Driven Architecture" |

## Prerequisites

`minutes` uses a local LLM for extraction. You need an OpenAI-compatible inference endpoint:

- **Ollama**: [ollama.com](https://ollama.com) — simple local LLM runner
- **LM Studio**: [lmstudio.ai](https://lmstudio.ai) — GUI-based local inference
- **vLLM**: [docs.vllm.ai](https://docs.vllm.ai) — high-throughput serving engine
- **OpenAI API** or any OpenAI-compatible provider (set `GATEWAY_URL` env var)

For best results: use a 4B–7B model (e.g., Qwen 2.5 7B, Llama 3 8B).

## Installation

### Core (extraction only)
```bash
pip install take-minutes
```

### With semantic search
```bash
pip install "take-minutes[search]"
```

### One-line setup (downloads embedding model)
```bash
minutes setup
```

The setup command pre-downloads the embedding model (~420MB) so you aren't surprised by a mid-run download.

## Usage

### Process a single file

```bash
# Extract from Claude Code session
minutes process session.jsonl

# Extract from meeting transcript
minutes process meeting.txt -o ./my-minutes

# Skip deduplication check (force reprocess)
minutes process session.jsonl --no-dedup

# Verbose output for debugging
minutes process session.jsonl -v
```

Output:
- `output/YYYY-MM-DD-HH-MM-SS.md` — structured knowledge in markdown
- `output/minutes.db` — SQLite index with full-text search
- `output/sessions.json` — metadata log for easy inspection

### Batch process Claude Code sessions

Scan `~/.claude/projects/` and extract from all main-thread sessions:

```bash
# Process sessions from last 2 weeks, sorted by date (newest first)
minutes batch

# Filter by project key (substring match)
minutes batch --project persistence

# Change time range (ISO date or relative: 7d, 2w, 1m)
minutes batch --since 2w --sort size

# Dry run: show what would be processed
minutes batch --dry-run --min-size 100KB

# Skip embedding generation
minutes batch --no-embed
```

Output:
- `~/.claude/minutes/{project_key}/` — project-specific minutes
- `~/.claude/minutes/{project_key}/minutes.db` — indexed extractions

### Search across all processed sessions

```bash
# Keyword search
minutes search "budget decision"

# Filter by category (decision, idea, question, action_item, concept, term)
minutes search "authentication" --category decision

# Vector (semantic) search
minutes search "how do we handle failures?" --mode vector

# Hybrid (keyword + vector) search
minutes search "persistence strategy" --mode hybrid --limit 5

# Search specific project
minutes search "deployment" --project persistence
```

Returns ranked results with scores, context, and source session.

### View configuration

```bash
minutes config
minutes config --env
```

Shows active settings: gateway URL, model, output directory, chunking parameters.

## Supported Formats

- **Claude Code JSONL** (native) — `~/.claude/projects/*/session.jsonl`
- **Plain text / Markdown** — conversation transcripts, meeting notes
- **Coming soon**: ChatGPT export, Codex CLI, Cline, Cursor

Auto-detection: `process` command detects format by file extension or content. Override with `--format`:

```bash
minutes process transcript.srt --format text
```

## Configuration

Create a `.env` file in your working directory:

```bash
# LLM backend
GATEWAY_URL=http://localhost:8800/v1
GATEWAY_MODEL=qwen3-4b

# Output
OUTPUT_DIR=./minutes-output/

# Chunking (for long transcripts)
MAX_CHUNK_SIZE=12000      # tokens per chunk
CHUNK_OVERLAP=200         # token overlap between chunks

# Retry logic
MAX_RETRIES=3

# Glossary (optional YAML file for cross-referencing)
GLOSSARY_PATH=./glossary.yaml

# Prompts (optional file paths for custom extraction prompts)
SYSTEM_PROMPT=./prompts/system.txt
EXTRACTION_PROMPT=./prompts/extraction.txt

# Debug
VERBOSE=true
```

View active config:

```bash
minutes config
```

## Architecture

```
Input (JSONL / TXT / MD)
         ↓
    Parser (extract messages/text)
         ↓
   Chunker (split into LLM-friendly chunks)
         ↓
  LLM Extraction (local model extracts structured data)
         ↓
Deduplication (fuzzy match across chunks)
         ↓
  SQLite Index (FTS5 full-text search)
         ↓
Embeddings (sentence-transformers, optional)
         ↓
Semantic Search (FAISS + embeddings)
```

## How It Works

1. **Parse**: Read input file (JSONL, plaintext, markdown) and extract dialogue or transcript text
2. **Chunk**: Split text into overlapping chunks to fit LLM context window (~12K tokens by default)
3. **Extract**: Send each chunk to local LLM with structured schema; collect decisions, ideas, questions, action items, concepts, terms
4. **Deduplicate**: Fuzzy-match extracted items across chunks; keep unique items
5. **Index**: Store in SQLite with FTS5 indexing for keyword search; optionally generate embeddings for semantic/hybrid search

Output is a single markdown file with all extractions, plus a queryable SQLite database.

## Example Workflow

### Scenario: Strategic planning sessions

1. **Extract from session**:
   ```bash
   minutes process 2026-02-22-strategy.jsonl -o ./strategy-minutes
   ```

2. **Review markdown**:
   ```bash
   cat ./strategy-minutes/2026-02-22-07-32-38.md
   ```

3. **Cross-reference with glossary** (if provided):
   ```bash
   # Your GLOSSARY_PATH contains definitions for SRE, NIST AI RMF, EU AI Act
   # Output markdown shows which extracted concepts are in glossary vs unknown
   ```

4. **Batch process all recent sessions**:
   ```bash
   minutes batch --since 1m --project strategy --sort date
   ```

5. **Search for related decisions across all sessions**:
   ```bash
   minutes search "idempotency" --category decision --mode hybrid
   ```

## Requirements

- Python 3.10+
- Local LLM running on `GATEWAY_URL` (default: `http://localhost:8800/v1`)
- Optional: Sentence Transformers for semantic search (`pip install "take-minutes[search]"`)

## Tips

- **Cold start**: First extraction takes longer due to model loading. Subsequent runs are faster.
- **Large transcripts**: Automatically chunks long inputs; adjust `MAX_CHUNK_SIZE` if needed.
- **Private data**: All processing is local; nothing sent to external APIs (unless you configure `GATEWAY_URL` to an external provider).
- **Incremental indexing**: Reprocessing same file is skipped unless you use `--no-dedup`.
- **Searching**: Use `--mode hybrid` for best results (combines keyword + semantic search).

## License

MIT
