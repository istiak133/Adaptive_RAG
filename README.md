# Adaptive Document Preparation System

A CLI + REST backend that turns a long PDF into MCQ-based study sessions
and learns from the user's mistakes across runs. Built for the Cloudly
AI/ML intern take-home assessment against `SLATEFALL_DOSSIER.pdf` — a
50-page, 10-section fictional dossier.

The core differentiator is the **adaptive loop**: every wrong answer
updates per-topic mastery weights in the database; on the next run over
overlapping sections, the system pulls those weights, biases the
retriever toward weak topics, and injects a WEAK / MASTERED summary
into the LLM prompt. Mastered topics are de-emphasised. Near-duplicate
questions from past sessions are hard-rejected at validation time.


## Table of contents

1. [What you get](#what-you-get)
2. [Run it in 5 minutes](#run-it-in-5-minutes)
3. [Scenario B — the main evaluation flow](#scenario-b--the-main-evaluation-flow)
4. [Scenario A and other commands](#scenario-a-and-other-commands)
5. [REST API](#rest-api)
6. [Tech stack and reasoning](#tech-stack-and-reasoning)
7. [How the adaptive loop works](#how-the-adaptive-loop-works)
8. [Knowledge-base schema](#knowledge-base-schema)
9. [Access patterns](#access-patterns)
10. [Output file formats](#output-file-formats)
11. [Configuration](#configuration)
12. [Important notes from the brief](#important-notes-from-the-brief)
13. [Limitations and assumptions](#limitations-and-assumptions)
14. [Project layout](#project-layout)
15. [Assessment criteria coverage](#assessment-criteria-coverage)


## What you get

- A **CLI** that runs the full adaptive prep cycle in one command
  (`scenario-b` for the assessment's main evaluation flow).
- A **REST API** with 16 endpoints serving the same business logic, plus
  auto-generated Swagger docs.
- A **persistent knowledge base** (PostgreSQL + ChromaDB) that records
  every prep session, every question, every answer, and the topic
  mastery state that drives adaptive behaviour.
- A **LangGraph state machine** that wires the prep flow with two real
  conditional branches: cold-start vs. adaptive routing, and a
  validate-or-retry loop on the LLM output.
- **Committed sample outputs** in `outputs/` so you can inspect the
  deliverables without running the LLM yourself.


## To Run

You need:

- **Docker Desktop** (recommended path), or Python 3.9+ for a manual install
- **Two free API keys**: `GROQ_API_KEY` and `GOOGLE_API_KEY`. Sign-up
  takes about a minute each at https://console.groq.com and
  https://aistudio.google.com/apikey.



### Option 1 — Docker (recommended)

A single compose file boots Postgres + the API in two containers. No
local Python required.

```bash
git clone https://github.com/istiak133/Adaptive_RAG
cd Adaptive_RAG

cp .env.example .env
# Open .env, paste your GROQ_API_KEY and GOOGLE_API_KEY.
# Leave DATABASE_URL untouched — it already points at the bundled Postgres.

docker compose up -d --build                          # boot services
docker compose exec app alembic upgrade head          # create 11 tables
docker compose exec app python -m src.ingestion.indexer  # parse PDF → KB (~50s)
docker compose exec app python -m src.cli stats       # sanity check
```

You should see:

```
  Sections                 10
  Chunks                   195
  Topics                   69
  ChromaDB vectors         195
  Alembic version          46105259c147
```

### Option 2 — Local Python

```bash
git clone https://github.com/istiak133/Adaptive_RAG
cd Adaptive_RAG

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Open .env, paste your two API keys, and set DATABASE_URL to point at
# any PostgreSQL instance you have access to.

alembic upgrade head
python -m src.ingestion.indexer
python -m src.cli stats
```

`DATABASE_URL` format:

```
postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
```

URL-encode any `@`, `%`, or `&` in your password (`%` → `%25`,
`&` → `%26`). Otherwise SQLAlchemy will not parse the string correctly.


## Scenario B — the main evaluation flow

This is the assessment's core deliverable. One command runs three
consecutive adaptive iterations end-to-end:

```bash
python -m src.cli scenario-b
```

What happens:

| Iteration | Sections covered | What the system does |
|-----------|------------------|---------------------|
| **1** | 5, 8 | Cold start. No history exists yet. Generates 10 MCQs (5 per section) over generic section content. |
| **2** | 6, 8, 9 | Adaptive. Reads mastery state from Iter 1, biases the retriever and prompt toward weak topics from section 8, picks fresh topics for the new sections 6 and 9. |
| **3** | 8 | Heavily adaptive. Section 8 has been studied twice already; the allocator and prompt strongly target topics the user got wrong, while mastered topics are skipped. |

Total wall-clock time: about 12–15 minutes on the Groq free tier. The
service paces calls with a 3-second floor between LLM invocations to
stay under Groq's 6,000 TPM rate limit. If Groq rate-limits anyway,
the LangChain fallback chain routes the next call to Gemini
transparently.

Six JSON files are written (exactly what the brief requires):

```
outputs/scenario_b_iter1/questions_iter1.json
outputs/scenario_b_iter1/kb_snapshot_iter1.json
outputs/scenario_b_iter2/questions_iter2.json
outputs/scenario_b_iter2/kb_snapshot_iter2.json
outputs/scenario_b_iter3/questions_iter3.json
outputs/scenario_b_iter3/kb_snapshot_iter3.json
```

A sample run from this repository (committed in `outputs/`) shows:

```
  Iter 1 (cold start)        80% correct
  Iter 2 (adaptive)          87% correct  ← carried over what was mastered
  Iter 3 (deep adaptive)     20% correct  ← targeted only weak topics
```

The Iter 3 score drop is **the adaptive system working as designed**.
It surfaces topics the simulated user keeps getting wrong, and the
weighted answer-simulator (which mimics struggling-on-weak-topics
behaviour) misses more of them. See `outputs/scenario_b_iter3/kb_snapshot_iter3.json`
for the actual paper trail.

### Different PDF? Use `--plan` to remap sections

If you swap in a substitute PDF that uses different section numbers,
override the three-iteration plan:

```bash
python -m src.cli scenario-b --plan "1,2 / 2,3 / 2"
```

The CLI validates every section ID against the KB up front. If you
reference a section that does not exist, it fails fast with the list of
section IDs that *do* exist — no cryptic empty-retrieval failures
further down the pipe.


## Scenario A and other commands

```bash
python -m src.cli --help                       # full command list

python -m src.cli scenario-a                   # cold-start over sections 1, 2
python -m src.cli scenario-a -s 5 -s 8 -q 5    # custom sections and count

python -m src.cli prep -s 2 -q 5               # one ad-hoc session, no scenario wrapping
python -m src.cli history --limit 10           # recent session history, table view
python -m src.cli snapshot 19                  # export top-5 KB snapshot for session 19
python -m src.cli stats                        # KB row counts + adaptive state summary
python -m src.cli ingest                       # re-run the PDF ingestion pipeline
```

Scenario A writes:

```
outputs/scenario_a/questions.json
outputs/scenario_a/kb_snapshot.json
```


## REST API

```bash
uvicorn src.main:app --reload --port 8000
```

Then http://localhost:8000/docs for the interactive Swagger UI. Same
business logic as the CLI — the CLI is a thin wrapper around the
service layer the REST routes also call.

The 16 endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/health` | Liveness probe |
| GET | `/api/v1/sections` | List all 10 sections |
| GET | `/api/v1/sections/{id}` | Section detail |
| GET | `/api/v1/sections/{id}/chunks` | Chunks for a section |
| GET | `/api/v1/topics` | List all 69 topics |
| GET | `/api/v1/topics/{id}` | Topic detail |
| POST | `/api/v1/prep/start` | Run one adaptive prep session |
| GET | `/api/v1/sessions` | Paginated session list |
| GET | `/api/v1/sessions/{id}` | Session detail with questions and answers |
| GET | `/api/v1/sessions/{id}/snapshot` | Top-5 KB snapshot for that session |
| GET | `/api/v1/mastery` | Global mastery state |
| GET | `/api/v1/mastery/regressions` | Topics that have regressed (mastered → wrong) |
| GET | `/api/v1/mastery/by-section/{id}` | Per-section mastery state |
| POST | `/api/v1/scenarios/b/run` | Trigger Scenario B + write all output files |
| GET | `/api/v1/admin/stats` | KB row counts |
| POST | `/api/v1/admin/reindex` | Re-run the ingestion pipeline |

Full request/response schemas are in the auto-generated Swagger.


## Tech stack and reasoning

The brief says reviewers credit deliberate, well-justified choices even
when they differ from the suggestions in the spec. Here is the full
picture in one table.

| Choice | Why this, not the alternatives |
|--------|-------------------------------|
| **Python 3.9+** | Ecosystem fit. Every component below has a first-class Python client. The Docker image runs 3.11 for cleaner async + tokenizer support. |
| **FastAPI** | REST is required. FastAPI gives auto-generated OpenAPI docs, request validation via Pydantic, and Pydantic models that double as the LLM's structured-output schema. Flask would require writing the OpenAPI spec by hand and re-validating payloads multiple times. |
| **Click** for the CLI | Argument parsing, sub-commands, type validation, and `--help` for free. argparse can do the same but with more boilerplate. The CLI is the primary surface for Scenario B, so its ergonomics matter. |
| **PostgreSQL** + SQLAlchemy 2.0 + Alembic | Real foreign keys, native JSONB columns (used for `sections_studied`, `adaptive_context`, `cross_refs`, `token_usage`), GIN indexes on JSONB for cross-session queries, and Alembic for migrations. SQLite handles most of this but its JSON ergonomics and concurrent-write story are weaker. |
| **ChromaDB** (separate from Postgres) | Purpose-built for vector similarity search. Supports metadata-based filtering directly (`where={"section_id": {"$in": [5, 8]}}`), which the retriever uses to scope a query to selected sections. pgvector would consolidate stores; keeping them split makes the role of each obvious in the code, and swapping to pgvector later touches only one repository class. |
| **LangChain 0.3** | LLM provider abstraction (`ChatGroq`, `ChatGoogleGenerativeAI`, `ChatOllama`), `ChatPromptTemplate` for prompt assembly, `JsonOutputParser(pydantic_object=MCQBatch)` for parsing the LLM's JSON output directly into a validated schema. `with_fallbacks()` chains providers with no glue code. |
| **LangGraph 0.2** | Models the prep flow as an explicit state machine with two real conditional branches (cold/adaptive routing, validate/retry). The graph compiles to a Mermaid diagram via `.get_graph()`, which has proved useful for verification. The flow could be plain Python — but the branching reads more clearly as a graph. |
| **Groq (LLaMA 3.1 8B Instant)** as primary LLM | Fastest free-tier inference for this model class — about 750 tok/sec, 14,400 requests/day, 6,000 TPM. Scenario B uses around 30 calls, well inside the budget. |
| **Gemini 2.0 Flash** as fallback | Independent rate-limit pool. If Groq 429s, LangChain's `with_fallbacks` routes the next call to Gemini and the prep flow continues. Reviewer with only one of the two keys still runs successfully. |
| **PyMuPDF** for PDF parsing | The SLATEFALL tables have no visible borders, so no table detector recognises them as tables. PyMuPDF's plain-text extraction preserves the column-aligned layout naturally — each cell lands on its own line, readable by both the embedder and the LLM. pdfplumber's text-strategy produced fragmented junk. |
| **sentence-transformers `all-MiniLM-L6-v2`** for embeddings | 384 dimensions, ~90 MB model, runs on CPU. Larger models (BGE-large, mpnet) gain marginal recall but are 5× slower on CPU, which matters when the reviewer is on a laptop. Sanity check: the query "Echo Lock failure triggers" returns §2.13 at top-1 with similarity 0.81. |
| **structure-first chunking** | The dossier author already chunked the content as sub-sections. Primary strategy: one chunk per sub-section. Fallbacks: sub-sections below 100 tokens are merged with a neighbour; sub-sections above 500 tokens are split via `RecursiveCharacterTextSplitter` with 50-token overlap; §10.1 (the Glossary) gets per-entry chunking so each of 69 terms is its own retrievable chunk. |
| **tiktoken** for token counting | Approximate — LLaMA and Gemini tokenizers differ from cl100k by maybe 5% — but cheap and consistent. The token-budget code adds a safety margin to allow for that drift. |


## How the adaptive loop works

This is the core differentiator. Three pieces working together.

### Piece 1 — every answer updates mastery

After every simulated answer, `update_mastery_after_answer` walks the
question's `question_topics` rows and updates **two** tables:

- `topic_mastery` — global per-topic state.
- `section_topic_mastery` — per-(section, topic) state.

Why both? A user can know the Echo Lock definition cold (high mastery
in §2 where it's defined) but still fail to recognise it in a Salta
incident question (low mastery in §9). The allocator reads both.

### Piece 2 — the weight formula

For each topic touched by an answer:

```
base   = 1.0 + (times_wrong × 0.5 × impact)        impact = 1.0 primary, 0.5 secondary
streak ≥ 2  → multiply by 0.3                      decay topics being answered correctly
streak ≥ 3  → set to 0.1, mark as mastered
streak < 0  → multiply by 1 + |streak| × 0.3       struggling topics get heavier weight
```

Final weight clamped to `[0.1, 5.0]`.

**Regression detection**: if a topic was `is_mastered=True` and the
next answer was wrong, the flag flips back to False and the weight
bumps to at least 2.5. The system actively re-prioritises something
the user used to know.

### Piece 3 — the next run reads everything back

On a session over sections that have any prior history,
`detect_mode_node` returns `is_cold_start=False` and the graph routes
through `load_adaptive_context_node`. That node:

1. Pulls top weak topics for these sections from `topic_mastery`.
2. Pulls currently-mastered topics.
3. Pulls the last 8 questions asked in these sections.
4. Renders them as a WEAK / MASTERED / RECENT-THEMES block.
5. Injects the block verbatim into the LLM prompt.

Alongside this, the **allocator** runs per section. It computes the
effective weight for every (section, topic) pair (global × section ×
depth-multiplier), sorts, and proportionally allocates the N questions
among the topics. So "5 questions for section 8" might become "2 about
Bofedal-3 (weight 2.5), 2 about Cell Cóndor (weight 1.8), 1 about a
fresh topic". Those topic seeds then become the semantic query the
retriever uses, so the chunks it pulls actually match the focus topics.

The exact text injected into the LLM prompt is persisted to
`sessions.adaptive_context` (JSONB), so a reviewer can open any KB
snapshot and see what the model literally consumed.


## Knowledge-base schema

11 tables across 4 conceptual layers. Full DDL is in
`alembic/versions/`. The structure:

### Layer 1 — PDF structure

| Table | Rows | Purpose |
|-------|------|---------|
| `sections` | 10 | One per main section. Title, page span, derived chunk_count. |
| `chunks` | 195 | The retrievable units. Carries sub_section_id (e.g., `"2.13"` or `"5.10+5.11"` for merged chunks), content, token count, page span, flags (has_table, has_bullets), chunk_kind enum (narrative / glossary_entry / split), and a JSONB cross_refs array of §X.Y references found in the chunk. |

### Layer 2 — the topic system

This layer is what makes adaptive logic possible.

| Table | Rows | Purpose |
|-------|------|---------|
| `topics` | 69 | One per glossary term. Stable id, slug (so "Echo Lock", "echo lock", "Echo-Lock" all resolve identically), category enum (power_mechanic, combat_tactic, adversary, …), importance flag. |
| `section_topics` | 211 | Many-to-many between sections and topics, with a `depth` enum: primary, secondary, mention. Echo Lock is `primary` in §2 (definition) but `mention` in §9 (Salta incident). Lets the system know which section "owns" a topic. |
| `chunk_topics` | 471 | Many-to-many between chunks and topics. Lets the retriever pull "all chunks that touch Echo Lock" across sections. |
| `question_topics` | 45+ | Many-to-many between questions and topics, with `is_primary` flag for impact-weighting in the mastery update. |

### Layer 3 — sessions, questions, answers

| Table | Purpose |
|-------|---------|
| `sessions` | One row per prep run. Notable column: `adaptive_context` JSONB stores the literal WEAK/MASTERED summary text sent to the LLM, plus allocator decisions, plus regression events. Complete audit trail. |
| `questions` | Every generated MCQ. Includes `source_chunk_ids` (JSONB), `source_quote` (verbatim PDF excerpt the LLM cited), difficulty, the four choices, correct answer, explanation. |
| `answers` | One row per answered question. `is_correct` computed at insert. |

### Layer 4 — mastery (the adaptive state)

| Table | Purpose |
|-------|---------|
| `topic_mastery` | Global state per topic: times_asked, times_correct, times_wrong, current_streak (positive = on a roll, negative = consecutive misses), weight (∈ [0.1, 5.0]), is_mastered, last-asked and last-wrong timestamps. |
| `section_topic_mastery` | Same shape, but per (section, topic). |


## Access patterns

Every brief requirement maps to a concrete query.

| Brief requirement | How it is satisfied |
|-------------------|---------------------|
| "Given section IDs, retrieve all prior prep sessions involving those sections" | `SELECT … FROM sessions WHERE sections_studied @> '[<id>]'::jsonb` — JSONB containment with a GIN index. |
| "Given a session, retrieve individual question-level results" | `SELECT … FROM questions JOIN answers ON answers.question_id = questions.id WHERE questions.session_id = X` |
| "Identify topics answered incorrectly across multiple sessions" | `SELECT … FROM topic_mastery ORDER BY weight DESC` (or `times_wrong DESC`). The weight already aggregates wrong-answer history; no aggregation query needed at read time. |
| "Retrieve a KB snapshot at session end" | `build_kb_snapshot()` in `src/output/snapshot.py`. Filtered by `after_session_id` so per-iteration snapshots are point-in-time correct. |

The CLI's `history`, `snapshot`, and `stats` commands hit exactly these
queries. The REST endpoints `/sessions`, `/sessions/{id}`,
`/sessions/{id}/snapshot`, `/mastery`, and `/mastery/regressions` are
thin wrappers around the same queries.


## Output file formats

### `questions_iter{N}.json` — what the LLM produced

```json
{
  "session_id": 17,
  "sections_studied": [5, 8],
  "is_cold_start": true,
  "score_pct": 80.0,
  "exported_at": "2026-05-26T14:27:50Z",
  "questions": [
    {
      "id": 161,
      "section_id": 5,
      "source_chunk_ids": ["5.4"],
      "difficulty": "medium",
      "question_text": "...",
      "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "C",
      "explanation": "...",
      "source_quote": "verbatim PDF excerpt the LLM cited",
      "user_answer": "C",
      "is_correct": true
    }
  ]
}
```

### `kb_snapshot_iter{N}.json` — the verifiable history

```json
{
  "snapshot_after_session": 19,
  "exported_at": "2026-05-26T14:40:00Z",
  "total_sessions_in_kb": 19,
  "recent_sessions": [ /* up to 5 most recent sessions, each with full
                         question detail including all four choices */ ],
  "adaptive_state": {
    "top_weak_topics": [
      {"name": "Bofedal-3", "weight": 1.95, "times_wrong": 1,
       "times_correct": 1, "streak": -1}
    ],
    "mastered_topics": []
  }
}
```

The key field for verifying adaptive behaviour is
`recent_sessions[i].adaptive_context.summary`. It contains the literal
text that was injected into the LLM prompt for that session — proof
that adaptive prompting is grounded in real history, not just a label.


## Configuration

All tunable parameters live in [config.yaml](config.yaml). Nothing
important is hardcoded in source. The sections most likely to need
adjustment:

| Section | What's in it |
|---------|--------------|
| `llm` | Provider, fallback chain, model IDs, temperature, max tokens, timeout, retries, inter-call delay (Groq TPM pacing) |
| `mcq` | Default difficulty, questions-per-section, source-quote requirement, hallucination fuzzy threshold, near-duplicate threshold |
| `adaptive` | Weight formula constants, mastery threshold (default 3), regression weight floor (default 2.5) |
| `chunking` | Min/max chunk tokens, recursive-splitter separators, glossary special-case flags |
| `retrieval` | `vector_top_k` (default 5), metadata filter behaviour |
| `history` | Adaptive-context token budget, compression behaviour across many iterations |
| `simulation` | Per-strategy correct ratios used by the weighted simulator |

Two values worth knowing about:

- `llm.inter_call_delay_seconds: 3.0` — paces calls so Scenario B
  doesn't trip Groq's 6,000 TPM ceiling. Raise it if you still hit 429s.
- `retrieval.vector_top_k: 5` — reduced from 10 because the larger
  value regularly pushed the prompt above the same ceiling.


## Important notes from the brief

The brief calls out five points the system must handle. Each is
implemented:

1. **No paid APIs.** Groq and Gemini both genuine free tier. The
   provider chain skips missing-key fallbacks with a warning rather
   than crashing, so a reviewer with only one key still gets a working
   system.
2. **Simulated user answers are acceptable.** Three simulator
   strategies in [src/prep/simulator.py](src/prep/simulator.py):
   `all_correct`, `random`, and `weighted` (default for `scenario-b`).
   The weighted strategy reads mastery weights to bias accuracy
   realistically — high accuracy on mastered topics, lower on weak ones.
3. **PDF section numbering may differ in the substitute PDF.** The
   parser regex accepts any integer numbering. The
   `scenario-b --plan` flag lets a reviewer remap iterations to
   whatever section IDs exist in their PDF.
4. **MCQ generation is non-deterministic.** This is expected — the
   LLM runs at temperature 0.7. What the system enforces is
   *structural correctness*, not output stability:
   - Pydantic schema check (4 choices, valid `A`/`B`/`C`/`D`, non-empty
     explanation)
   - Choices uniqueness check (no duplicate options)
   - Source-quote hallucination check (the verbatim PDF excerpt must
     fuzzily match the retrieved context)
   - Near-duplicate guard (rejects when `question_text` similarity
     exceeds 0.65 against any prior question for these sections)
   Failed generations retry up to 3× per topic seed before the slot
   is left unfilled.
5. **Development focus: CLI implementation supporting Scenario B is
   the core differentiator.** It is. `python -m src.cli scenario-b`
   runs the full three-iteration adaptive flow as a single command,
   writes the six required JSON files, and prints the score arc.
   The REST API and optional enhancements (Docker, structured logging,
   verification scripts) were added only after the core CLI flow was
   solid.


## Limitations and assumptions

- **PDF layout assumption.** Section and sub-section detection is
  regex-driven (`^Section N.` / `^N.M Title`). Works for SLATEFALL,
  would need adjustment for a PDF with a different convention.
- **Table extraction is best-effort.** SLATEFALL tables have no
  borders; they come through as space-separated rows. The data is
  preserved and the LLM can read it, but it is not formal markdown.
- **LLM non-determinism is real.** Same prompt + same temperature
  still varies across runs. The four validators above catch the worst
  cases. Failed MCQs retry up to 3×.
- **Rate limits are real.** Groq's free tier is 6,000 TPM. The
  service paces calls with a 3-second floor between them; if Groq
  still 429s, the LangChain fallback chain switches to Gemini
  transparently.
- **Glossary special case.** §10.1 is chunked per-entry rather than
  per-sub-section. Each of the 69 terms is its own chunk.
- **Postgres password URL-encoding.** If your password contains `@`,
  `%`, or `&`, URL-encode them in `DATABASE_URL`.
- **Re-running the indexer is idempotent.** Sections and topics are
  upserted, so existing sessions and mastery state survive a re-index.
  Only chunks and junction tables get rebuilt.
- **Python 3.9 deprecation noise.** A couple of Google packages emit
  `FutureWarning` on Python 3.9. Cosmetic only — the system runs
  fine. The Docker image uses Python 3.11, which avoids them.


## Project layout

```
src/
├── ingestion/      PDF parser, chunker, topic tagger, embedder, indexer
├── kb/             SQLAlchemy models, ChromaDB repo, mastery math
├── llm/            LangChain provider wrappers + fallback chain
├── rag/            Retriever, prompt builder, validator, token budget,
│                   history compressor (adaptive context renderer)
├── prep/           Allocator, simulator, scorer, difficulty controller
├── graph/          LangGraph state machine (state, nodes, graph assembly)
├── services/       Session lifecycle + run_prep_session orchestrator
├── api/            FastAPI routes (sections, prep, sessions, mastery,
│                   scenarios, admin)
├── output/         JSON snapshot exporter
├── main.py         FastAPI app entry point
└── cli.py          Click CLI entry point

alembic/            Schema migrations (one initial → all 11 tables)
config.yaml         Tunable parameters
outputs/            Scenario A and Scenario B JSON outputs (committed)
scripts/            verify_full_system.py, verify_pipeline.py, …
data/               The source PDF and the assessment brief
Dockerfile          Multi-stage Python 3.11 image, ~250 MB runtime
docker-compose.yml  Postgres 16 + app, healthcheck-gated
```


## Assessment criteria coverage

Mapping each brief requirement to the implementation that satisfies it.

| Requirement | Where it lives |
|-------------|----------------|
| PDF ingestion + sub-section chunking | [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py), [src/ingestion/chunker.py](src/ingestion/chunker.py) |
| Vector-based semantic retrieval | [src/rag/retriever.py](src/rag/retriever.py), [src/kb/chroma_repo.py](src/kb/chroma_repo.py) |
| MCQ generation grounded in PDF context | [src/rag/prompt_builder.py](src/rag/prompt_builder.py), [src/services/prep_service.py](src/services/prep_service.py) |
| Output schema validation | [src/rag/validator.py](src/rag/validator.py) (Pydantic + hallucination + duplicate check) |
| Free-tier LLM + graceful fallback | [src/llm/providers.py](src/llm/providers.py) |
| Sessions + questions + answers persisted | `sessions`, `questions`, `answers` tables |
| Adaptive question generation across runs | [src/kb/mastery.py](src/kb/mastery.py), [src/rag/history_compressor.py](src/rag/history_compressor.py), [src/prep/allocator.py](src/prep/allocator.py) |
| Mastery tracking + regression detection | [src/kb/mastery.py](src/kb/mastery.py) |
| Scenario A end-to-end | `scenario-a` command in [src/cli.py](src/cli.py) |
| Scenario B with three iterations | `scenario-b` command in [src/cli.py](src/cli.py) |
| Top-5 KB snapshot at session end | [src/output/snapshot.py](src/output/snapshot.py) |
| REST API | [src/api/routes/](src/api/routes/), 16 endpoints |
| CLI (primary surface) | [src/cli.py](src/cli.py), 7 commands |
| LangChain + LangGraph orchestration | [src/graph/](src/graph/), [src/llm/](src/llm/), [src/rag/](src/rag/) |
| Configuration via single source of truth | [config.yaml](config.yaml) |
| Simulated user answers (3 strategies) | [src/prep/simulator.py](src/prep/simulator.py) |
