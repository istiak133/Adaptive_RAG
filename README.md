# Adaptive Document Preparation System

A CLI + REST backend that ingests a long PDF, asks an LLM to generate
multiple-choice questions from sections the user picks, scores the user's
answers, and — on subsequent runs over the same sections — leans on the
user's mistake history to push question selection toward the topics they
keep getting wrong.

Built for the Cloudly AI/ML intern take-home. The corpus is
`SLATEFALL_DOSSIER.pdf` — a 50-page fictional dossier with ten sections.

That last bit, the adaptive loop, is the part the assessment cares about
most, and it's what most of the moving parts in this repo exist to enable.


## What it does, in plain terms

A user picks one or more sections from the dossier. The system pulls the
relevant chunks from a vector store, asks an LLM to write MCQs grounded
in those chunks, scores the user's answers (simulated, for evaluation
purposes), and records the result.

Run it again on the same section and the system notices which topics you
missed before. It biases the retriever toward those weak topics, tells the
LLM in the prompt "the user struggles with Echo Lock — lean into it," and
avoids re-asking things you've already nailed. After three consecutive
correct answers on a topic, the system marks it mastered; if you later miss
a question on a mastered topic, that's a regression and the weight gets
bumped back up.


## Quick start (under 10 minutes)

Prerequisites:
- Python 3.9+ (or just Docker, if you'd rather not deal with Python)
- A PostgreSQL instance — any will do. I used Supabase's free tier during
  development; the Docker path below spins up a local Postgres for you.
- Free API keys (see the **API keys** note below for the details):
  - **Groq** — primary LLM. `https://console.groq.com` (sign up, create
    API key — under a minute)
  - **Google Gemini** — fallback LLM. `https://aistudio.google.com/apikey`
    (about the same)

### A note on API keys

Both providers are free tier — no payment method required, no trial that
expires. Per the brief: paid services are explicitly disallowed, so I
stuck to genuinely free options.

You only strictly need one of the two keys to run the system. If you
supply only `GROQ_API_KEY`, the fallback chain is built without Gemini
(you'll see a single warning line at startup, then everything works
normally on Groq alone). Supplying both is recommended though — if Groq
rate-limits you mid-run, LangChain transparently routes the next call
to Gemini and the prep flow keeps going.

The keys live in `.env` (gitignored). If you'd rather not provision your
own, contact me and I'll share working keys out of band; they'll be
rotated after the review window closes.

### Option A — Docker (single command, ~5 min)

If you have Docker Desktop installed, this is the path of least resistance.

```bash
git clone <repo-url>
cd cloudly-intern

cp .env.example .env
# Open .env, paste your GROQ_API_KEY and GOOGLE_API_KEY.
# Leave DATABASE_URL untouched (it points at the bundled Postgres service).

docker compose up -d --build               # starts Postgres + API
docker compose exec app alembic upgrade head
docker compose exec app python -m src.ingestion.indexer
docker compose exec app python -m src.cli stats
```

The API is then reachable at `http://localhost:8000/docs`.

### Option B — Local Python (no Docker)

```bash
git clone <repo-url>
cd cloudly-intern

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Open .env, paste your Groq and Gemini keys, and set DATABASE_URL
# to your own Postgres (Supabase free tier works great for this).
```

A Supabase pooled connection string looks like:
```
postgresql://postgres.xxxxx:[PASSWORD]@aws-0-region.pooler.supabase.com:5432/postgres
```

Heads-up: if your password contains `@`, `%`, or `&`, URL-encode them
(`%` → `%25`, `&` → `%26`). This caught me out on day one.

Create the schema and ingest the PDF:

```bash
alembic upgrade head                  # creates the 11 KB tables
python -m src.ingestion.indexer       # parses PDF → fills the KB (~50s)
```

Sanity check:

```bash
python -m src.cli stats           # local install
# OR, on Docker:
docker compose exec app python -m src.cli stats
```

You should see something like:
```
  Sections                 10
  Chunks                   195
  Topics                   69
  ChromaDB vectors         195
  Alembic version          46105259c147
```

That's the entire setup. The whole thing — clone, install, configure,
ingest, verify — has run on a fresh machine in under nine minutes for me.


## Running the evaluation scenarios

### Scenario A — cold-start over any two sections

```bash
python -m src.cli scenario-a                       # defaults to sections 1, 2
python -m src.cli scenario-a -s 5 -s 8 -q 5        # custom sections / count
```

Writes:
```
outputs/scenario_a/questions.json
outputs/scenario_a/kb_snapshot.json
```

### Scenario B — three consecutive iterations

```bash
python -m src.cli scenario-b
```

This is the assessment's main evaluation flow. The runner executes three
iterations end-to-end (sections 5+8 → 6+8+9 → 8), simulates answers with a
weighted strategy that reads existing mastery weights, and writes six JSON
files exactly where the brief asks:

```
outputs/scenario_b_iter1/questions_iter1.json
outputs/scenario_b_iter1/kb_snapshot_iter1.json
outputs/scenario_b_iter2/questions_iter2.json
outputs/scenario_b_iter2/kb_snapshot_iter2.json
outputs/scenario_b_iter3/questions_iter3.json
outputs/scenario_b_iter3/kb_snapshot_iter3.json
```

If the substitute PDF uses different section numbering, override the plan:

```bash
python -m src.cli scenario-b --plan "1,2 / 2,3 / 2"
```

The CLI validates each iteration's section IDs against the KB before doing
anything. If you reference a section that doesn't exist, it fails fast with
a clear list of section IDs that *are* available — no cryptic
empty-retrieval failures further down the pipe.

Expect roughly 12–15 minutes of wall-clock time for the full run on the
free Groq tier; the service paces calls so we don't blow through the
6,000 TPM rate limit. Most of that is the LLM thinking, not anything on
our side.

The KB snapshots include the top-5 most recent session records (with each
question's correct answer, the user's simulated answer, the LLM's
explanation, and the source quote that justified the right answer) along
with an `adaptive_state` block showing current topic weights, mastery flags,
and the WEAK/MASTERED summary text that was actually sent to the LLM for
each session. That summary is the paper trail proving the adaptive prompt
was grounded in real history.

### Other CLI commands

```bash
python -m src.cli --help              # full list

python -m src.cli prep -s 2 -q 5      # run one ad-hoc session
python -m src.cli history --limit 10  # recent sessions, table view
python -m src.cli snapshot 19         # export top-5 snapshot for a session
python -m src.cli stats               # KB row counts
```

### Or, if you prefer HTTP

```bash
uvicorn src.main:app --reload --port 8000
```

Then `http://localhost:8000/docs` for an interactive Swagger UI. The same
business logic powers both interfaces — the CLI is a thin wrapper around
the same service layer the FastAPI routes call.


## How it works

The architecture is a fairly conventional retrieval-augmented setup
wrapped in a state machine. There's a one-page diagram in
[SYSTEM_DESIGN.pdf](SYSTEM_DESIGN.pdf); the textual sketch:

```
                    one-time ingestion pipeline
SLATEFALL.pdf ──▶ parser ──▶ chunker ──▶ topic tagger ──▶ embedder ──▶ KB
                                                                       │
                                                                       ▼
                                                            ┌──────────────────┐
                                                            │  PostgreSQL      │
                                                            │  (11 tables)     │
                                                            │  +               │
                                                            │  ChromaDB        │
                                                            │  (vectors)       │
                                                            └────────▲─────────┘
                                                                     │
                       per-session prep flow (LangGraph)              │
START ─▶ detect_mode ─┬─▶ create_session ─▶ generate ─▶ validate ──▶ record
                      │      (cold)              ▲          │
                      │                          └──────────┘
                      │      (adaptive)            retry on
                      └─▶ load_adaptive_context    short batch
                                                         │
                                                         ▼
                                          simulate_and_score ─▶ complete
                                            (updates mastery)
```

The state machine has two real conditional branches: cold-start runs skip
the `load_adaptive_context` node entirely (there's no history to summarize),
and the `validate` node loops back to `generate` if the LLM didn't produce
enough valid MCQs in this pass. The retry is bounded by `max_retries`
in `config.yaml`.


## Stack choices and why

### Backend: FastAPI

REST is required by the brief, and FastAPI gives auto-generated OpenAPI
docs at `/docs`, native Pydantic request validation, and Pydantic models
that double as the LLM's structured output schema. Flask would have worked
but I'd be writing the OpenAPI spec by hand and re-validating data three
times.

### Primary LLM: Groq (LLaMA 3.1 8B Instant). Fallback: Gemini 2.0 Flash

Groq's free tier is the fastest free inference I'm aware of — around
750 tokens/sec on this model, 14,400 requests per day, 6,000 tokens per
minute. Scenario B uses about 30 LLM calls, so headroom is fine.

Gemini Flash is the fallback because its rate-limit pool is independent.
If Groq 429s, LangChain's `with_fallbacks` transparently routes the call
to Gemini and the prep flow doesn't notice. Ollama is wired in too for
fully-offline use, but it's commented out of the default fallback chain
because installing it adds 4 GB to the reviewer's setup and our free APIs
work fine.

The brief says "no paid APIs required" and warns that submissions that
require paid services will be substituted to free tier. Both providers
here are genuine free tier.

### Orchestration: LangChain + LangGraph

LangChain provides the LLM provider abstraction (`ChatGroq`,
`ChatGoogleGenerativeAI`, `ChatOllama`), `ChatPromptTemplate` for prompt
assembly, and `JsonOutputParser(pydantic_object=MCQBatch)` for parsing the
LLM's JSON output directly into a validated Pydantic model — no manual
markdown-stripping or JSON regex.

LangGraph models the prep flow as a state machine. I considered keeping it
as a plain Python function (it's only nine steps) but the conditional
edges and the retry loop are easier to read and reason about as a graph.
The compiled graph also exposes a Mermaid diagram via `.get_graph()`,
which I dump into the docs.

### Storage: PostgreSQL + ChromaDB (separate)

The 11 KB tables benefit from real foreign keys, native JSONB columns
(used for `sections_studied`, `adaptive_context`, `cross_refs`, and
`token_usage`), and proper indexes. SQLite can do most of this but its
JSON ergonomics are weaker and concurrent writes are limited.

ChromaDB sits alongside in local persistent mode. It's purpose-built for
vector similarity search and supports the metadata-based filtering I use
to constrain a search to specific section IDs (`where={"section_id":
{"$in": [5, 8]}}`). pgvector would let me consolidate into one database;
keeping them split makes the role of each clearer in the code, and
swapping pgvector in later would only touch one repository class.

### PDF parsing: PyMuPDF, with the tables left as flat text

The SLATEFALL tables have no visible borders. Neither pdfplumber's
line-based table detector nor PyMuPDF's table finder identifies them.
pdfplumber's text-strategy did identify "tables" but produced fragmented
junk (single rows split across multiple columns, columns with single
characters).

PyMuPDF's plain text extraction does preserve the column-aligned layout
naturally — each cell ends up on its own line. That's readable enough for
both embeddings and for the LLM to interpret. Verified by asking the LLM
"what's the targeting failure rate in fog?" and getting "11.0% (Fog ≤ 12 m
visibility)" back, citing the source verbatim.

I documented the parser's assumptions in code: it expects headers like
`Section N.` and sub-section headers like `N.M Title` (the SLATEFALL
convention). It also handles three-level sub-sections like `9.9.5`, which
appear once in the dossier.

### Chunking strategy: structure-first, with size guards

The dossier's author already chunked the content for me by writing it as
sub-sections. So the primary strategy is: one chunk per sub-section. On
top of that:

- Sub-sections below 100 tokens are merged with their neighbour (otherwise
  the embeddings would be too noisy to be useful).
- Sub-sections above 500 tokens are split via LangChain's
  `RecursiveCharacterTextSplitter` along paragraph/sentence boundaries
  with a 50-token overlap.
- §10.1 (the Glossary) gets per-entry chunking — each of the 69 terms
  becomes its own short chunk. Without this, every "what is X?" query
  would surface the entire glossary as a single noisy result.

The recursive-character splitter is the standard 2026 best-practice for
prose chunking (highest accuracy in recent benchmarks). I'm using it only
as a fallback for oversized sub-sections rather than the primary
mechanism because the document's structure is more reliable than any
generic splitter.

### Embeddings: sentence-transformers `all-MiniLM-L6-v2`

384 dimensions, ~90 MB model, runs comfortably on CPU. A sanity check: the
query "Echo Lock failure triggers" returns the §2.13 chunk at top-1 with
similarity 0.81, the corresponding glossary entry at #2, and the
documented-vulnerabilities chunk (§2.16) at #3.

Larger models (BGE-large, mpnet) would marginally improve recall, but
they're 5× slower on CPU and the reviewer is going to be running this on
whatever laptop they have.

### Token counting: tiktoken

Approximate (LLaMA and Gemini tokenizers differ from cl100k by maybe 5%)
but cheap and consistent. The token-budget code adds a small safety
margin to allow for that drift.


## Knowledge-base schema

Eleven tables across four conceptual layers. Full DDL is in
`alembic/versions/`; the gist:

### Layer 1: PDF structure

- `sections` — 10 rows, one per main section. Carries title, page span,
  and a derived `chunk_count`.
- `chunks` — 195 rows. Carries `sub_section_id` (e.g., `"2.13"` or
  `"5.10+5.11"` for merged chunks), content, token count, page span,
  flags (`has_table`, `has_bullets`), the `chunk_kind`
  (`narrative` / `glossary_entry` / `split`), and a JSONB `cross_refs`
  array listing §X.Y references found in the chunk text.

### Layer 2: the topic system (this is the brain)

- `topics` — 69 rows, one per glossary term. Each has a stable ID, a slug
  (so "Echo Lock", "echo lock", and "Echo-Lock" all resolve to the same
  entity), a category (`power_mechanic`, `combat_tactic`, `adversary`,
  …), and an importance flag.
- `section_topics` — many-to-many. Adds a `depth` enum:
  `primary` / `secondary` / `mention`. Echo Lock is `primary` in §2 (its
  definition lives there), but `mention` in §9 (the Salta incident
  references it). This is what makes cross-section adaptive logic
  possible.
- `chunk_topics` — many-to-many. Records which chunks mention which
  topics, so the retriever can pull "all chunks that touch Echo Lock"
  regardless of section.
- `question_topics` — many-to-many. Tracks which topics a generated MCQ
  tested, with an `is_primary` flag (for impact-weighting in the mastery
  update).

### Layer 3: sessions, questions, answers

- `sessions` — one row per prep run. Notable: `adaptive_context` JSONB
  stores the actual WEAK/MASTERED summary text sent to the LLM for that
  session, plus the allocator's choices, plus any regression events. It's
  a complete audit trail of what the LLM saw.
- `questions` — every generated MCQ, with `source_chunk_ids` (JSONB
  array), `source_quote` (the verbatim PDF excerpt the LLM cited),
  difficulty, and the standard MCQ fields.
- `answers` — one row per question answered, with `is_correct` computed
  at insert time.

### Layer 4: mastery (adaptive state)

- `topic_mastery` — global state per topic: `times_asked`, `times_correct`,
  `times_wrong`, `current_streak` (positive = on-a-roll, negative =
  consecutive misses), `weight` (∈ [0.1, 5.0]), `is_mastered`, last-asked
  and last-wrong timestamps.
- `section_topic_mastery` — same shape, but per (section, topic) pair.
  A user can know the Echo Lock definition cold (§2 mastery high) and
  still fail at recognising it in a Salta-incident question (§9 mastery
  low). The allocator reads both.

### Query patterns

| Brief requirement | How it's satisfied |
|-------------------|-------------------|
| "Given section IDs, retrieve all prior prep sessions" | JSONB `@>` on `sessions.sections_studied`; index in place |
| "Given a session, retrieve question-level results" | `questions JOIN answers WHERE session_id = X` |
| "Topics answered incorrectly across multiple sessions" | `topic_mastery ORDER BY weight DESC` (or `times_wrong DESC`) |
| "KB snapshot at session end" | `build_kb_snapshot()` in `src/output/snapshot.py` |

The snapshot output is the file the brief asks for: top-5 most recent
sessions with full question detail, plus the global `adaptive_state` block
showing top weak topics and currently mastered topics. Reviewer can verify
both that history is being stored correctly and that adaptive prompting is
grounded in real data, in one file.


## How the adaptive loop actually closes

Every answer triggers a cascade:

1. `update_mastery_after_answer` walks the question's `question_topics`
   rows and updates **both** `topic_mastery` (global) and
   `section_topic_mastery` (section-specific). Primary topics take a 1.0×
   impact; secondary topics take 0.5×.

2. The weight is recomputed from the formula
   `base = 1.0 + (times_wrong × 0.5 × impact)`, with multipliers:
   streak ≥ 2 applies a 0.3× decay (decaying weights of topics you're
   getting right); streak ≥ 3 sets weight to 0.1 (mastered); streak < 0
   multiplies by `1 + |streak| × 0.3` (struggling topics get heavier
   priority). Clamped to [0.1, 5.0].

3. If the topic was `is_mastered=True` and the answer was wrong, regression
   is flagged: the mastered flag flips back to False and the weight bumps
   to at least 2.5 to force re-prioritisation.

On the next run that touches an overlapping section, `detect_mode_node`
returns `is_cold_start=False`, the graph routes through
`load_adaptive_context_node`, and a textual summary of the user's weak
and mastered topics gets injected into the LLM prompt verbatim.

Alongside this, the `allocator` runs for each section. It pulls the
effective weight for every section-topic pair (global × section ×
depth_multiplier), sorts, and proportionally allocates the N questions
among the topics. So "5 questions for section 8" might become
"2 about Bofedal-3 (weight 2.5), 2 about Cell Cóndor (weight 1.8),
1 about a fresh topic." These topic seeds are then fed back into the
retriever as the semantic query, so we pull chunks that actually match
the focus topics rather than generic section content.

You can watch this happen directly in
`outputs/scenario_b_iter3/kb_snapshot_iter3.json`. The
`adaptive_state.top_weak_topics` field after Iter 3 lists the topics the
system zeroed in on across all three iterations. The Iter 3 score drop
(20% vs Iter 1's 80%) is the system working as designed: it asked harder,
weak-topic-targeted questions, and the weighted simulator (which mimics a
struggling user on weak topics) missed more of them.


## Limitations, assumptions, and honest gotchas

- **PDF layout assumption.** Section and sub-section detection is regex-
  driven (`^Section N.` / `^N.M Title`). Works for SLATEFALL; would break
  on a PDF with a different convention.
- **Table extraction is best-effort.** SLATEFALL tables have no borders;
  they come through as space-separated rows. The data is preserved and
  the LLM can read it, but it's not formal markdown.
- **LLM non-determinism.** Same prompt + same temperature still varies
  across runs. The hallucination check (source-quote fuzzy match against
  the chunk) catches the worst cases; structural validation (4 unique
  choices, valid `A`/`B`/`C`/`D`, no empty fields) handles the rest.
  Failed MCQs are retried up to 3 times per topic seed.
- **Rate limits.** Groq's free tier is 6,000 tokens/min. The service paces
  LLM calls with a 3-second floor between them; if Groq still 429s, the
  LangChain fallback chain switches to Gemini transparently. Worst case
  for Scenario B is that an iteration takes a minute longer than otherwise.
- **Glossary special-case.** §10.1 doesn't follow the same chunking rule
  as the rest of the dossier — each of the 69 terms is its own chunk.
  Documented in `src/ingestion/chunker.py`.
- **Postgres password URL-encoding.** If your DB password contains `@`,
  `%`, or `&`, you must URL-encode them in `DATABASE_URL`.
- **Re-running the indexer.** Idempotent; sections and topics are upserted
  rather than deleted, so existing sessions and mastery rows survive a
  re-index. Only the chunks and junction tables get rebuilt.


## Project structure

```
src/
├── ingestion/      PDF parser, chunker, topic tagger, embedder, indexer
├── kb/             SQLAlchemy models, ChromaDB repo, mastery math
├── llm/            LangChain provider wrappers + fallback chain
├── rag/            Retriever, token budget, prompt builder, validator
├── prep/           Allocator, simulator, scorer, difficulty controller
├── graph/          LangGraph state machine (state, nodes, graph assembly)
├── services/       Session lifecycle + run_prep_session orchestrator
├── api/            FastAPI routes (sections, prep, sessions, mastery, …)
├── output/         JSON snapshot exporter
├── main.py         FastAPI app entry point
└── cli.py          Click CLI entry point

alembic/            Schema migrations (one initial migration → all 11 tables)
config.yaml         Tunable parameters (token budgets, mastery thresholds, …)
outputs/            Scenario A/B JSON outputs land here
scripts/            verify_full_system.py, verify_pipeline.py, …
data/               The source PDF
kb/chromadb/        ChromaDB persistent storage (generated; gitignored)
```

Everything tunable lives in `config.yaml` — questions per section, the
token-budget split, mastery threshold, chunking sizes, allocator
parameters. Nothing is hardcoded in source.


## Optional bits

- **Tests.** `scripts/verify_full_system.py` runs 68 integration checks
  across phases 0–7 (code hygiene, DB integrity, idempotency, LangGraph
  routing, adaptive cycle, all 16 API endpoints, CLI surface, output
  exporters). It hits real Groq once for the cold/adaptive cycle.
- **Logging.** Per-session token usage and the actual adaptive prompt are
  stored in `sessions.token_usage` and `sessions.adaptive_context` as
  JSONB — visible directly in any snapshot export.
- **Docker.** `Dockerfile` + `docker-compose.yml` are included and spin
  up Postgres and the API in two containers. After `docker compose up -d`,
  every command in this README runs identically inside the container via
  `docker compose exec app …`. The Postgres data lives in a named volume
  (`pgdata`); the generated `outputs/`, `logs/`, and `kb/chromadb/`
  directories are bind-mounted to the host so the reviewer can inspect
  them without entering the container.


## Endpoint reference (brief)

```
GET   /api/v1/health
GET   /api/v1/sections                       list all 10 sections
GET   /api/v1/sections/{id}                  one section
GET   /api/v1/sections/{id}/chunks           chunks for a section
GET   /api/v1/topics                         list all 69 topics
GET   /api/v1/topics/{id}                    one topic
POST  /api/v1/prep/start                     run one adaptive prep session
GET   /api/v1/sessions                       paginated session list
GET   /api/v1/sessions/{id}                  one session with questions/answers
GET   /api/v1/sessions/{id}/snapshot         top-5 KB snapshot
GET   /api/v1/mastery                        global mastery state
GET   /api/v1/mastery/regressions            topics that have regressed
GET   /api/v1/mastery/by-section/{id}        section-specific mastery
POST  /api/v1/scenarios/b/run                trigger Scenario B + write outputs
GET   /api/v1/admin/stats                    KB row counts
POST  /api/v1/admin/reindex                  re-run the ingestion pipeline
```

Full schema for each is in the auto-generated Swagger at `/docs`.
