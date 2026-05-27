# Adaptive Document Preparation System

A CLI + REST backend that ingests a long PDF, asks an LLM to generate
multiple-choice questions from sections the user picks, scores the user's
answers, and — on subsequent runs over the same sections — leans on the
user's mistake history to push question selection toward the topics they
keep getting wrong.

Every generated MCQ has four choices, one correct answer, a short
explanation, and a verbatim `source_quote` from the PDF that justifies
the right answer. The source-quote is also used as a hallucination
check: if the LLM cites something that isn't fuzzily present in the
retrieved context, the question is rejected before it ever reaches the
user. A second guard rejects near-duplicates of questions the user has
already seen in past sessions, so the brief's "avoid excessive
repetition" requirement holds even when the LLM ignores the prompt-side
instruction to vary its output.

Built for the Cloudly AI/ML intern take-home. The corpus is
`SLATEFALL_DOSSIER.pdf`, a 50-page fictional dossier with ten sections.
The adaptive loop is the part the assessment cares about most, and it's
what most of the moving parts in this repo exist to enable.


## Contents

1. [At a glance](#at-a-glance)
2. [Quick start](#quick-start)
3. [Running the evaluation scenarios](#running-the-evaluation-scenarios)
4. [Verifying the adaptive loop](#verifying-the-adaptive-loop)
5. [How it works](#how-it-works)
6. [The adaptive loop, end-to-end](#the-adaptive-loop-end-to-end)
7. [Knowledge-base schema](#knowledge-base-schema)
8. [Stack choices and reasoning](#stack-choices-and-reasoning)
9. [Configuration](#configuration)
10. [Output file schemas](#output-file-schemas)
11. [Assessment criteria coverage](#assessment-criteria-coverage)
12. [Limitations and assumptions](#limitations-and-assumptions)
13. [Project layout](#project-layout)
14. [API endpoint reference](#api-endpoint-reference)


## At a glance

A two-minute summary if you're triaging submissions:

- **Stack**: Python 3.9+, FastAPI, Click CLI, SQLAlchemy + Alembic on
  PostgreSQL, ChromaDB for vectors, LangChain + LangGraph for LLM
  orchestration, Groq (LLaMA 3.1 8B) primary and Gemini 2.0 Flash
  fallback. Both providers are free tier.
- **Setup**: under ten minutes on a fresh machine. Two paths: Docker
  (one command after `.env`) or local Python + your own Postgres.
- **Scenario B is the differentiator**: `python -m src.cli scenario-b`
  runs the three required iterations end-to-end and writes all six JSON
  files the brief asks for. Mastery state persists across iterations and
  visibly steers question selection on Iter 2 and Iter 3.
- **Adaptive loop is wired through real code, not vibes**: the
  `topic_mastery` and `section_topic_mastery` tables are updated after
  every answer; the allocator reads from them; the prompt builder
  injects a WEAK/MASTERED summary into the LLM input verbatim. There's a
  paper trail in `sessions.adaptive_context` (JSONB) showing exactly
  what the LLM saw on each run.
- **LangGraph is genuine, not cosmetic**: nine nodes, two real
  conditional branches (cold-start vs. adaptive routing, validate vs.
  retry). The compiled graph exposes a Mermaid diagram.
- **The reviewer-facing outputs** sit in `outputs/scenario_a/` and
  `outputs/scenario_b_iter{1,2,3}/`. A representative run is already
  committed so you can inspect the shape without spending API quota.


## Quick start

Prerequisites:

- Python 3.9+, or just Docker if you'd rather not deal with Python
- A PostgreSQL instance — any will do. Supabase's free tier was used
  during development; the Docker path below bundles one automatically.
- Free API keys (see [API keys](#a-note-on-api-keys) below):
  - **Groq** — primary LLM. https://console.groq.com (sign up + create
    key in under a minute)
  - **Google Gemini** — fallback LLM. https://aistudio.google.com/apikey

### A note on API keys

> **Reviewer shortcut — skip provisioning your own keys:** if you'd
> rather not spend a minute on Groq + Gemini signup, contact me at the
> address on the submission form and I will send pre-provisioned
> working keys directly. They are valid throughout the review window
> and will be rotated immediately after it closes. Committed secrets
> are an anti-pattern on a public repository, so the keys are shared
> out of band rather than in `.env.example`.

Both providers are genuine free tier. No payment method is required
and no trial expires. The assessment brief disallows paid services, so
the project uses options that stay free indefinitely.

Only one of the two keys is strictly required. If you supply only
`GROQ_API_KEY`, the fallback chain is built without Gemini — you will
see one warning line at startup and the system runs normally on Groq
alone. Supplying both is recommended, though: if Groq rate-limits a
call mid-run, the LangChain fallback transparently routes it to Gemini
and the prep flow continues without interruption.

The keys live in `.env`, which is gitignored.

### Option A — Docker (one command after `.env`)

If you have Docker Desktop installed, this is the fastest path.

```bash
git clone https://github.com/istiak133/Adaptive_RAG
cd Adaptive_RAG

cp .env.example .env
# Open .env, paste your GROQ_API_KEY and GOOGLE_API_KEY.
# Leave DATABASE_URL untouched — it points at the bundled Postgres service.

docker compose up -d --build
docker compose exec app alembic upgrade head
docker compose exec app python -m src.ingestion.indexer
docker compose exec app python -m src.cli stats
```

The API is then reachable at http://localhost:8000/docs. Bind mounts
expose `outputs/`, `logs/`, and `kb/chromadb/` on the host so you can
inspect generated files without entering the container.

### Option B — Local Python (no Docker)

```bash
git clone https://github.com/istiak133/Adaptive_RAG
cd Adaptive_RAG

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Open .env, paste your Groq and Gemini keys, and set DATABASE_URL to
# your own Postgres (Supabase free tier is a good fit).
```

A Supabase pooled connection string looks like:

```
postgresql://postgres.xxxxx:[PASSWORD]@aws-0-region.pooler.supabase.com:5432/postgres
```

Important: if your password contains `@`, `%`, or `&`, URL-encode them
(`%` → `%25`, `&` → `%26`). SQLAlchemy will otherwise fail to parse
the connection string.

Now create the schema and ingest the PDF:

```bash
alembic upgrade head                  # creates the 11 KB tables
python -m src.ingestion.indexer       # parses PDF → fills the KB (~50s)
```

Sanity check:

```bash
python -m src.cli stats                          # local install
# or
docker compose exec app python -m src.cli stats  # Docker
```

You should see:

```
  Sections                 10
  Chunks                   195
  Topics                   69
  ChromaDB vectors         195
  Alembic version          46105259c147
```

Clone, install, configure, ingest, verify — the full setup completes
in under nine minutes on a fresh machine.


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

A committed sample is already in `outputs/scenario_a/` from session 20.

### Scenario B — three consecutive iterations

```bash
python -m src.cli scenario-b
```

This is the assessment's main evaluation flow. The runner executes three
iterations end-to-end (sections 5+8 → 6+8+9 → 8), simulates answers
with a weighted strategy that reads existing mastery weights, and
writes the six JSON files the brief asks for:

```
outputs/scenario_b_iter1/questions_iter1.json
outputs/scenario_b_iter1/kb_snapshot_iter1.json
outputs/scenario_b_iter2/questions_iter2.json
outputs/scenario_b_iter2/kb_snapshot_iter2.json
outputs/scenario_b_iter3/questions_iter3.json
outputs/scenario_b_iter3/kb_snapshot_iter3.json
```

Expect roughly 12–15 minutes of wall-clock time for the full run on the
Groq free tier. The service paces calls with a 3-second floor between
them so we don't trip the 6,000-tokens-per-minute rate limit. Most of
the wall-clock is the LLM thinking, not anything on our side.

The KB snapshots include the five most recent session records — each
with the full question detail (correct answer, simulated user answer,
LLM explanation, source quote from the PDF) — and an `adaptive_state`
block showing current topic weights, mastery flags, and the
WEAK/MASTERED summary text actually sent to the LLM for each session.
That summary is the paper trail showing the adaptive prompt was grounded
in real history.

If the substitute PDF uses different section numbering, override the
plan:

```bash
python -m src.cli scenario-b --plan "1,2 / 2,3 / 2"
```

The CLI validates each iteration's section IDs against the KB before
doing anything. Reference a section that doesn't exist and it fails
fast with a clear list of the IDs that *are* available — no cryptic
empty-retrieval failures further down the pipe.

### Other CLI commands

```bash
python -m src.cli --help              # full list

python -m src.cli prep -s 2 -q 5      # one ad-hoc session
python -m src.cli history --limit 10  # recent sessions, table view
python -m src.cli snapshot 19         # export top-5 KB snapshot for session 19
python -m src.cli stats               # KB row counts + adaptive state summary
python -m src.cli ingest              # re-run the ingestion pipeline
```

### REST API

```bash
uvicorn src.main:app --reload --port 8000
```

Then http://localhost:8000/docs for interactive Swagger. The same
business logic powers both interfaces — the CLI is a thin wrapper around
the same service layer the FastAPI routes call into. 16 endpoints; see
[the endpoint reference](#api-endpoint-reference) at the bottom.


## Verifying the adaptive loop

Two commands prove the system isn't just generating MCQs in a vacuum.

**1. The cold → adaptive arc is visible in session history**:

```bash
python -m src.cli history --limit 5
```

After a Scenario B run you'll see something like:

```
  ID  Sections             Score   Cold  Difficulty  Started
  19  [8               ]     20%     no  medium    2026-05-26 14:40
  18  [6,8,9           ]     87%     no  medium    2026-05-26 14:33
  17  [5,8             ]     80%    yes  medium    2026-05-26 14:27
```

The `Cold` column flips to `no` after the first iteration — the system
correctly detected prior history. The Iter 3 score drop to 20% is the
adaptive system at work: it asked harder, weak-topic-targeted questions,
and the weighted simulator (which mimics a struggling user on weak
topics) missed more of them.

**2. The prompt the LLM actually saw is recorded in the DB**:

Open any kb_snapshot file (`outputs/scenario_b_iter2/kb_snapshot_iter2.json`)
and look at `recent_sessions[0].adaptive_context.summary` for the
WEAK/MASTERED block injected verbatim into the prompt for that run.
That value is read straight from the `sessions.adaptive_context`
JSONB column — it's not regenerated for the snapshot, so what you see
is the literal text the LLM consumed.

If you want a third proof point, the `adaptive_state.top_weak_topics`
field in any iter3 snapshot shows the topics the system zeroed in on
across all three iterations. Cross-reference that list with the topics
the LLM actually asked about in iter3, and you'll see the targeting in
action.


## How it works

The architecture is a fairly conventional retrieval-augmented setup
wrapped in a state machine. A one-page diagram lives in
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

### Inside the LangGraph state machine

Eight nodes, two real conditional branches, defined in `src/graph/`:

| Node | What it does |
|------|--------------|
| `detect_mode_node` | Looks for prior completed sessions covering overlapping sections; sets `is_cold_start` |
| `load_adaptive_context_node` | Cold-start runs skip this. Otherwise builds the WEAK/MASTERED summary text from `topic_mastery` |
| `create_session_node` | Inserts a row in `sessions`, captures the adaptive context as JSONB |
| `generate_node` | Per-section: allocator decides topic seeds → retriever pulls chunks → LLM is invoked → response parsed |
| `validate_generation_node` | Counts accepted MCQs; if short of target and retries remain, signals retry |
| `record_node` | Persists every generated MCQ into `questions` + `question_topics` |
| `simulate_and_score_node` | Runs the chosen answer-simulation strategy and inserts `answers` rows |
| `complete_node` | Updates mastery state (the closing step), marks the session complete |

The two conditional edges are:

- `detect_mode → {create_session, load_adaptive_context}`, depending on
  whether this is a cold start.
- `validate_generation → {generate, record}`, depending on whether we
  hit the requested MCQ count this pass. The retry is bounded by
  `llm.max_retries` in `config.yaml` (default 3).

The prep flow could be implemented as a plain Python function — it is
only eight steps — but the conditional branches and the retry loop
are easier to read and reason about as a graph. The compiled state
machine also exposes a Mermaid diagram via `.get_graph()`, which has
proved useful for verification.


## The adaptive loop, end-to-end

Every answer triggers a cascade:

1. `update_mastery_after_answer` walks the question's `question_topics`
   rows and updates **both** `topic_mastery` (global) and
   `section_topic_mastery` (section-specific). Primary topics take a 1.0×
   impact; secondary topics take 0.5×.

2. Weight is recomputed:

   ```
   base   = 1.0 + (times_wrong × 0.5 × impact)
   streak ≥ 2  → multiply by 0.3   (decay weights of topics you're getting right)
   streak ≥ 3  → set to 0.1        (mastered)
   streak < 0  → multiply by 1 + |streak| × 0.3   (struggling = heavier priority)
   ```

   Clamped to [0.1, 5.0].

3. If the topic was `is_mastered=True` and the answer was wrong,
   regression is flagged: the flag flips back to False and the weight
   bumps to at least 2.5 to force re-prioritisation on the next run.

On the next run that touches an overlapping section, `detect_mode_node`
returns `is_cold_start=False`, the graph routes through
`load_adaptive_context_node`, and a textual summary of the user's weak
and mastered topics gets injected into the LLM prompt verbatim.

Alongside this, the allocator runs for each section. It pulls the
effective weight for every section-topic pair (global × section ×
depth_multiplier), sorts, and proportionally allocates the N questions
among the topics. So "5 questions for section 8" might become "2 about
Bofedal-3 (weight 2.5), 2 about Cell Cóndor (weight 1.8), 1 about a
fresh topic." Those topic seeds become the semantic query the retriever
uses, so we pull chunks that actually match the focus topics rather
than generic section content.

You can watch this happen directly in
`outputs/scenario_b_iter3/kb_snapshot_iter3.json`. The
`adaptive_state.top_weak_topics` field after Iter 3 lists the topics
the system targeted across all three iterations. The Iter 3 score drop
(20% vs. Iter 1's 80%) is the system working as designed.


## Knowledge-base schema

Eleven tables across four conceptual layers. Full DDL is in
`alembic/versions/`; the gist:

### Layer 1: PDF structure

- `sections` — 10 rows, one per main section. Title, page span, derived
  `chunk_count`.
- `chunks` — 195 rows. Carries `sub_section_id` (e.g., `"2.13"` or
  `"5.10+5.11"` for merged chunks), content, token count, page span,
  flags (`has_table`, `has_bullets`), a `chunk_kind` enum
  (`narrative` / `glossary_entry` / `split`), and a JSONB `cross_refs`
  array listing §X.Y references found in the chunk text.

### Layer 2: the topic system (this is the brain)

- `topics` — 69 rows, one per glossary term. Each has a stable ID, a
  slug (so "Echo Lock", "echo lock", and "Echo-Lock" all resolve to the
  same entity), a category enum (`power_mechanic`, `combat_tactic`,
  `adversary`, …), and an importance flag.
- `section_topics` — many-to-many. Carries a `depth` enum:
  `primary` / `secondary` / `mention`. Echo Lock is `primary` in §2 (its
  definition lives there) but `mention` in §9 (the Salta incident
  references it). This is what makes cross-section adaptive logic work.
- `chunk_topics` — many-to-many. Records which chunks mention which
  topics, so the retriever can pull "all chunks that touch Echo Lock"
  regardless of section.
- `question_topics` — many-to-many. Tracks which topics a generated MCQ
  tested, with an `is_primary` flag for impact-weighting in the mastery
  update.

### Layer 3: sessions, questions, answers

- `sessions` — one row per prep run. Notable: `adaptive_context` JSONB
  stores the actual WEAK/MASTERED summary text sent to the LLM for that
  session, plus the allocator's choices, plus any regression events. A
  complete audit trail of what the LLM saw.
- `questions` — every generated MCQ, with `source_chunk_ids` (JSONB
  array), `source_quote` (the verbatim PDF excerpt the LLM cited),
  difficulty, and the standard MCQ fields.
- `answers` — one row per question answered, with `is_correct` computed
  at insert time.

### Layer 4: mastery (adaptive state)

- `topic_mastery` — global state per topic: `times_asked`,
  `times_correct`, `times_wrong`, `current_streak` (positive = on a
  roll, negative = consecutive misses), `weight` (∈ [0.1, 5.0]),
  `is_mastered`, last-asked and last-wrong timestamps.
- `section_topic_mastery` — same shape, but per (section, topic) pair.
  A user can know the Echo Lock definition cold (high mastery in §2)
  and still fail at recognising it in a Salta-incident question (low
  mastery in §9). The allocator reads both.

### Query patterns

| Brief requirement | How it's satisfied |
|-------------------|-------------------|
| "Given section IDs, retrieve all prior prep sessions" | JSONB `@>` on `sessions.sections_studied`; GIN index in place |
| "Given a session, retrieve question-level results" | `questions JOIN answers WHERE session_id = X` |
| "Topics answered incorrectly across multiple sessions" | `topic_mastery ORDER BY weight DESC` (or `times_wrong DESC`) |
| "KB snapshot at session end" | `build_kb_snapshot()` in [src/output/snapshot.py](src/output/snapshot.py) |

The snapshot output is exactly the file the brief asks for: top-5 most
recent sessions with full question detail, plus the global
`adaptive_state` block showing top weak topics and currently mastered
topics. A reviewer can confirm both that history is being stored and
that adaptive prompting is grounded in real data, from one file.


## Stack choices and reasoning

### Backend: FastAPI

REST is required by the brief, and FastAPI provides auto-generated
OpenAPI docs at `/docs`, native Pydantic request validation, and
Pydantic models that double as the LLM's structured output schema.
Flask would have worked but would require the OpenAPI spec to be
written by hand and the request body to be re-validated at multiple
layers.

### Primary LLM: Groq (LLaMA 3.1 8B Instant). Fallback: Gemini 2.0 Flash

Groq's free tier offers the fastest free inference currently available
for this model — approximately 750 tokens/sec, with limits of 14,400
requests per day and 6,000 tokens per minute. Scenario B requires
about 30 LLM calls, leaving comfortable headroom.

Gemini Flash is the fallback because its rate-limit pool is independent.
If Groq 429s, LangChain's `with_fallbacks` routes the call to Gemini
and the prep flow doesn't notice. Ollama is wired in too for fully
offline use, but it's not in the default fallback chain because
installing it adds 4 GB to a reviewer's setup and the free APIs cover
us.

### Orchestration: LangChain + LangGraph

LangChain provides the LLM provider abstraction (`ChatGroq`,
`ChatGoogleGenerativeAI`, `ChatOllama`), `ChatPromptTemplate` for
prompt assembly, and `JsonOutputParser(pydantic_object=MCQBatch)` for
parsing the LLM's JSON output directly into a validated Pydantic model.
No manual markdown stripping, no JSON regex.

LangGraph models the prep flow as a state machine with real conditional
edges (cold-start routing, validate-then-retry). The compiled graph
exposes a Mermaid diagram via `.get_graph()`, which is what's reproduced
above.

### Storage: PostgreSQL + ChromaDB (separate)

The 11 KB tables benefit from real foreign keys, native JSONB columns
(used for `sections_studied`, `adaptive_context`, `cross_refs`,
`token_usage`), and proper indexes. SQLite can do most of this, but its
JSON ergonomics are weaker and concurrent writes are limited.

ChromaDB sits alongside in local persistent mode. It is purpose-built
for vector similarity search and supports metadata-based filtering to
constrain a search to specific section IDs
(`where={"section_id": {"$in": [5, 8]}}`). pgvector would consolidate
both stores into one database; keeping them split makes the role of
each clearer in the code, and swapping pgvector in later would only
touch one repository class.

### PDF parsing: PyMuPDF, with tables left as flat text

The SLATEFALL tables have no visible borders. Neither pdfplumber's
line-based table detector nor PyMuPDF's table finder identifies them.
pdfplumber's text-strategy did identify "tables" but produced
fragmented junk — single rows split across multiple columns, columns
with single characters.

PyMuPDF's plain text extraction preserves the column-aligned layout
naturally: each cell lands on its own line. That's readable enough for
both the embedder and the LLM. A spot check: asking the LLM "what's
the targeting failure rate in fog?" returns "11.0% (Fog ≤ 12 m
visibility)", citing the source verbatim.

The parser's assumptions are documented in code. It expects headers
like `Section N.` and sub-section headers like `N.M Title` (the
SLATEFALL convention). It also handles three-level sub-sections like
`9.9.5`, which appear once in the dossier.

### Chunking strategy: structure-first, with size guards

The dossier's author already chunked the content by writing it as
sub-sections, so the primary strategy is one chunk per sub-section.
On top of that:

- Sub-sections below 100 tokens are merged with a neighbour. Otherwise
  the embeddings are too noisy to be useful.
- Sub-sections above 500 tokens are split via LangChain's
  `RecursiveCharacterTextSplitter` along paragraph and sentence
  boundaries, with a 50-token overlap.
- §10.1 (the Glossary) gets per-entry chunking — each of the 69 terms
  becomes its own short chunk. Without this, every "what is X?" query
  would surface the entire glossary as a single noisy result.

The recursive-character splitter is the standard 2026 best-practice
for prose chunking (highest accuracy in recent benchmarks). It is
applied only as a fallback for oversized sub-sections, since the
document's existing structure is more reliable than any generic
splitter.

### Embeddings: sentence-transformers `all-MiniLM-L6-v2`

384 dimensions, ~90 MB model, runs comfortably on CPU. A spot check:
the query "Echo Lock failure triggers" returns the §2.13 chunk at top-1
with similarity 0.81, the corresponding glossary entry at #2, and the
documented-vulnerabilities chunk (§2.16) at #3.

Larger models (BGE-large, mpnet) would marginally improve recall but
are 5× slower on CPU, and the reviewer is going to be running this on
whatever laptop they have.

### Token counting: tiktoken

Approximate — LLaMA and Gemini tokenizers differ from cl100k by maybe
5% — but cheap and consistent. The token-budget code adds a small
safety margin to allow for that drift.


## Configuration

Everything tunable lives in [config.yaml](config.yaml) (203 lines, 17
sections, fully commented). Nothing important is hardcoded in source.
The sections you're most likely to touch:

| Section | What's in it |
|---------|--------------|
| `llm` | Provider, fallback chain, model IDs, temperature, max tokens, timeouts, retries, inter-call delay (the Groq TPM pacing) |
| `mcq` | Default difficulty, questions-per-section, hallucination fuzzy threshold, source-quote requirement |
| `adaptive` | Weight formula constants, mastery threshold (default 3 correct in a row), regression weight bump (default 2.5) |
| `chunking` | Min/max chunk sizes, recursive splitter overlap, glossary special-case flags |
| `retrieval` | `vector_top_k` (default 5), metadata filter behaviour |
| `history` | Adaptive context token budget (1200), compression behaviour |
| `simulation` | Per-strategy correct ratios for the simulator's weighted mode |
| `paths` | Output/log/chromadb directories (all Docker-volume-friendly) |

Two parameters that matter for the assessment:

- `llm.inter_call_delay_seconds: 3.0` — paces calls so Scenario B
  doesn't trip the 6,000 TPM limit. Raise this if you hit 429s.
- `retrieval.vector_top_k: 5` — number of chunks fed to the LLM per
  call. Reduced from 10 to 5 after observing that the larger value
  regularly pushed the prompt above Groq's 6,000 TPM ceiling.


## Output file schemas

### `questions_iter{N}.json`

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
      "source_quote": "verbatim excerpt from the PDF the LLM cited",
      "user_answer": "C",
      "is_correct": true
    }
  ]
}
```

### `kb_snapshot_iter{N}.json`

```json
{
  "snapshot_after_session": 17,
  "exported_at": "2026-05-26T14:27:50Z",
  "total_sessions_in_kb": 17,
  "recent_sessions": [
    {
      "session_id": 17,
      "sections_studied": [5, 8],
      "is_cold_start": true,
      "difficulty_level": "medium",
      "score_pct": 80.0,
      "correct": 8,
      "wrong": 2,
      "total_questions": 10,
      "started_at": "...",
      "completed_at": "...",
      "adaptive_context": {
        "summary": "WEAK: Echo Lock (wrong 1×) / MASTERED: ..."
      },
      "token_usage": { "...": "..." },
      "questions": [ /* full question + answer detail */ ]
    }
    /* ... up to 5 most recent ... */
  ],
  "adaptive_state": {
    "top_weak_topics": [
      {"name": "Bofedal-3", "slug": "bofedal_3",
       "weight": 1.95, "times_wrong": 1, "times_correct": 1, "streak": -1}
    ],
    "mastered_topics": []
  }
}
```

`recent_sessions[i].adaptive_context.summary` is the literal text
injected into the LLM prompt for that session. That's the field a
reviewer should look at to confirm the adaptive logic is real and not
performative.


## Assessment criteria coverage

| Brief requirement | Where it's implemented |
|-------------------|----------------------|
| PDF ingestion + sub-section chunking | [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py), [src/ingestion/chunker.py](src/ingestion/chunker.py) |
| Vector-based semantic retrieval | [src/rag/retriever.py](src/rag/retriever.py), [src/kb/chroma_repo.py](src/kb/chroma_repo.py) |
| MCQ generation grounded in PDF context | [src/rag/prompt_builder.py](src/rag/prompt_builder.py), [src/services/prep_service.py](src/services/prep_service.py) |
| Output schema validation | [src/rag/validator.py](src/rag/validator.py) (Pydantic + hallucination check) |
| Free-tier LLM with graceful key handling | [src/llm/providers.py](src/llm/providers.py) |
| Sessions + answers persisted | `sessions`, `questions`, `answers` tables |
| Adaptive question generation across runs | [src/kb/mastery.py](src/kb/mastery.py), [src/rag/history_compressor.py](src/rag/history_compressor.py), [src/prep/allocator.py](src/prep/allocator.py) |
| Mastery tracking + regression detection | [src/kb/mastery.py](src/kb/mastery.py) |
| Scenario A end-to-end | `scenario-a` command in [src/cli.py](src/cli.py) |
| Scenario B with three iterations | `scenario-b` command in [src/cli.py](src/cli.py) |
| Top-5 KB snapshot at session end | [src/output/snapshot.py](src/output/snapshot.py) |
| REST API | [src/api/routes/](src/api/routes/), 16 endpoints |
| CLI | [src/cli.py](src/cli.py), 7 commands |
| LangChain + LangGraph | [src/graph/](src/graph/), [src/llm/](src/llm/), [src/rag/](src/rag/) |
| Configuration | [config.yaml](config.yaml) |
| Simulated user answers (3 strategies) | [src/prep/simulator.py](src/prep/simulator.py) |

The five "Important Notes" from the brief are addressed as follows:

1. **No paid APIs.** Groq and Gemini free tiers only. Provider chain
   skips missing-key fallbacks with a warning rather than crashing.
2. **Simulated user answers are acceptable.** Three strategies
   implemented; the default (`weighted`) reads mastery weights to bias
   accuracy realistically.
3. **Section-numbering mapping.** Parser regex accepts any integer
   numbering. The `scenario-b --plan` flag lets a reviewer remap the
   iterations if their substitute PDF uses different numbers.
4. **LLM output non-determinism.** Three validation guards before an
   MCQ reaches the user: Pydantic structural check (4 unique choices,
   valid `A`/`B`/`C`/`D`, non-empty explanation), source-quote
   hallucination check (fuzzy match against retrieved context), and a
   near-duplicate guard (rejects when `question_text` similarity exceeds
   0.65 against any previously-asked question for these sections).
   Failed generations retry up to 3× per topic seed.
5. **CLI-first.** `scenario-b` runs the full three-iteration adaptive
   flow in one command. REST exists alongside but isn't required.


## Limitations and assumptions

- **PDF layout assumption.** Section and sub-section detection is
  regex-driven (`^Section N.` / `^N.M Title`). Works for SLATEFALL,
  would break on a PDF with a different convention.
- **Table extraction is best-effort.** SLATEFALL tables have no borders;
  they come through as space-separated rows. The data is preserved and
  the LLM can read it, but it isn't formal markdown.
- **LLM non-determinism.** Same prompt + same temperature still varies
  across runs. Three validators catch the worst cases: structural
  (Pydantic schema + 4 unique choices), source-quote hallucination
  (fuzzy match against retrieved context), and near-duplicate detection
  against past `question_text` for these sections (rejects at
  SequenceMatcher ratio ≥ 0.65, tuned empirically). Failed MCQs retry
  up to 3× per topic seed.
- **Rate limits.** Groq's free tier is 6,000 tokens/min. The service
  paces LLM calls with a 3-second floor between them; if Groq still
  429s, the LangChain fallback chain switches to Gemini transparently.
  Worst case for Scenario B is an iteration taking a minute longer than
  otherwise.
- **Glossary special-case.** §10.1 doesn't follow the same chunking
  rule as the rest of the dossier — each of the 69 terms is its own
  chunk. Documented in [src/ingestion/chunker.py](src/ingestion/chunker.py).
- **Postgres password URL-encoding.** If your DB password contains `@`,
  `%`, or `&`, URL-encode them in `DATABASE_URL`.
- **Re-running the indexer.** Idempotent; sections and topics are
  upserted rather than deleted, so existing sessions and mastery rows
  survive a re-index. Only the chunks and junction tables get rebuilt.
- **Python 3.9 deprecation warnings.** A couple of Google packages emit
  `FutureWarning` on Python 3.9. They're cosmetic; the system runs
  fine. Python 3.11+ avoids them entirely (and is what the Docker
  image uses).


## Project layout

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
config.yaml         Tunable parameters (see Configuration above)
outputs/            Scenario A/B JSON outputs land here
scripts/            verify_full_system.py, verify_pipeline.py, verify_connectivity.py
data/               The source PDF + the assessment brief
kb/chromadb/        ChromaDB persistent storage (generated; gitignored)
Dockerfile          Multi-stage Python 3.11 image (~250 MB runtime)
docker-compose.yml  Postgres 16 + app, with healthcheck gating
```

### Verification scripts

`scripts/` holds a few diagnostic tools, useful if anything looks off:

- `verify_connectivity.py` — quick check that Postgres + ChromaDB +
  Groq are all reachable. Run first if something's failing.
- `verify_pipeline.py` — exercises the ingestion + retrieval path
  without spending API quota.
- `verify_full_system.py` — 68 integration checks across the whole
  system: code hygiene, DB integrity, idempotency, LangGraph routing,
  the adaptive cycle, every API endpoint, every CLI command, output
  exporters. This one does hit Groq for the cold/adaptive cycle.


## API endpoint reference

```
GET   /api/v1/health                          liveness probe
GET   /api/v1/sections                        list all 10 sections
GET   /api/v1/sections/{id}                   one section
GET   /api/v1/sections/{id}/chunks            chunks for a section
GET   /api/v1/topics                          list all 69 topics
GET   /api/v1/topics/{id}                     one topic
POST  /api/v1/prep/start                      run one adaptive prep session
GET   /api/v1/sessions                        paginated session list
GET   /api/v1/sessions/{id}                   one session with questions/answers
GET   /api/v1/sessions/{id}/snapshot          top-5 KB snapshot
GET   /api/v1/mastery                         global mastery state
GET   /api/v1/mastery/regressions             topics that have regressed
GET   /api/v1/mastery/by-section/{id}         section-specific mastery
POST  /api/v1/scenarios/b/run                 trigger Scenario B + write outputs
GET   /api/v1/admin/stats                     KB row counts
POST  /api/v1/admin/reindex                   re-run the ingestion pipeline
```

Full schema for each is in the auto-generated Swagger at `/docs`.
