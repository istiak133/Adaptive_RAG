# Adaptive Document Preparation System — Implementation Guide

> A step-by-step roadmap for building the project from scratch.
> Each phase ends with a **milestone check** — we don't move forward until
> that milestone passes.

---

## 📐 Development Philosophy

**Build vertical slices first, polish later.**

Instead of perfecting one layer before touching the next, we get a crude
end-to-end flow working as early as possible, then go back and harden each
piece. This way we always have a runnable project — even if rough — and
discover integration issues before they become expensive.

**Order discipline:**
1. Foundation (directory, deps, Docker, DB boot)
2. Ingestion (PDF → chunks → KB)
3. Basic RAG (one MCQ generation, no adaptation yet)
4. Sessions & scoring (one full session end-to-end)
5. Adaptive intelligence (mastery, weights, regression)
6. Orchestration (LangGraph state machine)
7. API layer (FastAPI endpoints)
8. CLI layer (Click commands)
9. Scenario B runner + outputs
10. Polish (logging, error handling, tests, README)

---

## 🗺️ PHASE 0 — Foundation Setup

**Goal:** A runnable scaffold. `docker-compose up` brings up Postgres and an
empty FastAPI app. Nothing functional yet, but everything is wired.

### Step 0.1 — Directory structure
Create the full folder skeleton (`src/`, `kb/`, `data/`, `outputs/`, `tests/`,
etc.). Empty `__init__.py` files where needed.

### Step 0.2 — `requirements.txt`
Pin all dependencies with versions:
- fastapi, uvicorn, click
- sqlalchemy, alembic, psycopg2-binary
- pymupdf, pdfplumber
- sentence-transformers, chromadb
- langchain, langgraph, langchain-groq, langchain-google-genai, langchain-ollama
- pydantic, pyyaml, structlog, httpx, tiktoken
- pytest, pytest-asyncio

### Step 0.3 — `config.yaml`
Single source of truth for all tunable parameters: paths, LLM provider,
token budgets, mastery thresholds, etc.

### Step 0.4 — `.env.example` + `.env`
API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`), DB credentials. `.env` is
gitignored.

### Step 0.5 — `Dockerfile`
Python 3.11 base, copy code, install requirements, expose 8000.

### Step 0.6 — `docker-compose.yml`
Two services: `postgres` and `api`. Volumes for pgdata and ChromaDB.

### Step 0.7 — Minimal `src/main.py`
A FastAPI app with a single `/api/v1/health` endpoint returning `{"status":
"ok"}`.

### Step 0.8 — Alembic init
Set up migrations infrastructure (we'll add real migrations in Phase 1).

**Milestone check 0:**
- `docker-compose up -d` succeeds
- `curl localhost:8000/api/v1/health` returns `{"status": "ok"}`
- `docker-compose exec postgres psql -U prep -d prep_kb -c '\dt'` runs

---

## 🗺️ PHASE 1 — Data Ingestion Pipeline

**Goal:** Read the PDF, chunk it, tag topics, embed, write to KB. Run once
at setup. By end of phase, KB has all data and we can query it.

### Step 1.1 — Database models (SQLAlchemy)
Define all 11 tables in `src/kb/models.py`:
- `sections`, `chunks`
- `topics`, `section_topics`, `chunk_topics`, `question_topics`
- `sessions`, `questions`, `answers`
- `topic_mastery`, `section_topic_mastery`

### Step 1.2 — Alembic initial migration
Generate migration from models. Run it. Verify all 11 tables exist in DB.

### Step 1.3 — PDF parser
`src/ingestion/pdf_parser.py`:
- PyMuPDF for prose extraction (page by page)
- pdfplumber for table extraction (preserve structure as markdown)
- Section detector via regex `^Section \d+\.` (corner case #3)
- Sub-section detector via regex `§\d+\.\d+`

### Step 1.4 — Chunker
`src/ingestion/chunker.py`:
- Sub-section boundary chunking
- Enforce min size 100 tokens (merge tiny) — corner case #14
- Enforce max size 500 tokens (split via LangChain `RecursiveCharacterTextSplitter`) — corner case #18
- Capture sub-section IDs (§2.13, §5.4) and titles
- Detect cross-references in chunk text — corner case #4

### Step 1.5 — Topic tagger
`src/ingestion/topic_tagger.py`:
- For each section, use LLM to extract topic names + categories
- Returns structured list: `[{name, slug, category, importance}]`
- Slug normalization (Echo Lock → echo_lock)
- Manual override list for known critical topics (Echo Lock, Tail Momentum,
  Three-Two-One Rule, etc.) to ensure reliability

### Step 1.6 — Embedder
`src/ingestion/embedder.py`:
- Wrapper around `sentence-transformers` `all-MiniLM-L6-v2`
- Returns 384-dim numpy vectors

### Step 1.7 — ChromaDB setup
`src/kb/chroma_repo.py`:
- Initialize persistent ChromaDB at `./kb/chromadb/`
- Create `chunks` collection
- Add/query interface

### Step 1.8 — Indexer (orchestrator)
`src/ingestion/indexer.py`:
- Runs the full pipeline: PDF → parse → chunk → tag → embed → write
- Writes to Postgres (sections, chunks, topics, section_topics, chunk_topics)
- Writes embeddings to ChromaDB with metadata
- Idempotent — safe to re-run

### Step 1.9 — Admin command
CLI subcommand: `python -m src.ingestion.indexer` to run ingestion.

**Milestone check 1:**
- `python -m src.ingestion.indexer` runs successfully
- Postgres has ~10 sections, ~80 chunks, ~50 topics, ~200 section_topics rows
- ChromaDB has ~80 embeddings
- Spot-check: query "Echo Lock" via ChromaDB returns §2.13 chunk on top

---

## 🗺️ PHASE 2 — Basic RAG (No Adaptation Yet)

**Goal:** Generate 5 MCQs from a single section, end-to-end, but adaptive
parts are stubbed (no history yet).

### Step 2.1 — LLM provider abstraction
`src/llm/providers.py`:
- LangChain wrappers: `ChatGroq`, `ChatGoogleGenerativeAI`, `ChatOllama`
- Single `get_llm()` factory function reads provider from config
- Fallback chain: Groq → Gemini → Ollama on failure — corner case #15

### Step 2.2 — Token budget manager
`src/rag/token_budget.py`:
- Budget allocation (system 400, instructions 300, history 1200, content 3600)
- `tiktoken` for counting
- `select_chunks_within_budget()` greedy fill
- Logs token usage

### Step 2.3 — Retriever
`src/rag/retriever.py`:
- `fetch_chunks_for_sections(section_ids)` — SQL filter
- `semantic_search(query, section_ids, top_k)` — ChromaDB query with metadata filter
- `resolve_cross_references(chunks)` — fetch §X.Y referenced chunks — corner case #4
- Hybrid retrieval: combines both

### Step 2.4 — Prompt builder
`src/rag/prompt_builder.py`:
- LangChain `PromptTemplate`
- Variables: `{content}`, `{adaptive_context}`, `{instructions}`, `{count}`
- Different templates for cold-start vs adaptive vs glossary section — corner case #2
- System prompt: "You are an MCQ generator..."

### Step 2.5 — Response validator
`src/rag/validator.py`:
- Pydantic models: `MCQ`, `MCQBatch`
- LangChain `PydanticOutputParser` + `OutputFixingParser` (auto-retry) — corner case #5
- Markdown strip + JSON extract
- Validate: 4 unique choices, valid correct (A-D), explanation present — corner case #7
- Validate count == expected — corner case #6
- Hallucination check: source quote fuzzy-match in chunk — corner case #8

### Step 2.6 — Basic prep service (no adaptation yet)
`src/services/prep_service.py`:
- `generate_questions(section_ids, count)` — wires retrieve → prompt → LLM → validate
- Returns validated MCQ list
- Stubbed history context (empty)

**Milestone check 2:**
- Run a Python script that calls `prep_service.generate_questions([5], 5)`
- Returns 5 valid MCQ objects with question/choices/answer/explanation
- Source quotes verified against retrieved chunks
- Token usage logged per call

---

## 🗺️ PHASE 3 — Sessions, Answers, Scoring

**Goal:** A complete prep session is persisted end-to-end. Adaptive logic is
still stubbed but sessions/answers/questions are all stored in KB.

### Step 3.1 — Session manager
`src/services/session_service.py`:
- `start_session(section_ids, questions_per_section)` — creates row in `sessions`
- `record_question(session_id, mcq, source_chunk_ids)` — writes to `questions` + `question_topics`
- `record_answer(question_id, user_answer)` — writes to `answers`, computes `is_correct`
- `complete_session(session_id)` — sets score_pct, completed_at

### Step 3.2 — Question allocator (simple version)
`src/prep/allocator.py`:
- For now: equal allocation across topics in a section
- Later (Phase 4): weight-based allocation

### Step 3.3 — Answer simulator
`src/prep/simulator.py`:
- Three strategies: `all_correct`, `random`, `weighted`
- Weighted strategy reads topic weights (used in Phase 4)
- Default `weighted` with 65% correct ratio

### Step 3.4 — Scorer
`src/prep/scorer.py`:
- Score a completed session
- For each wrong answer: show correct answer + LLM-generated explanation
- Returns structured score data

### Step 3.5 — Wire it all together
Update `prep_service.run_prep_session()`:
1. `session_service.start_session()`
2. `prep_service.generate_questions()`
3. For each question: record + simulate/collect answer
4. `scorer.score()`
5. `session_service.complete_session()`

**Milestone check 3:**
- One command runs a full prep session
- DB has: 1 session row, 5 questions, 5 answers, score recorded
- Wrong answers have explanations
- Re-query shows session history

---

## 🗺️ PHASE 4 — Adaptive Intelligence (The Differentiator)

**Goal:** Iteration 2+ genuinely uses history. Mastery state grows. Regression
detected. Weights influence retrieval and prompts.

### Step 4.1 — Mastery engine — weight calculation
`src/kb/mastery.py`:
- `update_mastery_after_answer(question_id, is_correct)`
- Cascade through `question_topics` → update both `topic_mastery` and
  `section_topic_mastery`
- Weight formula: base + wrong_count * 0.5 * impact (primary 1.0, secondary 0.5)
- Streak tracking: +ve consecutive correct, −ve consecutive wrong
- Mastery flag: streak ≥ 3 = TRUE

### Step 4.2 — Regression detection
- If `is_mastered == TRUE` and new answer wrong:
  - `is_mastered = FALSE`
  - Weight bumped to ≥ 2.5
  - Log "regression_detected" event — corner case #11

### Step 4.3 — Weighted question allocator (replace simple version)
`src/prep/allocator.py` v2:
- Read all topics for requested sections + their weights (global + section-specific + depth)
- Effective weight = global × section × depth_multiplier
- Mastered topics → effective_weight = 0.1 (almost skip)
- Proportional allocation with minimum 1 for weight > 2.0 topics
- Edge cases:
  - All mastered → trigger difficulty escalation — corner case #9
  - All weak → focus worst 3, easy difficulty — corner case #10

### Step 4.4 — History compressor
`src/rag/history_compressor.py`:
- Reads `topic_mastery` + `section_topic_mastery`
- Generates aggregated stats text (NOT raw question dump) — corner case #17
- Adaptive levels: full per-Q (iter 1-2), aggregated (iter 3-5), summary (iter 6+)
- Includes "recent question themes to avoid" list

### Step 4.5 — Update prompt builder
- Inject adaptive context when not cold-start
- Different template variant for "weak topic focus" vs "diverse coverage"

### Step 4.6 — Difficulty controller
`src/prep/difficulty.py`:
- Based on avg score across recent sessions for this section
- ≥ 85% → hard, 50-85% → medium, < 50% → easy
- Inject difficulty directive into prompt

**Milestone check 4:**
- Run two prep sessions over Section 8:
  - Iter 1: simulate all wrong on "Echo Lock", correct on others
  - Iter 2: should generate questions that emphasise Echo Lock
- Verify `topic_mastery` row for Echo Lock has weight > 2.0, wrong count > 0
- Verify Iter 2's prompt contains "Echo Lock: weight 2.5, wrong 4 times..."
- Run Iter 3 with all-correct simulation → mastery flag eventually flips

---

## 🗺️ PHASE 5 — LangGraph State Machine

**Goal:** Replace the linear prep flow with a proper state machine. Adds
retry logic, checkpointing, clean conditional routing.

### Step 5.1 — Define state
`src/graph/state.py`:
- `PrepState` TypedDict: section_ids, is_cold_start, mastery_data, chunks,
  questions, user_answers, score_data, retry_count, token_usage

### Step 5.2 — Define nodes
`src/graph/nodes.py`:
- `check_kb_node` — set is_cold_start
- `fetch_mastery_node` — populate mastery_data
- `retrieve_chunks_node` — populate chunks
- `compress_history_node`
- `build_prompt_node`
- `call_llm_node` — call LangChain LLM
- `validate_node` — if invalid + retry_count < 3, set retry flag
- `present_node` / `simulate_node`
- `score_node`
- `update_kb_node`
- `export_snapshot_node`

### Step 5.3 — Build graph
`src/graph/graph.py`:
- `StateGraph(PrepState)`
- Add nodes
- Conditional edges:
  - After `check_kb`: cold → skip fetch_mastery → retrieve / adaptive → fetch_mastery → retrieve
  - After `validate`: invalid + retries left → call_llm; else → next
- Add edges
- Compile with SqliteSaver checkpointer

### Step 5.4 — Wire graph into service
Update `prep_service.run_prep_session()` to invoke the compiled graph
instead of calling functions directly.

**Milestone check 5:**
- Same outputs as Phase 3-4, but flow is now graph-driven
- Test invalid JSON → automatic retry → success on 2nd attempt
- Test session resume via checkpoint (kill mid-flow, resume)

---

## 🗺️ PHASE 6 — FastAPI REST API Layer

**Goal:** All 22 endpoints work. Swagger UI accessible. Same business logic
as CLI.

### Step 6.1 — Pydantic request/response schemas
`src/api/schemas.py`:
- `PrepStartRequest`, `PrepStartResponse`
- `AnswerSubmitRequest`, `AnswerSubmitResponse`
- `SessionSummary`, `SessionDetail`, `KBSnapshot`
- `TopicMasteryView`, etc.

### Step 6.2 — Database dependency injection
`src/api/dependencies.py`:
- `get_db()` — yields SQLAlchemy session
- `get_chroma()` — yields ChromaDB client

### Step 6.3 — Route modules
`src/api/routes/`:
- `sections.py` — 3 endpoints
- `topics.py` — 2 endpoints
- `prep.py` — 5 endpoints (start, get, answer, submit, simulate)
- `sessions.py` — 4 endpoints
- `mastery.py` — 4 endpoints
- `scenarios.py` — 2 endpoints (A run, B run)
- `admin.py` — 4 endpoints

### Step 6.4 — Register routes in `main.py`
- Mount all routers under `/api/v1/`
- Configure CORS
- Custom OpenAPI metadata

### Step 6.5 — Error handling middleware
- Global exception handler
- Structured error responses
- 400 for invalid section IDs, 404 for missing sessions, 500 for LLM failures

**Milestone check 6:**
- Visit `localhost:8000/docs` — interactive Swagger UI
- Test each endpoint via Swagger: create session, submit answers, fetch snapshot
- All status codes correct, error responses structured

---

## 🗺️ PHASE 7 — Click CLI Layer

**Goal:** CLI commands work. Calls the same service layer as the API.

### Step 7.1 — CLI structure
`src/cli.py`:
- Click group
- Commands: `prep`, `scenario-b`, `history`, `snapshot`, `ingest`

### Step 7.2 — `prep` command
- Args: `--sections 5 8`, `--questions 5`, `--simulate`/`--interactive`
- Pretty output via Click colors
- Progress bar during LLM generation

### Step 7.3 — `scenario-b` command
- No args needed
- Runs all 3 iterations
- Prints progress
- Saves outputs

### Step 7.4 — `history` command
- Shows recent sessions as a formatted table
- `--last N` flag

### Step 7.5 — `snapshot <session_id>` command
- Dumps KB snapshot to stdout or file

**Milestone check 7:**
- `python cli.py prep --sections 5 8 --simulate` runs end-to-end
- `python cli.py scenario-b` produces all 6 output files in correct folders
- `python cli.py history` shows past sessions

---

## 🗺️ PHASE 8 — Scenario B Runner + Output Format

**Goal:** Produce the exact files the assessment requires, in the exact folder
structure.

### Step 8.1 — Output formatters
`src/output/snapshot.py`:
- `questions_json(session)` — list of MCQs with all metadata
- `kb_snapshot_json(session_id)` — top-5 recent sessions + adaptive state

### Step 8.2 — Scenario B runner
`src/services/scenario_service.py`:
- `run_scenario_b()`:
  - Iter 1: sections [5, 8], weighted-realistic simulation (65% correct)
  - Iter 2: sections [6, 8, 9], same simulation
  - Iter 3: sections [8], same simulation
- After each iter: write outputs to `outputs/scenario_b_iter{N}/`
- Logs per-iter token usage, mastery changes, regressions

### Step 8.3 — Smart simulator
- Weighted by topic weight (weak topics → more wrong) — Touch #6
- Realistic learning pattern across iterations
- Some topics improve, some regress, some stay weak

**Milestone check 8:**
- `outputs/scenario_b_iter1/questions_iter1.json` exists and is valid JSON
- `outputs/scenario_b_iter1/kb_snapshot_iter1.json` shows 1 session record
- After iter 2: snapshot shows 2 sessions, adaptive_context populated
- After iter 3: snapshot shows 3 sessions, mastery shifts visible
- Section 8 topic weights change measurably between iterations

---

## 🗺️ PHASE 9 — Polish & Submission Prep

**Goal:** Production-grade polish. Reviewer-ready.

### Step 9.1 — Structured logging
`src/observability/logging.py`:
- structlog setup (JSON output to file + console)
- Standard event names: session_started, mcq_generated, regression_detected,
  token_budget_exceeded, etc.
- Inject into all services

### Step 9.2 — Error handling sweep
- All file I/O wrapped in try/except
- LLM call retry decorator
- Database transaction rollback on error
- Graceful CLI error messages (no stack traces to user)

### Step 9.3 — Question quality scoring (optional touch)
`src/rag/quality.py`:
- Score each generated MCQ: source verified, choices unique, explanation
  substantial, distractors plausible
- Reject score < 50, regenerate
- Touch #5

### Step 9.4 — Tests
`tests/`:
- Unit: chunker, token budget, weight calculation, mastery transitions
- Integration: full prep flow with mocked LLM
- API: TestClient hits each endpoint
- Aim for ~50% coverage on critical paths

### Step 9.5 — README
- Project overview
- Architecture (link to SYSTEM_DESIGN.pdf)
- Quick start (Docker)
- Manual setup fallback
- Commands for each scenario
- Stack choices with reasoning
- Known limitations
- API documentation pointer

### Step 9.6 — .gitignore, LICENSE, final cleanup
- `.gitignore`: `.env`, `__pycache__`, `kb/chromadb/`, `logs/`, `*.db`
- Verify outputs/ structure matches assessment requirement
- Final `docker-compose up` from clean state to verify reviewer flow

**Milestone check 9:**
- Clone repo to fresh directory
- `docker-compose up -d` works without manual fixes
- `docker-compose exec api python cli.py scenario-b` produces all outputs
- Visit `localhost:8000/docs` — all endpoints listed
- `cat README.md` — clear, complete, well-formatted

---

## ⚠️ Cross-Cutting Concerns

These apply throughout all phases, not as separate steps:

### Configuration
Every new tunable goes into `config.yaml`. Never hardcode.

### Logging
Every new service function gets a structured log entry on entry/exit/error.

### Validation
Every new function that handles external data (LLM output, user input, file
contents) uses Pydantic.

### Corner case awareness
Reference `corner_cases_and_professional_touches.txt` whenever building a
component. Mark each corner case as handled when its mitigation is in place.

### Token tracking
Every LLM call logs input/output token counts to `sessions.token_usage`.

---

## 📅 Suggested Timeline (5-day budget)

| Day | Phases | Output |
|-----|--------|--------|
| 1 | Phase 0 + Phase 1 | Foundation up, PDF ingested, KB populated |
| 2 | Phase 2 + Phase 3 start | MCQs generate from one section, session persists |
| 3 | Phase 3 finish + Phase 4 | Full session flow + adaptive intelligence works |
| 4 | Phase 5 + Phase 6 + Phase 7 | LangGraph + API + CLI all working |
| 5 | Phase 8 + Phase 9 | Scenario B outputs, polish, README, submit |

Buffer: assume each phase takes 25% longer than planned.

---

## ✅ Definition of Done (Whole Project)

The project is "done" when **all** of the following are true:

- [ ] `git clone` + `docker-compose up -d` brings up working API
- [ ] `python cli.py scenario-b` produces correct outputs in correct folders
- [ ] Iteration 2+ visibly emphasises iter 1 weak topics in iter 2 questions
- [ ] KB snapshots show top-5 recent sessions in human-readable form
- [ ] All 22 API endpoints work (Swagger UI verifies)
- [ ] README has clear setup instructions verified to take < 10 minutes
- [ ] No paid APIs required (all free-tier or local)
- [ ] Structured logs visible in `logs/app.log`
- [ ] Pydantic validates LLM output (no hallucinated quotes pass through)
- [ ] Mastery weights update correctly after each session
- [ ] Regression detection works (mastered → wrong → reweighted)
- [ ] At least 6 of the 18 corner cases have explicit handling code
- [ ] At least 5 of the 9 professional touches implemented

---

## 🔄 Working Method

Before starting **each step**, the workflow is:

1. **Explain**: What we're building in this step
2. **Why**: Which assessment requirement or corner case this addresses
3. **How**: The approach we'll take (data structures, algorithms, library calls)
4. **Files**: Which files we'll create or modify
5. **Implement**: Write the code
6. **Verify**: Run a quick check to confirm it works
7. **Move on**: Only after milestone check passes

We don't skip explanations. We don't move on without verifying.
