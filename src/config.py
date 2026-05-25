"""Configuration loader.

Reads config.yaml and .env, validates with Pydantic, exposes a single
`settings` instance for use throughout the project.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ──────────────────────────────────────────────────────────────────────
# Section models — one per top-level key in config.yaml
# ──────────────────────────────────────────────────────────────────────


class AppConfig(BaseModel):
    name: str
    version: str
    environment: Literal["development", "production"]


class PathsConfig(BaseModel):
    pdf_corpus: str
    chromadb_dir: str
    outputs_dir: str
    logs_dir: str


class PdfConfig(BaseModel):
    use_pymupdf_for_prose: bool
    use_pdfplumber_for_tables: bool
    section_regex: str
    sub_section_regex: str
    table_format: str


class ChunkingConfig(BaseModel):
    strategy: str
    min_chunk_tokens: int = Field(ge=1)
    max_chunk_tokens: int = Field(ge=1)
    overlap_tokens: int = Field(ge=0)
    recursive_splitter_separators: List[str]


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    dimensions: int = Field(ge=1)
    device: str
    batch_size: int = Field(ge=1)
    normalize: bool


class VectorDbConfig(BaseModel):
    backend: str
    collection_name: str
    similarity_metric: str
    default_top_k: int = Field(ge=1)


class DatabaseConfig(BaseModel):
    host: str
    port: int = Field(ge=1, le=65535)
    name: str
    user: str
    pool_size: int = Field(ge=1)
    max_overflow: int = Field(ge=0)
    pool_recycle_seconds: int = Field(ge=0)
    echo_sql: bool


class LlmModelMap(BaseModel):
    groq: str
    gemini: str
    ollama: str


class LlmConfig(BaseModel):
    provider: Literal["groq", "gemini", "ollama"]
    fallback_providers: List[Literal["groq", "gemini", "ollama"]]
    model: LlmModelMap
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)
    max_retries: int = Field(ge=0)
    retry_backoff: Literal["exponential", "linear", "constant"]
    retry_base_delay: float = Field(ge=0.0)


class TokenManagementConfig(BaseModel):
    context_limit: int = Field(ge=1000)
    output_reserve: int = Field(ge=0)
    system_prompt_budget: int = Field(ge=0)
    instructions_budget: int = Field(ge=0)
    history_budget: int = Field(ge=0)
    content_budget: int = Field(ge=0)


class McqConfig(BaseModel):
    questions_per_section: int = Field(ge=1, le=50)
    default_difficulty: Literal["easy", "medium", "hard"]
    difficulty_levels: List[Literal["easy", "medium", "hard"]]
    require_source_quote: bool
    hallucination_fuzzy_threshold: float = Field(ge=0.0, le=1.0)
    validate_unique_choices: bool
    glossary_section_strategy: str


class AdaptiveConfig(BaseModel):
    mastery_threshold: int = Field(ge=2)
    weight_min: float = Field(ge=0.0)
    weight_max: float = Field(gt=0.0)
    weight_decay_on_correct: float = Field(ge=0.0)
    weight_boost_on_wrong: float = Field(ge=0.0)
    regression_detection: bool
    regression_weight_floor: float = Field(ge=0.0)
    difficulty_escalation: bool
    all_weak_focus_topics: int = Field(ge=1)
    cross_section_topic_tracking: bool


class HistoryConfig(BaseModel):
    enable_compression: bool
    compression_threshold_tokens: int = Field(ge=0)
    compression_level_iter_1_2: str
    compression_level_iter_3_5: str
    compression_level_iter_6_plus: str
    include_question_themes_to_avoid: bool


class RetrievalConfig(BaseModel):
    strategy: str
    sql_chunks_per_section: int = Field(ge=1)
    vector_top_k: int = Field(ge=1)
    resolve_cross_references: bool
    cross_reference_max_depth: int = Field(ge=0)


class SimulationConfig(BaseModel):
    default_strategy: Literal["weighted", "random", "all_correct"]
    weighted_correct_ratio_strong: float = Field(ge=0.0, le=1.0)
    weighted_correct_ratio_moderate: float = Field(ge=0.0, le=1.0)
    weighted_correct_ratio_weak: float = Field(ge=0.0, le=1.0)
    random_correct_ratio: float = Field(ge=0.0, le=1.0)
    scenario_b_correct_ratio: float = Field(ge=0.0, le=1.0)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    format: Literal["json", "console"]
    file_path: str
    console_output: bool
    include_caller_info: bool
    rotate_max_bytes: int = Field(ge=1)
    rotate_backup_count: int = Field(ge=0)


class ApiConfig(BaseModel):
    host: str
    port: int = Field(ge=1, le=65535)
    reload: bool
    workers: int = Field(ge=1)
    cors_origins: List[str]
    docs_url: str
    redoc_url: str
    openapi_url: str


class SnapshotConfig(BaseModel):
    recent_sessions_count: int = Field(ge=1)
    include_questions: bool
    include_adaptive_state: bool
    include_token_usage: bool
    format: Literal["json"]
    pretty_print: bool


class Secrets(BaseModel):
    """Loaded from .env. Optional — presence checked when actually needed."""

    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None
    database_url: Optional[str] = None
    postgres_db: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# Top-level Settings
# ──────────────────────────────────────────────────────────────────────


class Settings(BaseModel):
    app: AppConfig
    paths: PathsConfig
    pdf: PdfConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectordb: VectorDbConfig
    database: DatabaseConfig
    llm: LlmConfig
    token_management: TokenManagementConfig
    mcq: McqConfig
    adaptive: AdaptiveConfig
    history: HistoryConfig
    retrieval: RetrievalConfig
    simulation: SimulationConfig
    logging: LoggingConfig
    api: ApiConfig
    snapshot: SnapshotConfig
    secrets: Secrets


@lru_cache
def _load_settings() -> Settings:
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    data["secrets"] = {
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL"),
        "database_url": os.getenv("DATABASE_URL"),
        "postgres_db": os.getenv("POSTGRES_DB"),
        "postgres_user": os.getenv("POSTGRES_USER"),
        "postgres_password": os.getenv("POSTGRES_PASSWORD"),
    }

    return Settings(**data)


settings: Settings = _load_settings()
