"""Alembic environment configuration.

DATABASE_URL is read from src.config (which loads .env) and passed directly
to the engine / offline-mode context. We deliberately bypass ConfigParser's
sqlalchemy.url option because the Supabase URL contains URL-encoded
characters (%25, %26) that ConfigParser tries to interpolate.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# Make src.config importable when alembic runs from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

if settings.secrets.database_url is None:
    raise RuntimeError(
        "DATABASE_URL is not set. Add it to .env before running alembic."
    )

DATABASE_URL = settings.secrets.database_url


# target_metadata will be set in Phase 1 once SQLAlchemy models exist:
#   from src.kb.models import Base
#   target_metadata = Base.metadata
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL without DB connection)."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to DB and applies changes)."""
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
