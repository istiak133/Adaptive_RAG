"""LangChain wrappers for our three LLM providers with automatic fallback.

Provider selection comes from config.yaml (`llm.provider`). On API failure
(rate limit, timeout, etc.) LangChain transparently retries against the
configured fallback providers — currently Gemini, with Ollama optional.

Graceful key handling: a reviewer who only supplies one of the two free-tier
keys still gets a working system. Fallback providers whose keys are missing
are skipped with a one-line warning, not a hard crash.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from src.config import settings

logger = logging.getLogger(__name__)


ProviderName = Literal["groq", "gemini", "ollama"]


def _provider_available(provider: ProviderName) -> bool:
    """True if the provider can be built (keys / config present)."""
    if provider == "groq":
        return bool(settings.secrets.groq_api_key)
    if provider == "gemini":
        return bool(settings.secrets.google_api_key)
    if provider == "ollama":
        return True  # local — no key, just need the daemon running
    return False


def _build_provider(provider: ProviderName) -> BaseChatModel:
    if provider == "groq":
        if not settings.secrets.groq_api_key:
            raise RuntimeError("GROQ_API_KEY missing — set it in .env")
        return ChatGroq(
            model=settings.llm.model.groq,
            api_key=settings.secrets.groq_api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_output_tokens,
            timeout=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        )
    if provider == "gemini":
        if not settings.secrets.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY missing — set it in .env")
        return ChatGoogleGenerativeAI(
            model=settings.llm.model.gemini,
            google_api_key=settings.secrets.google_api_key,
            temperature=settings.llm.temperature,
            max_output_tokens=settings.llm.max_output_tokens,
            timeout=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        )
    if provider == "ollama":
        base_url = settings.secrets.ollama_base_url or "http://localhost:11434"
        return ChatOllama(
            model=settings.llm.model.ollama,
            base_url=base_url,
            temperature=settings.llm.temperature,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def get_llm(provider: Optional[ProviderName] = None) -> BaseChatModel:
    """Return a single-provider LLM (no fallback)."""
    return _build_provider(provider or settings.llm.provider)


def get_llm_with_fallback() -> BaseChatModel:
    """Primary LLM with automatic fallback to configured alternates.

    Fallback providers whose credentials are unavailable are silently
    skipped (with a warning log) rather than crashing the build. This lets
    a reviewer supply only one of the two free-tier keys and still run
    everything end-to-end.
    """
    primary = get_llm(settings.llm.provider)

    fallbacks: list[BaseChatModel] = []
    for name in settings.llm.fallback_providers:
        if not _provider_available(name):
            logger.warning(
                "Fallback provider %r unavailable (no credentials) — skipping.",
                name,
            )
            continue
        fallbacks.append(_build_provider(name))

    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary


if __name__ == "__main__":
    print(f"Primary provider:   {settings.llm.provider}")
    print(f"Fallback providers: {settings.llm.fallback_providers}")
    print()

    llm = get_llm_with_fallback()
    print(f"Primary LLM class: {type(llm.runnable).__name__}")

    print()
    print("Test query: 'Reply with just the word OK and nothing else.'")
    response = llm.invoke("Reply with just the word OK and nothing else.")
    print(f"Response: {response.content!r}")
    print(f"Type:     {type(response).__name__}")
