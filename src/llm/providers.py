"""LangChain wrappers for our three LLM providers with automatic fallback.

Provider selection comes from config.yaml (`llm.provider`). On API failure
(rate limit, timeout, etc.) LangChain transparently retries against the
configured fallback providers — currently Gemini, with Ollama optional.
"""

from __future__ import annotations

from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from src.config import settings


ProviderName = Literal["groq", "gemini", "ollama"]


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
    """Primary LLM with automatic fallback to configured alternates."""
    primary = get_llm(settings.llm.provider)
    fallbacks = [_build_provider(p) for p in settings.llm.fallback_providers]
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
