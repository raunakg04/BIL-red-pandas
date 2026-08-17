"""Provider-aware OpenAI/Azure OpenAI client builders for RAG modules."""

from __future__ import annotations

import os

from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)

DEFAULT_AZURE_API_VERSION = "2024-02-01"


def _is_azure_provider() -> bool:
    """Detect whether Azure OpenAI should be used."""
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_KEY"))


def _get_azure_api_key() -> str:
    """Read Azure API key with fallback to OPENAI_API_KEY for convenience."""
    return os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")


def validate_provider_environment() -> None:
    """Validate required credentials for the selected provider."""
    if _is_azure_provider():
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = _get_azure_api_key()
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI.")
        if not api_key:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY (or OPENAI_API_KEY fallback) is required for Azure OpenAI."
            )
        return

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env.")
    if not api_key.startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY appears malformed for OpenAI provider. It should start with 'sk-'. "
            "If you are using Azure OpenAI, set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
        )


def create_embeddings(default_model: str):
    """Create embeddings client for either OpenAI or Azure OpenAI."""
    if _is_azure_provider():
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = _get_azure_api_key()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION)
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv(
            "RAG_EMBEDDING_MODEL", default_model
        )
        return AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=api_key,
            openai_api_version=api_version,
            azure_deployment=deployment,
            model=deployment,
        )

    model = os.getenv("RAG_EMBEDDING_MODEL", default_model)
    return OpenAIEmbeddings(model=model)


def create_chat_llm(default_model: str, temperature: float = 0, max_tokens: int | None = None):
    """Create chat model client for either OpenAI or Azure OpenAI.

    `max_tokens`, when given, bounds the completion length - useful for
    tasks (like the suggestion-agent's explanation text) that are always a
    short, fixed-shape answer, so an unbounded response can't quietly
    inflate cost.
    """
    if _is_azure_provider():
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_key = _get_azure_api_key()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION)
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv(
            "RAG_LLM_MODEL", default_model
        )
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    model = os.getenv("RAG_LLM_MODEL", default_model)
    return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
