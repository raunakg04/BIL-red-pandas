"""Retrieval utilities for company-reference FAISS index."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Workaround for macOS OpenMP runtime duplication issues when loading FAISS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

from retrieval.provider import create_embeddings

load_dotenv(override=True)

DEFAULT_INDEX_DIR = Path("retrieval/vector_store/company_reference_faiss")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5


@lru_cache(maxsize=1)
def _load_vector_store(index_dir: str, embedding_model: str) -> FAISS:
    """Load FAISS index once per process for faster repeated lookups."""
    embeddings = create_embeddings(default_model=embedding_model)
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_company_context(company_name: str, k: int = DEFAULT_TOP_K) -> str:
    """Return top-k retrieved chunks as a single context string for a company.

    Cached per (company_name, k): the same company can appear across many
    failing loans in a portfolio, and without caching each one would re-embed
    the same query and re-run FAISS search from scratch - wasted embedding
    calls and latency for an answer that's identical every time.
    """
    return _get_company_context_cached(company_name, k)


@lru_cache(maxsize=256)
def _get_company_context_cached(company_name: str, k: int) -> str:
    index_dir = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_dir}. Run retrieval.ingest first."
        )

    vector_store = _load_vector_store(str(index_dir), embedding_model)

    query = (
        f"Company profile for {company_name}. Include headquarters, pledgeable assets, "
        "asset values, asset currencies, and related entities."
    )
    docs = vector_store.similarity_search(query, k=k)

    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        context_blocks.append(
            f"[Chunk {i} | source={source} | page={page}]\n{doc.page_content.strip()}"
        )

    return "\n\n".join(context_blocks)
