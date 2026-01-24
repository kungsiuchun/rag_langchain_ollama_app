from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_base_url: str = _get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = _get("OLLAMA_LLM_MODEL", "llama3.1")
    embed_model: str = _get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Chroma
    chroma_persist_dir: str = _get("CHROMA_PERSIST_DIR", "./storage/chroma")
    chroma_collection: str = _get("CHROMA_COLLECTION", "kb")

    # Retrieval
    retriever_k: int = int(_get("RETRIEVER_K", "4"))

    # Neo4j (optional)
    neo4j_uri: str | None = _get("NEO4J_URI")
    neo4j_username: str | None = _get("NEO4J_USERNAME")
    neo4j_password: str | None = _get("NEO4J_PASSWORD")


settings = Settings()
