from __future__ import annotations

import argparse

from langchain_ollama import OllamaEmbeddings

from app.config import settings
from app.loaders import load_documents
from app.splitters import split_documents
from app.vectorstore import get_or_create_chroma, reset_collection, add_documents


def build_index(source: str = './knowledge_base', splitter: str = 'hybrid', reset: bool = False) -> dict:
    """Build / refresh the Chroma vector index for local docs."""
    embeddings = OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )

    docs = load_documents(source)
    if not docs:
        raise ValueError(f"No supported documents found under {source}")

    chunks = split_documents(docs, splitter=splitter, embeddings=embeddings)

    vs = get_or_create_chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings,
    )

    if reset:
        reset_collection(vs)

    add_documents(vs, chunks)

    return {
        'docs': len(docs),
        'chunks': len(chunks),
        'persist_dir': settings.chroma_persist_dir,
        'collection': settings.chroma_collection,
    }


def main():
    parser = argparse.ArgumentParser(description="Build / refresh a Chroma vector index for local docs")
    parser.add_argument("--source", default="./knowledge_base", help="Directory of documents")
    parser.add_argument(
        "--splitter",
        default="hybrid",
        choices=["hybrid", "token", "code", "markdown", "semantic"],
        help="Splitting strategy",
    )
    parser.add_argument("--reset", action="store_true", help="Delete existing vectors first")
    args = parser.parse_args()

    stats = build_index(args.source, args.splitter, args.reset)

    print(f"Indexed {stats['docs']} docs into {stats['chunks']} chunks")
    print(f"Chroma dir: {stats['persist_dir']} | collection: {stats['collection']}")


if __name__ == "__main__":
    main()
