from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

try:
    from langchain_experimental.text_splitter import SemanticChunker  # type: ignore
except Exception:  # pragma: no cover
    SemanticChunker = None


def _split_python(doc: Document, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents([doc])


def _split_markdown(doc: Document, token_chunk_size: int = 500, token_overlap: int = 50) -> List[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    md_docs = header_splitter.split_text(doc.page_content)

    out: List[Document] = []
    token_splitter = TokenTextSplitter(chunk_size=token_chunk_size, chunk_overlap=token_overlap)
    for mdd in md_docs:
        meta = dict(doc.metadata)
        meta.update(mdd.metadata)
        out.extend(token_splitter.split_documents([Document(page_content=mdd.page_content, metadata=meta)]))
    return out


def _split_tokens(doc: Document, token_chunk_size: int = 450, token_overlap: int = 50) -> List[Document]:
    splitter = TokenTextSplitter(chunk_size=token_chunk_size, chunk_overlap=token_overlap)
    return splitter.split_documents([doc])


def _split_semantic(doc: Document, embeddings) -> List[Document]:
    if SemanticChunker is None:
        return _split_tokens(doc)
    splitter = SemanticChunker(embeddings)
    return splitter.split_documents([doc])


def split_documents(docs: List[Document], splitter: str, embeddings=None) -> List[Document]:
    splitter = splitter.lower().strip()
    out: List[Document] = []

    for doc in docs:
        ext = (doc.metadata.get("ext") or "").lower()

        if splitter == "token":
            out.extend(_split_tokens(doc))
        elif splitter == "code":
            out.extend(_split_python(doc) if ext == ".py" else _split_tokens(doc))
        elif splitter == "markdown":
            out.extend(_split_markdown(doc) if ext == ".md" else _split_tokens(doc))
        elif splitter == "semantic":
            if embeddings is None:
                raise ValueError("Semantic splitter requires embeddings")
            out.extend(_split_semantic(doc, embeddings))
        elif splitter == "hybrid":
            if ext == ".py":
                out.extend(_split_python(doc))
            elif ext == ".md":
                out.extend(_split_markdown(doc))
            else:
                out.extend(_split_tokens(doc))
        else:
            raise ValueError(f"Unknown splitter: {splitter}")

    counter = {}
    for d in out:
        src = d.metadata.get("source", "")
        counter[src] = counter.get(src, 0) + 1
        d.metadata["chunk"] = counter[src]

    return out
