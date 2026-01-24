from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma


def get_or_create_chroma(collection_name: str, persist_directory: str, embedding_function):
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function,
    )


def reset_collection(vs: Chroma):
    vs._collection.delete(where={})


def add_documents(vs: Chroma, docs: List[Document]):
    vs.add_documents(docs)
    vs.persist()
