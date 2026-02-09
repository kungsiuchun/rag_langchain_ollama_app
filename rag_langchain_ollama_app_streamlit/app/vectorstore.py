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
    # 1) delete collection
    vs.delete_collection()

    # 2) re-create / initialize collection on same instance
    getattr(vs, "_Chroma__ensure_collection")()



def add_documents(vs: Chroma, docs: List[Document]):
    vs.add_documents(docs)
