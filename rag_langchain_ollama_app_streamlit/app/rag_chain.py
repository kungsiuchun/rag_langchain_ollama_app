from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        ck = d.metadata.get("chunk", "?")
        parts.append(f"[source:{src}#{ck}]\n{d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(llm, retriever):
    system = (
        "You are an AI engineer assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. "
        "When relevant, cite sources using the [source:...] markers."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
