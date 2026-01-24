from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from app.config import settings
from app.vectorstore import get_or_create_chroma
from app.rag_chain import build_rag_chain


def load_jsonl(path: str):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG with Ragas")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    args = parser.parse_args()

    rows = load_jsonl(args.dataset)

    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
    embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)

    vs = get_or_create_chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings,
    )
    retriever = vs.as_retriever(search_kwargs={"k": settings.retriever_k})
    rag = build_rag_chain(llm, retriever)

    prepared = []
    for r in rows:
        q = r["user_input"]
        if "retrieved_contexts" not in r:
            docs = retriever.invoke(q)
            r["retrieved_contexts"] = [d.page_content for d in docs]
        if "response" not in r:
            r["response"] = rag.invoke(q)
        prepared.append(r)

    dataset = EvaluationDataset.from_list(prepared)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\nRagas scores (higher is better):")
    print(result)


if __name__ == "__main__":
    main()
