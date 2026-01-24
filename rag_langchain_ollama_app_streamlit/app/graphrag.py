from __future__ import annotations

"""Optional GraphRAG (Neo4j) demo.

- Ingest unstructured docs into Neo4j using an LLM-based graph transformer.
- Query the graph using GraphCypherQAChain.

This module only works if Neo4j env vars are set.
"""

import argparse

from langchain_ollama.llms import OllamaLLM

from app.config import settings
from app.loaders import load_documents


def _require_neo4j():
    if not (settings.neo4j_uri and settings.neo4j_username and settings.neo4j_password):
        raise SystemExit(
            "Neo4j settings not found. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env"
        )


def ingest_to_graph(source: str):
    _require_neo4j()

    from langchain_community.graphs import Neo4jGraph

    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
    except Exception as e:
        raise SystemExit("Graph transformer requires langchain-experimental") from e

    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)

    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )

    docs = load_documents(source)
    if not docs:
        raise SystemExit("No documents found to ingest")

    transformer = LLMGraphTransformer(llm=llm)
    graph_docs = transformer.convert_to_graph_documents(docs)
    graph.add_graph_documents(graph_docs, include_source=True)

    print(f"Ingested {len(docs)} docs into Neo4j as {len(graph_docs)} graph documents")


def graph_answer(question: str) -> str:
    _require_neo4j()

    from langchain_community.graphs import Neo4jGraph
    from langchain.chains import GraphCypherQAChain

    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )

    chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=False)
    result = chain.invoke({"query": question})
    return str(result.get("result", result))


def query_graph(question: str):
    print(graph_answer(question))


def main():
    parser = argparse.ArgumentParser(description="Optional GraphRAG demo (Neo4j)")
    parser.add_argument("--source", default="./knowledge_base", help="Directory of docs")
    parser.add_argument("--ingest", action="store_true", help="Ingest docs into Neo4j")
    parser.add_argument("--query", type=str, help="Ask a question via graph QA")
    args = parser.parse_args()

    if args.ingest:
        ingest_to_graph(args.source)
        return

    if args.query:
        q = args.query
        if q.lower().startswith("graph:"):
            q = q.split(":", 1)[1].strip()
        query_graph(q)
        return

    raise SystemExit("Specify --ingest or --query")


if __name__ == "__main__":
    main()
