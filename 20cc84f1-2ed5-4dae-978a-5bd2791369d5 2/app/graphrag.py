from __future__ import annotations

"""Optional GraphRAG implementation.

This module is intentionally optional: it only runs if Neo4j config is provided.

It demonstrates:
- LLM-based conversion of unstructured text -> graph documents
- Storing the graph in Neo4j
- Querying via Cypher QA chain

Run:
  python -m app.graphrag --ingest --source ./knowledge_base
  python -m app.graphrag --query "graph: ..."
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

    # Neo4j integration
    from langchain_community.graphs import Neo4jGraph

    # Graph transformer (experimental)
    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
    except Exception as e:
        raise SystemExit(
            "Graph transformer requires langchain-experimental. Install it (already in requirements.txt)"
        ) from e

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


def query_graph(question: str):
    _require_neo4j()

    from langchain_community.graphs import Neo4jGraph
    from langchain.chains import GraphCypherQAChain

    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )

    chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True)
    result = chain.invoke({"query": question})
    # GraphCypherQAChain returns dict with 'result'
    print(result.get("result", result))


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
