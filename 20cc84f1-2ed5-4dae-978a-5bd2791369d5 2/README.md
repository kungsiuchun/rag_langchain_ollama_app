# LangChain + Ollama (llama3.1) RAG / Vector / Embeddings / GraphRAG Demo

This repo is a **hands-on RAG application** that showcases the core topics an AI engineer uses in production:

- **RAG** (retrieval-augmented generation) over your local docs
- **Embeddings + Vector store** (Ollama embeddings + Chroma)
- **Multiple splitting strategies**
  - code-aware splitting for **Python**
  - Markdown header-aware splitting
  - **token-based splitting**
  - optional **semantic splitting**
- Optional **GraphRAG** (Neo4j knowledge graph + Cypher QA)
- Optional **evaluation** with **Ragas** and tracing with **LangSmith**

> LLM runtime: **Ollama** with **llama3.1**.

---

## 0) Prereqs

1. Install Ollama and start it.
2. Pull the models you want:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
# (optional) alternative embeddings:
# ollama pull mxbai-embed-large
```

---

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy env template:

```bash
cp .env.example .env
```

---

## 2) Add knowledge

Put documents into `knowledge_base/`.

This repo includes a small sample (`knowledge_base/course_description.md`) so you can run immediately.

---

## 3) Build the vector index (Chroma)

```bash
python -m app.build_index --source ./knowledge_base --splitter hybrid
```

Splitters:
- `hybrid` (recommended): code-aware for .py, header-aware for .md, token-based for others
- `token`: token-based splitting
- `code`: code-aware splitting for Python
- `semantic`: embedding-based semantic chunking (requires `langchain-experimental`)

---

## 4) Ask questions (CLI)

```bash
python -m app.cli
```

Example queries:
- `What is RAG and why does it help LLMs?`
- `List the splitting strategies used in this app.`

---

## 5) Run the API (FastAPI)

```bash
uvicorn app.api:app --reload --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does this demo implement?"}'
```

---

## 6) Optional: GraphRAG with Neo4j

Set Neo4j vars in `.env`:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

Ingest docs into graph:

```bash
python -m app.graphrag --ingest --source ./knowledge_base
```

Ask via graph QA:

```bash
python -m app.graphrag --query "graph: summarize entities and relationships in the course description"
```

---

## 7) Optional: Evaluate with Ragas (+ LangSmith tracing)

Create a small dataset file (JSONL) in `eval/eval_dataset.jsonl` and run:

```bash
python -m app.eval_ragas --dataset eval/eval_dataset.jsonl
```

If you want LangSmith tracing, set:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=...  # LangSmith
export LANGCHAIN_PROJECT=rag-demo
```

---

## Project layout

```text
app/
  api.py            # FastAPI server
  build_index.py    # indexing pipeline (load -> split -> embed -> store)
  cli.py            # interactive chat
  config.py         # environment/config
  loaders.py        # read local files into Documents
  rag_chain.py      # retrieval + generation chain
  splitters.py      # splitter strategies
  vectorstore.py    # Chroma persistence helpers
  graphrag.py       # optional GraphRAG (Neo4j)
  eval_ragas.py     # optional Ragas evaluation
knowledge_base/
  course_description.md
storage/
  chroma/           # persisted vectors
```

---

## Notes

- If you change documents, re-run `build_index`.
- This app is designed for **local-first** privacy: everything runs on your machine.
