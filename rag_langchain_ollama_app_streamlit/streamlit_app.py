from __future__ import annotations

import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from app.config import settings
from app.vectorstore import get_or_create_chroma
from app.rag_chain import build_rag_chain
from app.build_index import build_index

# Optional GraphRAG
try:
    from app.graphrag import graph_answer, ingest_to_graph
except Exception:
    graph_answer = None
    ingest_to_graph = None


st.set_page_config(page_title="Local RAG + GraphRAG (Ollama)", layout="wide")
st.title("🧠 Local RAG + GraphRAG (LangChain + Ollama)")


@st.cache_resource
def init_vector():
    """Initialize LLM, embeddings, vectorstore, retriever, and RAG chain once per session."""
    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
    embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
    vs = get_or_create_chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings,
    )
    retriever = vs.as_retriever(search_kwargs={"k": settings.retriever_k})
    chain = build_rag_chain(llm, retriever)
    return llm, embeddings, vs, retriever, chain


def ensure_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Vector RAG"


ensure_state()

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption(f"LLM: {settings.llm_model} | Embeddings: {settings.embed_model}")

    mode = st.radio("Mode", ["Vector RAG", "GraphRAG (Neo4j)"], index=0)
    st.session_state.mode = mode

    st.divider()
    st.subheader("Indexing")
    source_dir = st.text_input("Knowledge base folder", value="./knowledge_base")
    splitter = st.selectbox(
        "Splitter",
        ["hybrid", "token", "code", "markdown", "semantic"],
        index=0,
        help="hybrid = code-aware for .py, header-aware for .md, token-based for others",
    )
    reset = st.checkbox("Reset vector index before indexing", value=False)

    if st.button("Build / Refresh Vector Index", use_container_width=True):
        with st.status("Indexing documents...", expanded=True):
            stats = build_index(source=source_dir, splitter=splitter, reset=reset)
            st.write(stats)
            st.success("Index built")
        init_vector.clear()

    if mode.startswith("GraphRAG"):
        st.divider()
        st.subheader("GraphRAG")
        if st.button("Ingest docs into Neo4j graph", use_container_width=True):
            if ingest_to_graph is None:
                st.error("GraphRAG module unavailable")
            else:
                with st.status("Ingesting into Neo4j...", expanded=True):
                    ingest_to_graph(source_dir)
                    st.success("Graph ingestion complete")


# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Ask a question about your knowledge base")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.mode == "Vector RAG":
        llm, embeddings, vs, retriever, chain = init_vector()

        with st.chat_message("assistant"):
            with st.status("Retrieving + generating...", expanded=False):
                docs = retriever.invoke(prompt)
                answer = chain.invoke(prompt)
                st.markdown(answer)

            with st.expander("Retrieved chunks", expanded=False):
                if not docs:
                    st.write("No documents retrieved.")
                for i, d in enumerate(docs, start=1):
                    src = d.metadata.get("source", "unknown")
                    ck = d.metadata.get("chunk", "?")
                    st.markdown(f"**{i}. {src}#{ck}**")
                    st.code(d.page_content)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        with st.chat_message("assistant"):
            if graph_answer is None:
                answer = "GraphRAG is not available in this environment."
                st.error(answer)
            else:
                try:
                    with st.status("Querying Neo4j graph...", expanded=False):
                        answer = graph_answer(prompt)
                    st.markdown(answer)
                except Exception as e:
                    answer = f"GraphRAG error: {e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
