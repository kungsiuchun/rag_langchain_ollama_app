from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from app.config import settings
from app.vectorstore import get_or_create_chroma
from app.rag_chain import build_rag_chain

app = FastAPI(title="Local RAG Demo", version="1.0")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
vs = get_or_create_chroma(
    collection_name=settings.chroma_collection,
    persist_directory=settings.chroma_persist_dir,
    embedding_function=embeddings,
)
retriever = vs.as_retriever(search_kwargs={"k": settings.retriever_k})
rag = build_rag_chain(llm, retriever)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = rag.invoke(req.question)
    return ChatResponse(answer=answer)
