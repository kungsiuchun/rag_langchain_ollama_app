from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from app.config import settings
from app.vectorstore import get_or_create_chroma
from app.rag_chain import build_rag_chain


console = Console()


def main():
    console.print("[bold]Local RAG (LangChain + Ollama)[/bold]")
    console.print(f"LLM: {settings.llm_model} | Embeddings: {settings.embed_model}")
    console.print("Type 'exit' to quit.\n")

    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
    embeddings = OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)

    vs = get_or_create_chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=embeddings,
    )
    retriever = vs.as_retriever(search_kwargs={"k": settings.retriever_k})

    chain = build_rag_chain(llm, retriever)

    while True:
        q = Prompt.ask("[cyan]You[/cyan]")
        if q.strip().lower() in {"exit", "quit"}:
            break

        try:
            a = chain.invoke(q)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            continue

        console.print("[green]Assistant[/green]:")
        console.print(a)
        console.print()


if __name__ == "__main__":
    main()
