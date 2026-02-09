from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader

SUPPORTED_EXTS = {".md", ".txt", ".py", ".csv"}


def load_documents(source_dir: str) -> List[Document]:
    """Load local files from a directory into LangChain Documents.

    Notes:
      - For CSV files, we load one Document per row (row = atomic chunk), which works
        best for structured data with many short rows.
    """
    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")

    docs: List[Document] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue

        # CSV: one row -> one Document
        if ext == ".csv":
            rel = str(path.relative_to(root))
            loader = CSVLoader(file_path=str(path), csv_args={"delimiter": ","})
            row_docs = loader.load()
            for idx, d in enumerate(row_docs):
                # Keep existing loader metadata (like source) but add our common fields.
                # Use absolute path as `source` for compatibility with downstream logic.
                d.metadata["source"] = str(path)
                d.metadata["rel_source"] = rel
                d.metadata["abs_path"] = str(path)
                d.metadata["ext"] = ext
                d.metadata["size_bytes"] = path.stat().st_size
                d.metadata["row"] = idx
            docs.extend(row_docs)
            continue

        # Text-like files
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")

        rel = str(path.relative_to(root))
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": rel,
                    "abs_path": str(path),
                    "ext": ext,
                    "size_bytes": path.stat().st_size,
                },
            )
        )

    docs.sort(key=lambda d: d.metadata.get("source", ""))
    return docs
