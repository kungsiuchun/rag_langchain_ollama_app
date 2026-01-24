from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

SUPPORTED_EXTS = {".md", ".txt", ".py"}


def load_documents(source_dir: str) -> List[Document]:
    """Load local files from a directory into LangChain Documents."""
    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")

    docs: List[Document] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue

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
                    "ext": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                },
            )
        )

    docs.sort(key=lambda d: d.metadata.get("source", ""))
    return docs
