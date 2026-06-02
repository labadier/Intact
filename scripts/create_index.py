import json
import shutil
from pathlib import Path
from typing import Any

import dvc.api
import torch
from fire import Fire
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load extracted document chunks from JSON."""
    if not chunks_path.exists():
        msg = f"Chunks file not found: {chunks_path}"
        raise FileNotFoundError(msg)

    with chunks_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_index(
    chunks_path: str,
    output_path: str,
    collection_name: str = "friend_index",
    recreate: bool = True,
) -> str:
    """Create a text-only Chroma index from a chunks.json file."""
    chunks_path = Path(chunks_path)
    output_path = Path(output_path)
    if output_path.exists() and not output_path.is_dir():
        msg = f"Output path must be a directory: {output_path}"
        raise NotADirectoryError(msg)

    output_path.mkdir(parents=True, exist_ok=True)

    if recreate and any(output_path.iterdir()):
        shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_path)
    documents = [
        Document(page_content=chunk["text"], metadata={"document": chunk.get("document")})
        for chunk in chunks
        if chunk.get("text")
    ]

    params = dvc.api.params_show()

    embedding_function = HuggingFaceEmbeddings(
        model_name=params["models"]["search"]["name"],
        multi_process=True,
        show_progress=True,
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt_name": params["models"]["search"]["prompts"]["query"],
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "prompt_name": params["models"]["search"]["prompts"]["chunk"],
        },
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(output_path),
    )

    if documents:
        vectorstore.add_documents(documents)

    print(f"Indexed {len(documents)} chunks into {output_path}")
    return str(output_path)


if __name__ == "__main__":
    Fire(build_index)
