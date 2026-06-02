import os
from pathlib import Path
from typing import Any, AsyncIterator

import dvc.api
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from openai import APIError, APITimeoutError, AsyncOpenAI, AuthenticationError, RateLimitError


class ChatManagerError(Exception):
    """Base exception for chat manager failures."""


class ChatConfigurationError(ChatManagerError):
    """Raised when required chat configuration is missing or invalid."""


class ChatRetrievalError(ChatManagerError):
    """Raised when retrieval fails."""


class ChatGenerationError(ChatManagerError):
    """Raised when answer generation fails."""


class ChatManager:
    """Coordinate retrieval and OpenAI-compatible streamed generation."""

    def __init__(
        self,
        prompts_dir: Path,
        index_path: str,
        collection_name: str,
        openai_api_key: str = os.environ.get("OPENAI_API_KEY", ""),
        openai_api_base: str = os.environ.get("OPENAI_API_BASE", "https://api.mistral.ai/v1"),
        llm_model_name: str = os.environ.get("OPENAI_MODEL", ""),
    ) -> None:
        """Initialize prompts, LLM client, and retriever."""
        prompt_path = prompts_dir / "answer.txt"
        if not prompt_path.exists():
            msg = f"Prompt file not found: {prompt_path}"
            raise ChatConfigurationError(msg)

        self.system_prompt = prompt_path.read_text()

        self.llm_model_name = llm_model_name
        self.llm = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        self.retriever = self.load_retriever(
            retrieved_chunks=5,
            index_path=index_path,
            collection_name=collection_name,
        )

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Return serialized chunks retrieved for a question."""
        try:
            docs = self.retriever.invoke(question)
        except Exception as exc:
            msg = "Failed to retrieve chunks for the question."
            raise ChatRetrievalError(msg) from exc

        return [serialize_doc(doc, index=index) for index, doc in enumerate(docs, start=1)]

    def load_retriever(
        self,
        retrieved_chunks: int,
        index_path: str,
        collection_name: str,
    ) -> object:
        """Load a Chroma retriever using the configured embedding model."""
        params = dvc.api.params_show()
        model_name = params["models"]["search"]["name"]
        prompts = params["models"]["search"]["prompts"]

        try:
            embedding_function = HuggingFaceEmbeddings(
                model_name=model_name,
                multi_process=True,
                show_progress=True,
                query_encode_kwargs={"normalize_embeddings": True, "prompt_name": prompts["query"]},
                encode_kwargs={"normalize_embeddings": True, "prompt_name": prompts["chunk"]},
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )

            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=index_path,
            )
        except Exception as exc:
            msg = "Failed to initialize the retriever."
            raise ChatConfigurationError(msg) from exc

        return vectorstore.as_retriever(search_kwargs={"k": retrieved_chunks})

    async def stream_answer(self, question: str) -> AsyncIterator[dict[str, Any]]:
        """Stream answer deltas and then emit retrieved chunks."""
        try:
            docs = self.retriever.invoke(question)
        except Exception as exc:
            msg = "Failed to retrieve chunks for answer generation."
            raise ChatRetrievalError(msg) from exc

        context = collate_docs(docs)

        generation_prefix = "Answer: "
        prefix_buffer = ""
        prefix_checked = False

        try:
            stream = await self.llm.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                    },
                    {"role": "assistant", "content": generation_prefix, "prefix": True},
                ],
                temperature=0,
                stream=True,
            )
        except (AuthenticationError, RateLimitError, APITimeoutError, APIError) as exc:
            msg = "Failed to start answer generation from the LLM provider."
            raise ChatGenerationError(msg) from exc

        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue

                if prefix_checked:
                    yield {"type": "answer_delta", "content": delta}
                    continue

                prefix_buffer += delta
                candidate = prefix_buffer.lstrip()

                if candidate.startswith(generation_prefix):
                    prefix_checked = True
                    remainder = candidate[len(generation_prefix) :].lstrip()
                    if remainder:
                        yield {"type": "answer_delta", "content": remainder}
                    continue

                if generation_prefix.startswith(candidate):
                    continue

                prefix_checked = True
                yield {"type": "answer_delta", "content": prefix_buffer}
        except (AuthenticationError, RateLimitError, APITimeoutError, APIError) as exc:
            msg = "Answer generation failed while streaming from the LLM provider."
            raise ChatGenerationError(msg) from exc

        if prefix_buffer and not prefix_checked:
            yield {"type": "answer_delta", "content": prefix_buffer}

        yield {
            "type": "chunks",
            "chunks": [serialize_doc(doc, index=index) for index, doc in enumerate(docs, start=1)],
        }


def collate_docs(docs: list[Document]) -> str:
    """Format retrieved documents as prompt context."""
    chunks = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("document", "unknown")
        chunks.append(f"[{index}] Source: {source}\n{doc.page_content}")

    return "\n\n".join(chunks)


def serialize_doc(doc: Document, index: int) -> dict[str, Any]:
    """Serialize a retrieved document for API responses."""
    return {
        "index": index,
        "text": doc.page_content,
        "metadata": dict(doc.metadata),
    }
