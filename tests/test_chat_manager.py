from typing import AsyncIterator

import pytest
from langchain_core.documents import Document

from models.chat_manager import ChatManager, collate_docs, serialize_doc


class FakeRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.questions: list[str] = []

    def invoke(self, question: str) -> list[Document]:
        self.questions.append(question)
        return self.docs


class FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.delta = FakeDelta(content)


class FakeStreamChunk:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeStream:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[FakeStreamChunk]:
        for chunk in self.chunks:
            yield FakeStreamChunk(chunk)


class FakeCompletions:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self.chunks)


class FakeChat:
    def __init__(self, chunks: list[str]) -> None:
        self.completions = FakeCompletions(chunks)


class FakeLLM:
    def __init__(self, chunks: list[str]) -> None:
        self.chat = FakeChat(chunks)


def test_serialize_doc_includes_index_text_and_metadata():
    doc = Document(page_content="Audit evidence", metadata={"document": "policy.pdf"})

    serialized = serialize_doc(doc, index=2)

    assert serialized == {
        "index": 2,
        "text": "Audit evidence",
        "metadata": {"document": "policy.pdf"},
    }


def test_collate_docs_formats_context_with_sources():
    docs = [
        Document(page_content="First chunk", metadata={"document": "a.pdf"}),
        Document(page_content="Second chunk", metadata={}),
    ]

    context = collate_docs(docs)

    assert "[1] Source: a.pdf\nFirst chunk" in context
    assert "[2] Source: unknown\nSecond chunk" in context


@pytest.mark.anyio
async def test_stream_answer_retrieves_generates_and_emits_chunks():
    docs = [
        Document(page_content="First evidence", metadata={"document": "a.pdf"}),
        Document(page_content="Second evidence", metadata={"document": "b.pdf"}),
    ]
    manager = ChatManager.__new__(ChatManager)
    manager.system_prompt = "System instructions"
    manager.llm_model_name = "test-model"
    manager.retriever = FakeRetriever(docs)
    manager.llm = FakeLLM(["Answer: ", "Supported claim <doc1>", " and more <doc2>"])

    events = [event async for event in manager.stream_answer("What is supported?")]

    assert manager.retriever.questions == ["What is supported?"]
    assert events == [
        {"type": "answer_delta", "content": "Supported claim <doc1>"},
        {"type": "answer_delta", "content": " and more <doc2>"},
        {
            "type": "chunks",
            "chunks": [
                {"index": 1, "text": "First evidence", "metadata": {"document": "a.pdf"}},
                {"index": 2, "text": "Second evidence", "metadata": {"document": "b.pdf"}},
            ],
        },
    ]

    call = manager.llm.chat.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0
    assert call["stream"] is True
    assert call["messages"] == [
        {"role": "system", "content": "System instructions"},
        {
            "role": "user",
            "content": (
                "Context:\n"
                "[1] Source: a.pdf\nFirst evidence\n\n"
                "[2] Source: b.pdf\nSecond evidence\n\n"
                "Question:\nWhat is supported?"
            ),
        },
        {"role": "assistant", "content": "Answer: ", "prefix": True},
    ]
