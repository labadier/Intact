from typing import AsyncIterator, Protocol

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chat.api.routes import router
from models.chat_manager import ChatGenerationError, ChatRetrievalError


class ChatManagerProtocol(Protocol):
    def retrieve(self, question: str) -> list[dict[str, object]]: ...

    async def stream_answer(self, question: str) -> AsyncIterator[dict[str, object]]: ...


class FakeChatManager:
    def retrieve(self, question: str) -> list[dict[str, object]]:
        return [
            {
                "index": 1,
                "text": f"Evidence for {question}",
                "metadata": {"document": "source.pdf"},
            }
        ]

    async def stream_answer(self, question: str) -> AsyncIterator[dict[str, object]]:
        yield {"type": "answer_delta", "content": f"Answer for {question}"}
        yield {"type": "chunks", "chunks": self.retrieve(question)}


class FailingRetrievalManager(FakeChatManager):
    def retrieve(self, question: str) -> list[dict[str, object]]:
        raise ChatRetrievalError("Retrieval backend unavailable.")


class FailingStreamManager(FakeChatManager):
    async def stream_answer(self, question: str) -> AsyncIterator[dict[str, object]]:
        yield {"type": "answer_delta", "content": "partial"}
        raise ChatGenerationError("LLM stream failed.")


def make_client(chat_manager: ChatManagerProtocol | None = None) -> TestClient:
    app = FastAPI()
    app.state.chat_manager = chat_manager or FakeChatManager()
    app.include_router(router)
    return TestClient(app)


def test_health_endpoint():
    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_retrieve_rejects_empty_question():
    client = make_client()

    response = client.post("/retrieve", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "question must not be empty"


def test_retrieve_returns_chunks():
    client = make_client()

    response = client.post("/retrieve", json={"question": "policy"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "policy",
        "chunks": [
            {
                "index": 1,
                "text": "Evidence for policy",
                "metadata": {"document": "source.pdf"},
            }
        ],
    }


def test_retrieve_maps_retrieval_errors_to_service_unavailable():
    client = make_client(FailingRetrievalManager())

    response = client.post("/retrieve", json={"question": "policy"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Retrieval backend unavailable."


def test_chat_streams_answer_and_chunks_as_sse():
    client = make_client()

    with client.stream("POST", "/chat", json={"question": "policy"}) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert (
        'event: answer_delta\ndata: {"type":"answer_delta","content":"Answer for policy"}' in body
    )
    assert 'event: chunks\ndata: {"type":"chunks","chunks":' in body


def test_chat_streams_error_event_when_generation_fails():
    client = make_client(FailingStreamManager())

    with client.stream("POST", "/chat", json={"question": "policy"}) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert 'event: answer_delta\ndata: {"type":"answer_delta","content":"partial"}' in body
    assert 'event: error\ndata: {"type":"error","message":"LLM stream failed."}' in body
