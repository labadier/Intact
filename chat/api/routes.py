import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from models.chat_manager import ChatGenerationError, ChatManagerError, ChatRetrievalError

router = APIRouter()


class RetrievalRequest(BaseModel):
    """Request body for retrieval and chat endpoints."""

    question: str


def validate_question(question: str) -> str:
    """Normalize and validate a question string."""
    normalized = question.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="question must not be empty")
    return normalized


def service_error_response(exc: ChatManagerError) -> HTTPException:
    """Map chat service errors to API errors."""
    status_code = 503 if isinstance(exc, ChatRetrievalError) else 500
    return HTTPException(status_code=status_code, detail=str(exc))


@router.get("/")
async def index() -> FileResponse:
    """Serve the chat UI."""
    ui_path = Path(__file__).resolve().parents[1] / "ui" / "index.html"
    return FileResponse(ui_path)


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint to verify the server's status."""
    return {"status": "healthy"}


@router.post("/retrieve")
async def retrieve(payload: RetrievalRequest, request: Request) -> dict[str, Any]:
    """Return retrieved chunks for a question."""
    question = validate_question(payload.question)
    chat_manager = request.app.state.chat_manager
    try:
        return {"question": question, "chunks": chat_manager.retrieve(question)}
    except ChatManagerError as exc:
        raise service_error_response(exc) from exc


@router.post("/chat")
async def chat(payload: RetrievalRequest, request: Request) -> StreamingResponse:
    """Stream an answer and finish with retrieved chunks as SSE events."""
    question = validate_question(payload.question)
    chat_manager = request.app.state.chat_manager

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in chat_manager.stream_answer(question):
                data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                yield f"event: {event['type']}\ndata: {data}\n\n"
        except (ChatGenerationError, ChatRetrievalError) as exc:
            event = {"type": "error", "message": str(exc)}
            data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
