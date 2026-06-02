import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from models.chat_manager import ChatGenerationError, ChatManagerError, ChatRetrievalError

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


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


def chunk_summary(chunks: list[dict[str, Any]]) -> str:
    """Return a compact log summary for retrieved chunks."""
    summaries = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("document", "unknown") if isinstance(metadata, dict) else "unknown"
        source_name = Path(str(source)).name
        summaries.append(f"#{chunk.get('index', '?')} {source_name}")
    return ", ".join(summaries)


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
    logger.info("retrieve request received question=%r", question)
    try:
        chunks = chat_manager.retrieve(question)
        logger.info(
            "retrieve response returned question=%r chunk_count=%s chunks=[%s]",
            question,
            len(chunks),
            chunk_summary(chunks),
        )
        return {"question": question, "chunks": chunks}
    except ChatManagerError as exc:
        logger.exception("retrieve request failed question=%r", question)
        raise service_error_response(exc) from exc


@router.post("/chat")
async def chat(payload: RetrievalRequest, request: Request) -> StreamingResponse:
    """Stream an answer and finish with retrieved chunks as SSE events."""
    question = validate_question(payload.question)
    chat_manager = request.app.state.chat_manager
    logger.info("chat request received question=%r", question)

    async def event_stream() -> AsyncIterator[str]:
        first_event_logged = False
        try:
            async for event in chat_manager.stream_answer(question):
                event_type = event["type"]
                if not first_event_logged:
                    logger.info(
                        "chat initial response returned question=%r event=%s", question, event_type
                    )
                    first_event_logged = True
                if event_type == "chunks":
                    chunks = event.get("chunks", [])
                    logger.info(
                        "chat chunks returned question=%r chunk_count=%s chunks=[%s]",
                        question,
                        len(chunks),
                        chunk_summary(chunks),
                    )
                data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                yield f"event: {event_type}\ndata: {data}\n\n"
        except (ChatGenerationError, ChatRetrievalError) as exc:
            event = {"type": "error", "message": str(exc)}
            data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            logger.exception("chat stream failed question=%r", question)
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
