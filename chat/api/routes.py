from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class RetrievalRequest(BaseModel):
    question: str


@router.get("/health")
async def health():
    """Health check endpoint to verify the server's status."""

    return {"status": "healthy"}


@router.post("/retrieve")
async def retrieve(payload: RetrievalRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    chat_manager = request.app.state.chat_manager
    return {"question": question, "chunks": chat_manager.retrieve(question)}


@router.post("/chat")
async def chat(payload: RetrievalRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    chat_manager = request.app.state.chat_manager
    return StreamingResponse(
        chat_manager.stream_answer(question),
        media_type="text/plain",
    )
