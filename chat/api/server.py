from pathlib import Path

from fastapi import FastAPI
from models.chat_manager import ChatManager
from chat.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI()

    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"

    app.state.chat_manager = ChatManager(
        prompts_dir=prompts_dir,
        index_path="output/co1/index",
        collection_name="c1_index",
    )

    app.include_router(api_router)

    return app


app = create_app()
