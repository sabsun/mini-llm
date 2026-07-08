from fastapi import FastAPI
from serving.openai_routes import router as openai_router
from serving.config import ServerConfig
from serving.loader import ModelLoader
from serving.routes import router
from serving.state import AppState
from serving.models import router as models_router
app = FastAPI(
    title="MiniLlama",
    version="1.0.0",
)

config = ServerConfig()

loader = ModelLoader(config)

app.state.llm = AppState(loader)

app.include_router(router)

app.include_router(router)

app.include_router(openai_router)
app.include_router(models_router)