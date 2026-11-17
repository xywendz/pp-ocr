from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .routers import analyze, health
from .services.analysis_service import AnalysisService
from .services.merge_service import MergeService
from .services.model_manager import ModelManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.uvicorn_log_level().upper())

    manager = ModelManager(settings)
    app.state.settings = settings
    app.state.models = manager
    app.state.analysis = AnalysisService(settings, manager, MergeService())

    try:
        manager.load()
    except Exception as exc:
        logger.error("模型加载失败，服务将以不可用状态启动: %s", exc)

    yield


app = FastAPI(title="PP-OCR API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理异常: %s", exc)
    return JSONResponse(status_code=500, content={"detail": f"服务内部错误: {exc}"})
