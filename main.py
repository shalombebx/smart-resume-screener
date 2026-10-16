"""
Smart Résumé Screener — Entry Point
====================================
Launches the FastAPI backend with CORS, serves the single-page frontend
(frontend/index.html) and registers all routes.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api import resumes, analysis, results, health
from app.core.config import get_settings
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

_FRONTEND = Path(__file__).parent / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("🚀 Smart Résumé Screener started — API docs: /api/docs")
    yield
    logger.info("Shutting down Smart Résumé Screener")


app = FastAPI(
    title="Smart Résumé Screener",
    description="AI-powered résumé ranking using OpenAI embeddings + GPT reasoning",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)
_orig_openapi = app.openapi

def _openapi_with_binary_files():
    if app.openapi_schema:
        return app.openapi_schema
    schema = _orig_openapi()

    def fix(node):
        if isinstance(node, dict):
            if node.get("contentMediaType") == "application/octet-stream":
                node.pop("contentMediaType")
                node["format"] = "binary"
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    app.openapi_schema = schema
    return schema

app.openapi = _openapi_with_binary_files



app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(resumes.router, prefix="/api/resumes", tags=["Resumes"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])


# ── Frontend ───────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    if _FRONTEND.is_file():
        return FileResponse(_FRONTEND)
    return JSONResponse({"detail": "frontend/index.html not found. API docs at /api/docs"}, status_code=404)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )