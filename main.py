"""
Smart Résumé Screener — Production Entry Point
================================================
Launches the FastAPI backend with CORS, static file serving,
and all route registrations.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api import resumes, analysis, results, health
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Application factory ────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Résumé Screener",
    description="AI-powered résumé ranking using OpenAI embeddings + GPT reasoning",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router,   prefix="/api",          tags=["Health"])
app.include_router(resumes.router,  prefix="/api/resumes",  tags=["Resumes"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(results.router,  prefix="/api/results",  tags=["Results"])

# ── Frontend static files (built React app) ────────────────────────────────────
_frontend_dist = Path(__file__).parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        return FileResponse(str(_frontend_dist / "index.html"))


# ── Startup / shutdown events ─────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info("🚀 Smart Résumé Screener started — API docs: /api/docs")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down Smart Résumé Screener")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
