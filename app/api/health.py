"""
app/api/health.py
──────────────────
GET /api/health  — System health check
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        openai_ready=bool(settings.OPENAI_API_KEY),
    )
