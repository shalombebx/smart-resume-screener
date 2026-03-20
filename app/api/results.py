"""
app/api/results.py
───────────────────
GET  /api/results/{session_id}         — Retrieve analysis session
GET  /api/results/{session_id}/export  — Download CSV / JSON export
"""

from __future__ import annotations
import csv
import io
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.models.schemas import AnalysisSession, ExportRequest
from app.services.analyzer import resume_analyzer
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_EXPORT_DIR = Path(settings.EXPORT_DIR)
_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@router.get(
    "/{session_id}",
    response_model=AnalysisSession,
    summary="Retrieve a completed analysis session",
)
async def get_results(session_id: str):
    session = resume_analyzer.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


@router.get(
    "/{session_id}/export",
    summary="Export results as CSV or JSON",
)
async def export_results(session_id: str, format: str = "csv"):
    session = resume_analyzer.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    if format.lower() == "csv":
        return _export_csv(session)
    elif format.lower() == "json":
        return _export_json(session)
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'.")


# ── CSV export ─────────────────────────────────────────────────────────────────

def _export_csv(session: AnalysisSession) -> StreamingResponse:
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "Rank",
        "Name",
        "Email",
        "Filename",
        "Final Score (%)",
        "Embedding Similarity (%)",
        "Keyword Match (%)",
        "Experience Alignment (%)",
        "Education Match (%)",
        "Bonus Skills (%)",
        "Fit Level",
        "Matching Skills",
        "Missing Skills",
        "AI Summary",
        "Recommendation",
    ])

    for c in session.candidates:
        fit   = c.ai_explanation.fit_level.value if c.ai_explanation else ""
        summ  = c.ai_explanation.summary if c.ai_explanation else ""
        recom = c.ai_explanation.recommendation if c.ai_explanation else ""

        writer.writerow([
            c.rank,
            c.full_name or "",
            c.email or "",
            c.filename,
            f"{c.score.final_score:.1f}",
            f"{c.score.embedding_similarity:.1f}",
            f"{c.score.keyword_match:.1f}",
            f"{c.score.experience_alignment:.1f}",
            f"{c.score.education_match:.1f}",
            f"{c.score.bonus_skills:.1f}",
            fit,
            "; ".join(c.matching_skills),
            "; ".join(c.missing_skills),
            summ,
            recom,
        ])

    output.seek(0)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename  = f"screening_results_{timestamp}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── JSON export ────────────────────────────────────────────────────────────────

def _export_json(session: AnalysisSession) -> StreamingResponse:
    data = session.model_dump(mode="json")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename  = f"screening_results_{timestamp}.json"

    return StreamingResponse(
        iter([json.dumps(data, indent=2, default=str)]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
