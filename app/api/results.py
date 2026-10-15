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
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import AnalysisSession
from app.services.analyzer import resume_analyzer
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/{session_id}", response_model=AnalysisSession, summary="Retrieve a completed analysis session")
async def get_results(session_id: str):
    session = resume_analyzer.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


@router.get("/{session_id}/export", summary="Export results as CSV or JSON")
async def export_results(session_id: str, format: str = "csv"):
    session = resume_analyzer.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    fmt = format.lower()
    if fmt == "csv":
        return _export_csv(session)
    if fmt == "json":
        return _export_json(session)
    raise HTTPException(status_code=400, detail="Unsupported format. Use 'csv' or 'json'.")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_cell(value) -> str:
    """Neutralise spreadsheet formula injection — names/summaries come from untrusted résumés."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


# ── CSV export ─────────────────────────────────────────────────────────────────

def _export_csv(session: AnalysisSession) -> StreamingResponse:
    output = io.StringIO()
    output.write("\ufeff")                       # BOM so Excel reads UTF-8 (é, ü …) correctly
    writer = csv.writer(output)

    writer.writerow([
        "Rank", "Name", "Email", "Filename",
        "Final Score (%)", "Embedding Similarity (%)", "Keyword Match (%)",
        "Experience Alignment (%)", "Education Match (%)", "Bonus Skills (%)",
        "Fit Level", "Matching Skills", "Missing Skills", "AI Summary", "Recommendation",
    ])

    for c in session.candidates:
        ai = c.ai_explanation
        writer.writerow([
            c.rank,
            _safe_cell(c.full_name),
            _safe_cell(c.email),
            _safe_cell(c.filename),
            f"{c.score.final_score:.1f}",
            f"{c.score.embedding_similarity:.1f}",
            f"{c.score.keyword_match:.1f}",
            f"{c.score.experience_alignment:.1f}",
            f"{c.score.education_match:.1f}",
            f"{c.score.bonus_skills:.1f}",
            ai.fit_level.value if ai else "",
            _safe_cell("; ".join(c.matching_skills)),
            _safe_cell("; ".join(c.missing_skills)),
            _safe_cell(ai.summary if ai else ""),
            _safe_cell(ai.recommendation if ai else ""),
        ])

    filename = f"screening_results_{_stamp()}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── JSON export ────────────────────────────────────────────────────────────────

def _export_json(session: AnalysisSession) -> StreamingResponse:
    data = session.model_dump(mode="json")
    filename = f"screening_results_{_stamp()}.json"
    return StreamingResponse(
        iter([json.dumps(data, indent=2, ensure_ascii=False, default=str)]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
