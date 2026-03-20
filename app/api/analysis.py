"""
app/api/analysis.py
────────────────────
POST /api/analysis/run   — Run full AI screening pipeline
"""

from fastapi import APIRouter, HTTPException, status

from app.models.schemas  import AnalyzeRequest, AnalysisSession
from app.services.analyzer import resume_analyzer
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/run",
    response_model=AnalysisSession,
    status_code=status.HTTP_200_OK,
    summary="Run the AI résumé screening pipeline",
)
async def run_analysis(request: AnalyzeRequest):
    """
    Execute the full screening pipeline:
    1. Compute OpenAI embeddings for JD and all résumés
    2. Run hybrid scoring (embedding + keyword + experience + education + bonus)
    3. Generate GPT-4 explanations per candidate
    4. Return ranked results with full breakdowns

    - `job_description`: The full job description text (min 50 chars)
    - `candidate_ids`: Optional list of IDs to analyse; omit to analyse all uploaded résumés
    """
    resumes_available = resume_analyzer.list_resumes()
    if not resumes_available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No résumés uploaded. Please upload at least one résumé before running analysis.",
        )

    try:
        session = resume_analyzer.analyze(
            job_description=request.job_description,
            candidate_ids=request.candidate_ids,
        )
        return session

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.exception("Analysis runtime error")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(exc)}",
        )
