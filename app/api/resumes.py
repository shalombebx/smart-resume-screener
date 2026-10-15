"""
app/api/resumes.py
───────────────────
Endpoints for uploading and managing résumé files.

POST /api/resumes/upload       — Upload one or more files
GET  /api/resumes/             — List all uploaded résumés
DELETE /api/resumes/{id}       — Remove a résumé
DELETE /api/resumes/           — Clear all résumés
"""

from __future__ import annotations
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.logger import get_logger
from app.models.schemas import UploadResponse
from app.services.extractor import extractor
from app.services.parser import resume_parser
from app.services.analyzer import resume_analyzer

logger = get_logger(__name__)
router = APIRouter()

_UPLOAD_DIR = Path(settings.UPLOAD_DIR)
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _error(filename: str | None, message: str) -> UploadResponse:
    return UploadResponse(candidate_id="", filename=filename or "unknown", status="error", message=message)


@router.post(
    "/upload",
    response_model=List[UploadResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload one or more résumé files (PDF / DOCX)",
)
async def upload_resumes(files: List[UploadFile] = File(...)):
    """
    Accept PDF or DOCX résumé files, extract text, parse structured data,
    and store them in the in-memory résumé store.

    Returns a list of UploadResponse objects — one per file, in request order.
    """
    if len(files) > settings.MAX_RESUMES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)} (max {settings.MAX_RESUMES_PER_BATCH} per upload).",
        )

    responses: list[UploadResponse] = []

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()

        if suffix not in settings.ALLOWED_EXTENSIONS:
            responses.append(_error(
                file.filename, f"Unsupported file type '{suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}"))
            continue

        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            responses.append(_error(file.filename, f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit."))
            continue

        save_path = _UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"     # never use the client's filename on disk

        try:
            save_path.write_bytes(content)

            raw_text = await run_in_threadpool(extractor.extract, save_path)
            if not raw_text.strip():
                save_path.unlink(missing_ok=True)
                responses.append(_error(
                    file.filename, "Could not extract text from file. Is it scanned/image-only?"))
                continue

            parsed = resume_parser.parse(raw_text, file.filename or save_path.name)

            final_path = _UPLOAD_DIR / f"{parsed.candidate_id}{suffix}"
            save_path.rename(final_path)
            save_path = final_path                                # so cleanup below targets the right file

            resume_analyzer.add_resume(parsed)

            responses.append(UploadResponse(
                candidate_id=parsed.candidate_id,
                filename=file.filename or "unknown",
                status="success",
                message=f"Parsed successfully. Skills found: {len(parsed.skills)}",
            ))
            logger.info("Uploaded and parsed: %s (id=%s)", file.filename, parsed.candidate_id[:8])

        except Exception as exc:
            logger.exception("Failed to process file %s", file.filename)
            save_path.unlink(missing_ok=True)
            responses.append(_error(file.filename, f"Processing error: {exc}"))

    return responses


@router.get("/", response_model=List[dict], summary="List all uploaded résumés")
async def list_resumes():
    """Return a lightweight summary of all stored résumés."""
    return [
        {
            "candidate_id": r.candidate_id,
            "filename": r.filename,
            "full_name": r.full_name,
            "email": r.email,
            "skills_count": len(r.skills),
            "experience_years": r.total_experience_years,
            "education": r.highest_education.value,
            "top_skills": r.skills[:8],
        }
        for r in resume_analyzer.list_resumes()
    ]


@router.delete("/{candidate_id}", summary="Remove a résumé by ID")
async def delete_resume(candidate_id: str):
    if not resume_analyzer.remove_resume(candidate_id):
        raise HTTPException(status_code=404, detail="Résumé not found.")

    for ext in settings.ALLOWED_EXTENSIONS:
        (_UPLOAD_DIR / f"{candidate_id}{ext}").unlink(missing_ok=True)

    return {"message": f"Résumé {candidate_id[:8]} deleted."}


@router.delete("/", summary="Clear all résumés")
async def clear_all_resumes():
    count = len(resume_analyzer.list_resumes())
    resume_analyzer.clear_resumes()

    for f in _UPLOAD_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in settings.ALLOWED_EXTENSIONS:
            f.unlink()

    return {"message": f"Cleared {count} résumé(s)."}
