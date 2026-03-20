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
import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logger import get_logger
from app.models.schemas import ParsedResume, UploadResponse
from app.services.extractor import extractor
from app.services.parser    import resume_parser
from app.services.analyzer  import resume_analyzer

logger = get_logger(__name__)
router = APIRouter()

_UPLOAD_DIR = Path(settings.UPLOAD_DIR)
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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

    Returns a list of UploadResponse objects — one per file.
    """
    responses: list[UploadResponse] = []

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()

        # ── Validate extension ─────────────────────────────────────────────
        if suffix not in settings.ALLOWED_EXTENSIONS:
            responses.append(UploadResponse(
                candidate_id="",
                filename=file.filename or "unknown",
                status="error",
                message=f"Unsupported file type '{suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
            ))
            continue

        # ── Validate file size ─────────────────────────────────────────────
        content = await file.read()
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            responses.append(UploadResponse(
                candidate_id="",
                filename=file.filename or "unknown",
                status="error",
                message=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit.",
            ))
            continue

        # ── Save to disk ───────────────────────────────────────────────────
        temp_id   = str(uuid.uuid4())
        safe_name = f"{temp_id}{suffix}"
        save_path = _UPLOAD_DIR / safe_name

        try:
            with save_path.open("wb") as f:
                f.write(content)

            # ── Extract text ───────────────────────────────────────────────
            raw_text = extractor.extract(save_path)

            if not raw_text.strip():
                responses.append(UploadResponse(
                    candidate_id="",
                    filename=file.filename or "unknown",
                    status="error",
                    message="Could not extract text from file. Is it scanned/image-only?",
                ))
                save_path.unlink(missing_ok=True)
                continue

            # ── Parse structured data ──────────────────────────────────────
            parsed = resume_parser.parse(raw_text, file.filename or safe_name)

            # Override the temp_id with the parser's UUID for consistency
            save_path.rename(_UPLOAD_DIR / f"{parsed.candidate_id}{suffix}")

            # ── Store in analyzer ──────────────────────────────────────────
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
            responses.append(UploadResponse(
                candidate_id="",
                filename=file.filename or "unknown",
                status="error",
                message=f"Processing error: {str(exc)}",
            ))

    return responses


@router.get(
    "/",
    response_model=List[dict],
    summary="List all uploaded résumés",
)
async def list_resumes():
    """Return a lightweight summary of all stored résumés."""
    resumes = resume_analyzer.list_resumes()
    return [
        {
            "candidate_id":   r.candidate_id,
            "filename":       r.filename,
            "full_name":      r.full_name,
            "email":          r.email,
            "skills_count":   len(r.skills),
            "experience_years": r.total_experience_years,
            "education":      r.highest_education.value,
            "top_skills":     r.skills[:8],
        }
        for r in resumes
    ]


@router.delete(
    "/{candidate_id}",
    summary="Remove a résumé by ID",
)
async def delete_resume(candidate_id: str):
    resume = resume_analyzer.get_resume(candidate_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Résumé not found.")

    resume_analyzer._store.pop(candidate_id, None)

    # Remove from disk
    for ext in settings.ALLOWED_EXTENSIONS:
        p = _UPLOAD_DIR / f"{candidate_id}{ext}"
        p.unlink(missing_ok=True)

    return {"message": f"Résumé {candidate_id[:8]} deleted."}


@router.delete(
    "/",
    summary="Clear all résumés",
)
async def clear_all_resumes():
    count = len(resume_analyzer.list_resumes())
    resume_analyzer.clear_resumes()

    # Clear upload directory
    for f in _UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()

    return {"message": f"Cleared {count} résumé(s)."}
