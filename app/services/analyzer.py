"""
app/services/analyzer.py
─────────────────────────
Top-level orchestrator for the résumé screening pipeline.

Pipeline:
  1. Load parsed résumés from the in-memory store
  2. Extract skills from the JD
  3. Embed the JD once (cached)
  4. Per résumé: embedding similarity → hybrid score → skills overlap → AI reasoning
  5. Sort by final_score, assign ranks
  6. Return AnalysisSession

This module is the ONLY entry-point for running a full analysis.
"""

from __future__ import annotations
import threading
import time
import uuid
from typing import Dict, List, Optional

from app.models.schemas import AnalysisSession, CandidateResult, ParsedResume
from app.services.embedder import embedding_service
from app.services.scorer import hybrid_scorer
from app.services.reasoner import ai_reasoner
from app.utils.skill_ontology import extract_skills_from_text, skills_overlap
from app.core.logger import get_logger

logger = get_logger(__name__)


class ResumeAnalyzer:
    """
    Orchestrates the full résumé-screening pipeline.
    ParsedResumes are kept in an in-memory store (dict keyed by candidate_id).
    In production, replace with a database / Redis store.
    """

    def __init__(self) -> None:
        self._store: Dict[str, ParsedResume] = {}
        self._sessions: Dict[str, AnalysisSession] = {}
        self._lock = threading.Lock()      # FastAPI runs sync endpoints in a thread pool

    # ── Store management ───────────────────────────────────────────────────────

    def add_resume(self, resume: ParsedResume) -> None:
        with self._lock:
            self._store[resume.candidate_id] = resume
        logger.info("Stored résumé: %s (id=%s)", resume.filename, resume.candidate_id[:8])

    def get_resume(self, candidate_id: str) -> Optional[ParsedResume]:
        return self._store.get(candidate_id)

    def remove_resume(self, candidate_id: str) -> bool:
        with self._lock:
            return self._store.pop(candidate_id, None) is not None

    def list_resumes(self) -> List[ParsedResume]:
        with self._lock:
            return list(self._store.values())

    def clear_resumes(self) -> None:
        with self._lock:
            self._store.clear()

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        return self._sessions.get(session_id)

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def analyze(
        self,
        job_description: str,
        candidate_ids: Optional[List[str]] = None,
    ) -> AnalysisSession:
        """
        Run the full pipeline and return a ranked AnalysisSession.

        Raises:
            ValueError: If no résumés are available.
        """
        t_start = time.perf_counter()
        session_id = str(uuid.uuid4())

        resumes = self._select_resumes(candidate_ids)
        if not resumes:
            raise ValueError("No résumés found. Please upload résumés before analysing.")

        logger.info(
            "Starting analysis session %s — %d résumé(s), JD length=%d chars",
            session_id[:8], len(resumes), len(job_description),
        )

        jd_skills = extract_skills_from_text(job_description)
        logger.info("JD skills detected: %s", jd_skills[:15])

        logger.info("Computing JD embedding…")
        jd_vector = embedding_service.embed(job_description)      # once, not once per résumé

        results: list[CandidateResult] = []
        for resume in resumes:
            logger.info("Processing: %s", resume.filename)

            emb_score = embedding_service.similarity_to_vector(jd_vector, resume.raw_text)

            score_breakdown = hybrid_scorer.score(
                resume=resume,
                jd_text=job_description,
                jd_skills=jd_skills,
                embedding_score=emb_score,
            )

            matching_skills, missing_skills = skills_overlap(resume.skills, jd_skills)

            ai_explanation = ai_reasoner.explain(
                candidate_name=resume.full_name or resume.filename,
                resume_text=resume.raw_text,
                jd_text=job_description,
                score=score_breakdown,
                matching_skills=matching_skills,
                missing_skills=missing_skills,
            )

            results.append(CandidateResult(
                rank=0,                              # assigned after sorting
                candidate_id=resume.candidate_id,
                filename=resume.filename,
                full_name=resume.full_name,
                email=resume.email,
                score=score_breakdown,
                matching_skills=matching_skills,
                missing_skills=missing_skills,
                ai_explanation=ai_explanation,
                parsed_resume=resume,
            ))

        results.sort(key=lambda r: r.score.final_score, reverse=True)
        for i, result in enumerate(results, start=1):
            result.rank = i

        elapsed = round(time.perf_counter() - t_start, 2)
        logger.info(
            "Analysis session %s complete — %d candidates in %.2fs",
            session_id[:8], len(results), elapsed,
        )

        session = AnalysisSession(
            session_id=session_id,
            job_description=job_description,
            candidates=results,
            total_resumes=len(results),
            processing_time_seconds=elapsed,
        )
        self._sessions[session_id] = session
        return session

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _select_resumes(self, candidate_ids: Optional[List[str]]) -> List[ParsedResume]:
        if candidate_ids:
            return [r for cid in candidate_ids if (r := self._store.get(cid)) is not None]
        return self.list_resumes()


# ── Application-level singleton ────────────────────────────────────────────────
resume_analyzer = ResumeAnalyzer()
