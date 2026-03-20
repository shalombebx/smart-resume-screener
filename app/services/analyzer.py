"""
app/services/analyzer.py
─────────────────────────
Top-level orchestrator for the résumé screening pipeline.

Pipeline:
  1. Load parsed résumés from in-memory store
  2. Extract skills from JD
  3. Compute OpenAI embeddings (JD + all résumés) — cached
  4. Compute hybrid scores per candidate
  5. Compute AI reasoning per candidate
  6. Sort by final_score, assign ranks
  7. Return AnalysisSession

This module is the ONLY entry-point for running a full analysis.
"""

from __future__ import annotations
import time
import uuid
from typing import Dict, List, Optional

from app.models.schemas import (
    AnalysisSession, CandidateResult, ParsedResume
)
from app.services.embedder  import embedding_service
from app.services.scorer    import hybrid_scorer
from app.services.reasoner  import ai_reasoner
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
        # In-memory store: candidate_id → ParsedResume
        self._store: Dict[str, ParsedResume] = {}
        # Session store: session_id → AnalysisSession
        self._sessions: Dict[str, AnalysisSession] = {}

    # ── Store management ───────────────────────────────────────────────────────

    def add_resume(self, resume: ParsedResume) -> None:
        self._store[resume.candidate_id] = resume
        logger.info("Stored résumé: %s (id=%s)", resume.filename, resume.candidate_id[:8])

    def get_resume(self, candidate_id: str) -> Optional[ParsedResume]:
        return self._store.get(candidate_id)

    def list_resumes(self) -> List[ParsedResume]:
        return list(self._store.values())

    def clear_resumes(self) -> None:
        self._store.clear()

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        return self._sessions.get(session_id)

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def analyze(
        self,
        job_description: str,
        candidate_ids:   Optional[List[str]] = None,
    ) -> AnalysisSession:
        """
        Run the full pipeline and return a ranked AnalysisSession.

        Args:
            job_description: Full JD text.
            candidate_ids:   If provided, only analyse these candidates.
                             None → analyse all stored résumés.

        Returns:
            AnalysisSession with ranked CandidateResult list.

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

        # ── Step 1: Extract JD skills ─────────────────────────────────────────
        jd_skills = extract_skills_from_text(job_description)
        logger.info("JD skills detected: %s", jd_skills[:15])

        # ── Step 2: Embed JD ──────────────────────────────────────────────────
        logger.info("Computing JD embedding…")
        jd_embedding_score_map: Dict[str, float] = {}

        # ── Step 3: Embed + score each résumé ─────────────────────────────────
        results: list[CandidateResult] = []

        for resume in resumes:
            logger.info("Processing: %s", resume.filename)

            # 3a. Embedding similarity
            emb_score = embedding_service.similarity(job_description, resume.raw_text)

            # 3b. Hybrid score
            score_breakdown = hybrid_scorer.score(
                resume=resume,
                jd_text=job_description,
                jd_skills=jd_skills,
                embedding_score=emb_score,
            )

            # 3c. Skills overlap
            matching_skills, missing_skills = skills_overlap(resume.skills, jd_skills)

            # 3d. AI explanation
            ai_explanation = ai_reasoner.explain(
                candidate_name=resume.full_name or resume.filename,
                resume_text=resume.raw_text,
                jd_text=job_description,
                score=score_breakdown,
                matching_skills=matching_skills,
                missing_skills=missing_skills,
            )

            results.append(CandidateResult(
                rank=0,                    # assigned after sorting
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

        # ── Step 4: Sort & rank ────────────────────────────────────────────────
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
            return [
                r for cid in candidate_ids
                if (r := self._store.get(cid)) is not None
            ]
        return list(self._store.values())


# ── Application-level singleton ────────────────────────────────────────────────
resume_analyzer = ResumeAnalyzer()
