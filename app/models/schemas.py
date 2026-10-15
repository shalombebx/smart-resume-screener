"""
app/models/schemas.py
─────────────────────
All Pydantic v2 request / response schemas.

This is the single source of truth for the data contract used by the
parser, scorer, reasoner, analyzer, API routers, demo script and frontend.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class EducationLevel(str, Enum):
    NONE = "None"
    DIPLOMA = "Diploma"
    ASSOCIATE = "Associate"
    BACHELORS = "Bachelor's"
    MASTERS = "Master's"
    PHD = "PhD"


# Higher number = higher qualification. Shared by parser and scorer.
EDUCATION_RANK: dict[EducationLevel, int] = {
    EducationLevel.NONE: 0,
    EducationLevel.DIPLOMA: 1,
    EducationLevel.ASSOCIATE: 2,
    EducationLevel.BACHELORS: 3,
    EducationLevel.MASTERS: 4,
    EducationLevel.PHD: 5,
}


class FitLevel(str, Enum):
    STRONG = "Strong Fit"
    MODERATE = "Moderate Fit"
    WEAK = "Weak Fit"


# ── Parsed résumé ─────────────────────────────────────────────────────────────

class ExperienceEntry(BaseModel):
    duration: str = ""
    years: float = 0.0
    title: Optional[str] = None


class EducationEntry(BaseModel):
    level: EducationLevel


class ParsedResume(BaseModel):
    """Structured representation of a parsed résumé document."""
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    raw_text: str = ""
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    total_experience_years: float = 0.0
    highest_education: EducationLevel = EducationLevel.NONE


# ── Scoring ───────────────────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    embedding_similarity: float = Field(ge=0, le=100)
    keyword_match: float = Field(ge=0, le=100)
    experience_alignment: float = Field(ge=0, le=100)
    education_match: float = Field(ge=0, le=100)
    bonus_skills: float = Field(ge=0, le=100)
    final_score: float = Field(ge=0, le=100)


class AIExplanation(BaseModel):
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    recommendation: str = ""
    fit_level: FitLevel = FitLevel.WEAK


class CandidateResult(BaseModel):
    rank: int
    candidate_id: str
    filename: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    score: ScoreBreakdown
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    ai_explanation: Optional[AIExplanation] = None
    parsed_resume: Optional[ParsedResume] = None


class AnalysisSession(BaseModel):
    session_id: str
    job_description: str
    candidates: list[CandidateResult]
    total_resumes: int
    processing_time_seconds: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Requests / responses ──────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    job_description: str = Field(min_length=50, description="Full job description text")
    candidate_ids: Optional[list[str]] = Field(
        default=None, description="Subset of candidate IDs; omit to analyse all uploaded résumés"
    )


class ExportRequest(BaseModel):
    format: Literal["csv", "json"] = "csv"


class UploadResponse(BaseModel):
    candidate_id: str
    filename: str
    status: Literal["success", "error"]
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    openai_ready: bool
