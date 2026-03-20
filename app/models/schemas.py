"""
app/models/schemas.py
─────────────────────
All Pydantic v2 request / response schemas.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class HiringRecommendation(str, Enum):
    STRONG_FIT = "Strong Fit"
    MODERATE_FIT = "Moderate Fit"
    WEAK_FIT = "Weak Fit"
    NOT_RECOMMENDED = "Not Recommended"


# ── Parsed Resume ─────────────────────────────────────────────────────────────

class ParsedResume(BaseModel):
    """Structured representation of a parsed résumé document."""
    filename: str
    raw_text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: list[str] = Field(default_factory=list)
    experience_years: float = 0.0
    education_level: str = ""        # e.g. "Bachelor's", "Master's", "PhD"
    education_field: str = ""        # e.g. "Computer Science"
    job_titles: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    summary: str = ""


# ── Score Breakdown ───────────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    embedding_similarity: float = Field(ge=0, le=100)
    keyword_match: float = Field(ge=0, le=100)
    experience_alignment: float = Field(ge=0, le=100)
    education_match: float = Field(ge=0, le=100)
    bonus_skills: float = Field(ge=0, le=100)
    final_score: float = Field(ge=0, le=100)


# ── Candidate Result ──────────────────────────────────────────────────────────

class CandidateResult(BaseModel):
    rank: int
    filename: str
    candidate_name: str
    score: float = Field(ge=0, le=100, description="Final weighted score (0–100)")
    score_breakdown: ScoreBreakdown
    matching_skills: list[str]
    missing_skills: list[str]
    experience_years: float
    education: str
    ai_explanation: str
    strengths: list[str]
    weaknesses: list[str]
    recommendation: HiringRecommendation
    parsed_data: ParsedResume


# ── Analysis Request / Response ───────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    job_description: str = Field(min_length=50, description="Full job description text")
    session_id: str = Field(description="UUID linking to uploaded resume files")


class AnalysisResponse(BaseModel):
    session_id: str
    job_title: str
    total_candidates: int
    processing_time_seconds: float
    results: list[CandidateResult]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Upload Response ────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    files_received: list[str]
    message: str


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    openai_connected: bool
    cache_entries: int
    version: str = "1.0.0"
