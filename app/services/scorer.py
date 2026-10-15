"""
app/services/scorer.py
───────────────────────
Hybrid scoring engine.

Combines five signals into a weighted final score (0–100):

  Component            Weight  Method
  ────────────────────────────────────────────────────────────────────────
  Embedding Similarity  50%    Cosine similarity between JD & résumé embeddings
  Keyword Match         20%    % of JD skills found in résumé (via ontology)
  Experience Alignment  15%    Candidate years vs. years required by the JD
  Education Match       10%    Degree level vs JD requirement
  Bonus Skills          5%     In-demand extras not explicitly required
"""

from __future__ import annotations
import re
from typing import List

from app.core.config import settings
from app.models.schemas import EDUCATION_RANK, EducationLevel, ParsedResume, ScoreBreakdown
from app.utils.skill_ontology import keyword_match_score
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Explicit "years of experience" statements in a JD ─────────────────────────
_YEARS_PATTERNS = [
    re.compile(r"(?:at\s+least|minimum(?:\s+of)?|min\.?)\s*(\d{1,2})\s*\+?\s*years?", re.I),
    re.compile(r"(\d{1,2})\s*\+\s*years?", re.I),
    re.compile(r"(\d{1,2})\s*(?:-|–|—|to)\s*\d{1,2}\s*\+?\s*years?", re.I),
    re.compile(r"(\d{1,2})\s*years?\s+(?:of\s+|in\s+|with\s+)?(?:[\w\-]+\s+){0,3}?(?:experience|exp)\b", re.I),
]

# ── Seniority keywords — only searched in the JD's title area ─────────────────
# (searching the whole text made "mentor junior engineers" look like an entry-level role)
_SENIORITY: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(staff|director|architect|head of)\b", re.I), 10.0),
    (re.compile(r"\b(lead|principal)\b", re.I), 8.0),
    (re.compile(r"\b(senior|sr\.?)\b", re.I), 6.0),
    (re.compile(r"\bmid[\s\-]?level\b", re.I), 4.0),
    (re.compile(r"\b(intern|internship|entry[\s\-]?level|junior|jr\.?|graduate)\b", re.I), 1.0),
]
_TITLE_WINDOW = 300

# ── Education requirement patterns ─────────────────────────────────────────────
_L, _R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"      # \b breaks after a trailing "." (e.g. "M.S.")
_EDU_REQ: list[tuple[EducationLevel, re.Pattern]] = [
    (EducationLevel.PHD, re.compile(rf"{_L}(?:ph\.?\s?d|doctorate|doctoral){_R}", re.I)),
    (EducationLevel.MASTERS, re.compile(rf"{_L}(?:master(?:'s|’s|s)?|m\.?sc|m\.s\.?|mba){_R}", re.I)),
    (EducationLevel.BACHELORS, re.compile(rf"{_L}(?:bachelor(?:'s|’s|s)?|b\.?sc|b\.s\.?|b\.?tech){_R}", re.I)),
    (EducationLevel.ASSOCIATE, re.compile(rf"{_L}(?:associate(?:'s|’s)?\s+degree|diploma|hnd){_R}", re.I)),
]
_GENERIC_DEGREE = re.compile(r"\bdegree\b", re.I)

# ── Bonus / in-demand skills (must be canonical names from the ontology) ──────
_BONUS_SKILLS = [
    "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "MLOps", "MLflow", "LangChain", "Hugging Face",
    "Generative AI", "LLMs", "Terraform", "CI/CD", "FastAPI",
]


class HybridScorer:
    """Stateless scorer — call ``score()`` to receive a populated ``ScoreBreakdown``."""

    def score(
        self,
        resume: ParsedResume,
        jd_text: str,
        jd_skills: List[str],
        embedding_score: float,            # pre-computed 0–100
    ) -> ScoreBreakdown:
        embedding_score = min(max(embedding_score, 0.0), 100.0)
        kw = keyword_match_score(resume.skills, jd_skills)
        exp = self._score_experience(resume.total_experience_years, jd_text)
        edu = self._score_education(resume.highest_education, jd_text)
        bonus = self._score_bonus(resume.skills)

        final = (
            embedding_score * settings.WEIGHT_EMBEDDING
            + kw * settings.WEIGHT_KEYWORD
            + exp * settings.WEIGHT_EXPERIENCE
            + edu * settings.WEIGHT_EDUCATION
            + bonus * settings.WEIGHT_BONUS
        )

        logger.debug(
            "Scores for %s — emb=%.1f kw=%.1f exp=%.1f edu=%.1f bonus=%.1f → final=%.1f",
            resume.filename, embedding_score, kw, exp, edu, bonus, final,
        )

        return ScoreBreakdown(
            embedding_similarity=round(embedding_score, 2),
            keyword_match=round(kw, 2),
            experience_alignment=round(exp, 2),
            education_match=round(edu, 2),
            bonus_skills=round(bonus, 2),
            final_score=round(min(max(final, 0.0), 100.0), 2),
        )

    # ── Experience alignment ───────────────────────────────────────────────────

    def _score_experience(self, candidate_years: float, jd_text: str) -> float:
        required_years = self._parse_required_experience(jd_text)

        if required_years == 0:
            # JD doesn't specify → base score on reasonable expectations
            if candidate_years >= 5:
                return 90.0
            if candidate_years >= 3:
                return 75.0
            if candidate_years >= 1:
                return 60.0
            return 40.0

        ratio = candidate_years / required_years
        if ratio >= 1.5:
            return 100.0
        if ratio >= 1.0:
            return 90.0
        if ratio >= 0.75:
            return 70.0
        if ratio >= 0.5:
            return 50.0
        return max(10.0, ratio * 100)

    def _parse_required_experience(self, jd_text: str) -> float:
        """Minimum years required by the JD; 0.0 if it cannot be determined."""
        found = [float(m.group(1)) for p in _YEARS_PATTERNS for m in p.finditer(jd_text)]
        found = [y for y in found if 0 < y <= 20]
        if found:
            return max(found)

        title_area = jd_text[:_TITLE_WINDOW]
        for pattern, years in _SENIORITY:
            if pattern.search(title_area):
                return years
        return 0.0

    # ── Education match ────────────────────────────────────────────────────────

    def _score_education(self, candidate_level: EducationLevel, jd_text: str) -> float:
        required = self._parse_required_education(jd_text)
        cand_rank = EDUCATION_RANK.get(candidate_level, 0)
        req_rank = EDUCATION_RANK.get(required, 0)

        if req_rank == 0:
            return 75.0                       # JD didn't specify — neutral score

        if cand_rank >= req_rank:
            return min(100.0, 80.0 + (cand_rank - req_rank) * 10)
        return max(10.0, 70.0 - (req_rank - cand_rank) * 20)

    def _parse_required_education(self, jd_text: str) -> EducationLevel:
        """
        Lowest level the JD mentions. "Bachelor's or Master's" → Bachelor's, and
        "Bachelor's required, PhD preferred" → Bachelor's (the old code returned
        the highest level mentioned, penalising candidates who met the real minimum).
        """
        mentioned = [lvl for lvl, pat in _EDU_REQ if pat.search(jd_text)]
        if mentioned:
            return min(mentioned, key=lambda l: EDUCATION_RANK[l])
        if _GENERIC_DEGREE.search(jd_text):
            return EducationLevel.BACHELORS
        return EducationLevel.NONE

    # ── Bonus skills ───────────────────────────────────────────────────────────

    def _score_bonus(self, candidate_skills: List[str]) -> float:
        """1 match = 20 points … 5+ matches = 100."""
        candidate_set = set(candidate_skills)
        matches = sum(1 for s in _BONUS_SKILLS if s in candidate_set)
        return min(100.0, matches * 20.0)


# ── Singleton ──────────────────────────────────────────────────────────────────
hybrid_scorer = HybridScorer()
