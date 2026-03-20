"""
app/services/scorer.py
───────────────────────
Hybrid scoring engine.

Combines five signals into a weighted final score (0–100):

  Component            Weight  Method
  ────────────────────────────────────────────────────────────────────────
  Embedding Similarity  50%    Cosine similarity between JD & résumé embeddings
  Keyword Match         20%    % of JD skills found in résumé (via ontology)
  Experience Alignment  15%    Heuristic match of years / seniority keywords
  Education Match       10%    Degree level vs JD requirement
  Bonus Skills          5%     In-demand extras not explicitly required

"""

from __future__ import annotations
import re
from typing import List, Optional

from app.core.config import settings
from app.models.schemas import (
    ParsedResume, ScoreBreakdown, EducationLevel
)
from app.utils.skill_ontology import (
    extract_skills_from_text, keyword_match_score
)
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Education level numeric rank (higher = better) ────────────────────────────
_EDU_RANK: dict[EducationLevel, int] = {
    EducationLevel.NONE:       0,
    EducationLevel.DIPLOMA:    1,
    EducationLevel.ASSOCIATE:  2,
    EducationLevel.BACHELORS:  3,
    EducationLevel.MASTERS:    4,
    EducationLevel.PHD:        5,
}

# ── Seniority/experience keywords in JD ───────────────────────────────────────
_EXP_LEVELS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(intern|entry.?level|junior|0[\–\-–—]?\s*1\s*year)", re.I), 1.0),
    (re.compile(r"\b(1[\–\-–—]?\s*2\s*years?|1\+|2\+)",                  re.I), 2.0),
    (re.compile(r"\b(2[\–\-–—]?\s*3\s*years?|3\+)",                       re.I), 3.0),
    (re.compile(r"\b(3[\–\-–—]?\s*5\s*years?|mid.?level)",                re.I), 4.0),
    (re.compile(r"\b(5[\–\-–—]?\s*7\s*years?|5\+|senior)",                re.I), 6.0),
    (re.compile(r"\b(7[\–\-–—]?\s*10\s*years?|7\+|lead|principal)",       re.I), 8.0),
    (re.compile(r"\b(10\+|10[\–\-–—]?\s*\d+|staff|director|architect)",   re.I), 10.0),
]

# ── Bonus / in-demand skills ──────────────────────────────────────────────────
_BONUS_SKILLS = [
    "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "MLOps", "MLflow", "LangChain", "Hugging Face",
    "Generative AI", "LLMs", "GPT / OpenAI",
    "Terraform", "GitHub Actions", "FastAPI",
]


class HybridScorer:
    """
    Stateless scorer — call ``score()`` with a résumé and job-description
    context to receive a populated ``ScoreBreakdown``.
    """

    def score(
        self,
        resume: ParsedResume,
        jd_text: str,
        jd_skills: List[str],
        embedding_score: float,            # pre-computed 0–100
    ) -> ScoreBreakdown:
        """
        Compute all scoring components and return a ScoreBreakdown.

        Args:
            resume:           Parsed résumé object.
            jd_text:          Raw job description text.
            jd_skills:        Skills extracted from JD.
            embedding_score:  Pre-computed cosine similarity × 100.

        Returns:
            ScoreBreakdown with each component and final weighted score.
        """
        kw    = keyword_match_score(resume.skills, jd_skills)
        exp   = self._score_experience(resume.total_experience_years, jd_text)
        edu   = self._score_education(resume.highest_education, jd_text)
        bonus = self._score_bonus(resume.skills)

        final = (
            embedding_score * settings.WEIGHT_EMBEDDING
            + kw            * settings.WEIGHT_KEYWORD
            + exp           * settings.WEIGHT_EXPERIENCE
            + edu           * settings.WEIGHT_EDUCATION
            + bonus         * settings.WEIGHT_BONUS
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
            final_score=round(min(final, 100.0), 2),
        )

    # ── Experience alignment ───────────────────────────────────────────────────

    def _score_experience(self, candidate_years: float, jd_text: str) -> float:
        """
        Compare candidate's years of experience against the JD's expected level.
        Returns 0–100.
        """
        required_years = self._parse_required_experience(jd_text)

        if required_years == 0:
            # JD doesn't specify → base score on reasonable expectations
            if candidate_years >= 5:
                return 90.0
            elif candidate_years >= 3:
                return 75.0
            elif candidate_years >= 1:
                return 60.0
            else:
                return 40.0

        ratio = candidate_years / required_years
        if ratio >= 1.5:
            return 100.0
        elif ratio >= 1.0:
            return 90.0
        elif ratio >= 0.75:
            return 70.0
        elif ratio >= 0.5:
            return 50.0
        else:
            return max(10.0, ratio * 100)

    def _parse_required_experience(self, jd_text: str) -> float:
        """
        Extract the minimum required years from a JD.
        Returns 0.0 if not found.
        """
        for pattern, years in _EXP_LEVELS:
            if pattern.search(jd_text):
                return years

        # Try explicit "X years" anywhere
        m = re.search(r"(\d+)\s*\+?\s*years?", jd_text, re.I)
        if m:
            return float(m.group(1))

        return 0.0

    # ── Education match ────────────────────────────────────────────────────────

    def _score_education(self, candidate_level: EducationLevel, jd_text: str) -> float:
        """
        Score 0–100: does the candidate meet or exceed the required degree?
        """
        required = self._parse_required_education(jd_text)
        cand_rank = _EDU_RANK.get(candidate_level, 0)
        req_rank  = _EDU_RANK.get(required, 0)

        if req_rank == 0:
            return 75.0                       # JD didn't specify — neutral score

        if cand_rank >= req_rank:
            # Meets or exceeds — full or bonus points
            excess = cand_rank - req_rank
            return min(100.0, 80.0 + excess * 10)
        else:
            # Doesn't meet requirement
            gap = req_rank - cand_rank
            return max(10.0, 70.0 - gap * 20)

    def _parse_required_education(self, jd_text: str) -> EducationLevel:
        lower = jd_text.lower()
        if any(k in lower for k in ["phd", "ph.d", "doctorate"]):
            return EducationLevel.PHD
        if any(k in lower for k in ["master", "m.s.", "msc", "mba"]):
            return EducationLevel.MASTERS
        if any(k in lower for k in ["bachelor", "b.s.", "bsc", "b.tech", "degree"]):
            return EducationLevel.BACHELORS
        if any(k in lower for k in ["associate", "diploma"]):
            return EducationLevel.ASSOCIATE
        return EducationLevel.NONE

    # ── Bonus skills ───────────────────────────────────────────────────────────

    def _score_bonus(self, candidate_skills: List[str]) -> float:
        """
        Award bonus points for in-demand/premium skills not necessarily
        listed in the JD. Returns 0–100.
        """
        candidate_set = set(candidate_skills)
        matches = sum(1 for s in _BONUS_SKILLS if s in candidate_set)
        if not matches:
            return 0.0
        # 1 match = 20, 2 = 40, …, 5+ = 100
        return min(100.0, matches * 20.0)


# ── Singleton ──────────────────────────────────────────────────────────────────
hybrid_scorer = HybridScorer()
