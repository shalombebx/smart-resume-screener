"""
app/services/reasoner.py
─────────────────────────
Uses OpenAI Chat API (GPT-4.1) to generate human-readable candidate
evaluation summaries, strengths/weaknesses, and hiring recommendations.

Prompt engineering:
- System prompt primes GPT as a senior technical recruiter
- User prompt injects structured résumé + JD data
- Response is parsed into AIExplanation schema
"""

from __future__ import annotations
import json
import re
from typing import List

from openai import OpenAI

from app.core.config import settings
from app.models.schemas import AIExplanation, FitLevel, ScoreBreakdown
from app.core.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior technical recruiter and hiring manager with 15+ years of experience \
evaluating candidates across software engineering, data science, and AI/ML roles.

Your task is to evaluate a candidate's résumé against a job description and produce \
a structured, insightful evaluation. Be concise, honest, and objective.

You MUST respond with ONLY valid JSON — no markdown, no preamble, no explanation outside the JSON.

JSON schema to follow EXACTLY:
{
  "summary": "<2–3 sentence overview of the candidate's fit>",
  "strengths": ["<strength 1>", "<strength 2>", "..."],
  "weaknesses": ["<weakness 1>", "..."],
  "missing_requirements": ["<requirement not met 1>", "..."],
  "recommendation": "<one clear sentence hiring recommendation>",
  "fit_level": "<Strong Fit | Moderate Fit | Weak Fit>"
}
"""


class AIReasoner:
    """
    Generates GPT-powered hiring evaluations for each candidate.
    """

    def __init__(self) -> None:
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not configured.")
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def explain(
        self,
        *,
        candidate_name: str,
        resume_text:    str,
        jd_text:        str,
        score:          ScoreBreakdown,
        matching_skills: List[str],
        missing_skills:  List[str],
    ) -> AIExplanation:
        """
        Call GPT and return a structured AIExplanation.
        Falls back to a template-based explanation if the API call fails.
        """
        try:
            return self._call_gpt(
                candidate_name=candidate_name,
                resume_text=resume_text,
                jd_text=jd_text,
                score=score,
                matching_skills=matching_skills,
                missing_skills=missing_skills,
            )
        except Exception as exc:
            logger.warning("GPT reasoning failed for %s: %s — using fallback", candidate_name, exc)
            return self._fallback_explanation(score, matching_skills, missing_skills)

    # ── GPT call ───────────────────────────────────────────────────────────────

    def _call_gpt(
        self,
        *,
        candidate_name:  str,
        resume_text:     str,
        jd_text:         str,
        score:           ScoreBreakdown,
        matching_skills: List[str],
        missing_skills:  List[str],
    ) -> AIExplanation:
        # Truncate inputs to stay well within context limits
        resume_snippet = resume_text[:3000]
        jd_snippet     = jd_text[:2000]

        user_prompt = f"""
## Job Description (excerpt)
{jd_snippet}

## Candidate: {candidate_name or 'Unknown'}

### Résumé (excerpt)
{resume_snippet}

### Score Breakdown
- Embedding Similarity:  {score.embedding_similarity:.1f}%
- Keyword Match:         {score.keyword_match:.1f}%
- Experience Alignment:  {score.experience_alignment:.1f}%
- Education Match:       {score.education_match:.1f}%
- Bonus Skills:          {score.bonus_skills:.1f}%
- **Final Score:         {score.final_score:.1f}%**

### Matching Skills
{', '.join(matching_skills) if matching_skills else 'None identified'}

### Missing Skills
{', '.join(missing_skills) if missing_skills else 'None identified'}

Analyse this candidate against the job description and explain why they received \
this score. Focus on technical fit, experience level, and cultural alignment. \
Return ONLY the JSON object specified.
"""

        response = self.client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=settings.OPENAI_MAX_TOKENS,
            temperature=settings.OPENAI_TEMPERATURE,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        return self._parse_response(raw, score)

    # ── Response parsing ───────────────────────────────────────────────────────

    def _parse_response(self, raw: str, score: ScoreBreakdown) -> AIExplanation:
        """Parse GPT JSON response into AIExplanation."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON block
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}

        fit_str = data.get("fit_level", "")
        fit_level = self._parse_fit_level(fit_str, score.final_score)

        return AIExplanation(
            summary=data.get("summary", "No summary available."),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            missing_requirements=data.get("missing_requirements", []),
            recommendation=data.get("recommendation", ""),
            fit_level=fit_level,
        )

    def _parse_fit_level(self, fit_str: str, score: float) -> FitLevel:
        lower = fit_str.lower()
        if "strong" in lower:
            return FitLevel.STRONG
        if "moderate" in lower or "medium" in lower:
            return FitLevel.MODERATE
        if "weak" in lower or "poor" in lower:
            return FitLevel.WEAK
        # Derive from score if GPT didn't classify properly
        if score >= 70:
            return FitLevel.STRONG
        elif score >= 45:
            return FitLevel.MODERATE
        else:
            return FitLevel.WEAK

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _fallback_explanation(
        self,
        score: ScoreBreakdown,
        matching_skills: List[str],
        missing_skills: List[str],
    ) -> AIExplanation:
        """
        Template-based explanation when GPT is unavailable.
        """
        fit = self._parse_fit_level("", score.final_score)

        if fit == FitLevel.STRONG:
            summary = (
                f"This candidate demonstrates strong alignment with the role, "
                f"achieving an overall score of {score.final_score:.1f}%. "
                f"Their skill set closely matches the job requirements."
            )
            recommendation = "Strongly recommended for interview."
        elif fit == FitLevel.MODERATE:
            summary = (
                f"This candidate shows moderate fit for the role ({score.final_score:.1f}%). "
                f"While several key skills are present, some gaps exist."
            )
            recommendation = "Consider for interview pending further review."
        else:
            summary = (
                f"This candidate shows limited alignment with the role ({score.final_score:.1f}%). "
                f"Multiple key requirements appear to be unmet."
            )
            recommendation = "Not recommended at this time."

        return AIExplanation(
            summary=summary,
            strengths=[f"Proficient in: {s}" for s in matching_skills[:5]],
            weaknesses=[f"Limited evidence of: {s}" for s in missing_skills[:3]],
            missing_requirements=missing_skills[:5],
            recommendation=recommendation,
            fit_level=fit,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
ai_reasoner = AIReasoner()
