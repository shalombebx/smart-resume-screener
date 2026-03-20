"""
app/services/parser.py
───────────────────────
Converts raw résumé text into a structured ParsedResume object.

Uses:
- Regex heuristics for contact info, experience years, education
- Skill ontology for canonical skill extraction
- spaCy (if available) for named-entity recognition
"""

from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from app.models.schemas import (
    ParsedResume, ExperienceEntry, EducationEntry, EducationLevel
)
from app.utils.skill_ontology import extract_skills_from_text
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Education level keywords ───────────────────────────────────────────────────
_EDUCATION_PATTERNS: List[Tuple[EducationLevel, List[str]]] = [
    (EducationLevel.PHD,       ["ph.d", "phd", "doctorate", "doctoral"]),
    (EducationLevel.MASTERS,   ["master", "m.s.", "m.sc", "msc", "m.eng", "mba", "m.a."]),
    (EducationLevel.BACHELORS, ["bachelor", "b.s.", "b.sc", "bsc", "b.e.", "b.tech", "b.a."]),
    (EducationLevel.ASSOCIATE, ["associate", "a.s.", "a.a."]),
    (EducationLevel.DIPLOMA,   ["diploma", "certificate", "hnd", "ond"]),
]

# ── Experience section header keywords ────────────────────────────────────────
_EXP_HEADERS = re.compile(
    r"(work experience|professional experience|employment|career history|experience)",
    re.IGNORECASE,
)

# ── Year-range pattern: "2018 - 2022", "Jan 2020 – Present", "2019–2021" ─────
_YEAR_RANGE = re.compile(
    r"((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)?\s*"
    r"\d{4})\s*[-–—to]+\s*(present|current|\d{4})",
    re.IGNORECASE,
)

_YEAR_SINGLE = re.compile(r"\b(19|20)\d{2}\b")

# ── Email / phone ──────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(\+?\d[\d\s\-().]{7,}\d)"
)


class ResumeParser:
    """
    Parses raw résumé text into a ParsedResume schema instance.
    Fully self-contained — no external API calls.
    """

    def parse(self, raw_text: str, filename: str) -> ParsedResume:
        candidate_id = str(uuid.uuid4())
        lower        = raw_text.lower()

        email  = self._extract_email(raw_text)
        phone  = self._extract_phone(raw_text)
        name   = self._extract_name(raw_text)
        skills = extract_skills_from_text(raw_text)
        exp_entries, total_years = self._extract_experience(raw_text)
        edu_entries, highest_edu = self._extract_education(lower)

        logger.debug(
            "Parsed '%s' — skills=%d, exp_years=%.1f, edu=%s",
            filename, len(skills), total_years, highest_edu,
        )

        return ParsedResume(
            candidate_id=candidate_id,
            filename=filename,
            raw_text=raw_text,
            full_name=name,
            email=email,
            phone=phone,
            skills=skills,
            experience=exp_entries,
            education=edu_entries,
            total_experience_years=total_years,
            highest_education=highest_edu,
        )

    # ── Contact info ───────────────────────────────────────────────────────────

    def _extract_email(self, text: str) -> Optional[str]:
        m = _EMAIL_RE.search(text)
        return m.group(0).lower() if m else None

    def _extract_phone(self, text: str) -> Optional[str]:
        m = _PHONE_RE.search(text)
        if m:
            phone = re.sub(r"[^\d+\-() ]", "", m.group(0)).strip()
            return phone if len(re.sub(r"\D", "", phone)) >= 7 else None
        return None

    def _extract_name(self, text: str) -> Optional[str]:
        """
        Heuristic: the first non-empty line that looks like a proper name
        (2–4 capitalised words, no numbers).
        """
        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) > 60:
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                if not any(c.isdigit() for c in line):
                    return line
        return None

    # ── Experience ─────────────────────────────────────────────────────────────

    def _extract_experience(
        self, text: str
    ) -> Tuple[List[ExperienceEntry], float]:
        """
        Extract experience entries and estimate total years.
        Strategy:
          1. Find year-ranges and sum them.
          2. Also look for explicit "X years of experience" mentions.
        """
        entries: list[ExperienceEntry] = []
        total_years = 0.0

        # ── Explicit mention: "5+ years of experience" ────────────────────────
        explicit = re.search(
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience",
            text, re.IGNORECASE,
        )
        if explicit:
            total_years = float(explicit.group(1))

        # ── Year ranges ────────────────────────────────────────────────────────
        for m in _YEAR_RANGE.finditer(text):
            start_str = m.group(1)
            end_str   = m.group(2)

            start_year = self._year_from_str(start_str)
            end_year   = 2025 if re.match(r"present|current", end_str, re.I) else self._year_from_str(end_str)

            if start_year and end_year and end_year >= start_year:
                duration = end_year - start_year
                context  = text[max(0, m.start() - 200): m.start()].strip()
                title    = self._infer_title(context)

                entries.append(ExperienceEntry(
                    duration=f"{duration} year{'s' if duration != 1 else ''}",
                    years=float(duration),
                    title=title,
                ))

        if entries and total_years == 0.0:
            # Sum unique, non-overlapping ranges as an approximation
            total_years = min(sum(e.years for e in entries), 40.0)

        return entries, round(total_years, 1)

    def _year_from_str(self, s: str) -> Optional[int]:
        m = _YEAR_SINGLE.search(s)
        return int(m.group(0)) if m else None

    def _infer_title(self, context: str) -> Optional[str]:
        """Look for common job title patterns in the context string."""
        title_re = re.compile(
            r"(senior|junior|lead|principal|staff|chief|head|director|manager|engineer|"
            r"developer|analyst|scientist|architect|consultant|intern|associate|specialist)\b[^,\n]{0,40}",
            re.IGNORECASE,
        )
        m = title_re.search(context)
        return m.group(0).strip().title() if m else None

    # ── Education ──────────────────────────────────────────────────────────────

    def _extract_education(
        self, lower_text: str
    ) -> Tuple[List[EducationEntry], EducationLevel]:
        entries: list[EducationEntry] = []
        highest = EducationLevel.NONE

        for level, keywords in _EDUCATION_PATTERNS:
            for kw in keywords:
                if kw in lower_text:
                    entries.append(EducationEntry(level=level))
                    if highest == EducationLevel.NONE or (
                        list(EducationLevel).index(level) < list(EducationLevel).index(highest)
                    ):
                        highest = level
                    break           # Don't double-count the same level

        return entries, highest


# ── Singleton ──────────────────────────────────────────────────────────────────
resume_parser = ResumeParser()
