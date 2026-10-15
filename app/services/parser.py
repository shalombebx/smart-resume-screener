"""
app/services/parser.py
───────────────────────
Converts raw résumé text into a structured ParsedResume object.

Uses:
- Regex heuristics for contact info, experience years, education
- Skill ontology for canonical skill extraction
Fully self-contained — no external API calls.
"""

from __future__ import annotations
import re
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from app.models.schemas import (
    EDUCATION_RANK, EducationEntry, EducationLevel, ExperienceEntry, ParsedResume,
)
from app.utils.skill_ontology import extract_skills_from_text
from app.core.logger import get_logger

logger = get_logger(__name__)

_B = r"(?<![a-z0-9])"      # left boundary (text is lower-cased before matching)
_E = r"(?![a-z0-9])"       # right boundary

# ── Education level patterns (word-bounded; previously plain substring tests,
#    so "ond" matched "second", "master" matched "Scrum Master", etc.) ─────────
_EDUCATION_PATTERNS: List[Tuple[EducationLevel, List[re.Pattern]]] = [
    (EducationLevel.PHD, [re.compile(p) for p in (
        rf"{_B}ph\.?\s?d{_E}", rf"{_B}doctorate{_E}", rf"{_B}doctoral{_E}")]),
    (EducationLevel.MASTERS, [re.compile(p) for p in (
        rf"{_B}master(?:'s|’s|s)?\s+(?:of|degree|in){_E}", rf"{_B}masters{_E}", rf"{_B}master(?:'s|’s){_E}",
        rf"{_B}m\.?sc{_E}", rf"{_B}m\.s\.?{_E}", rf"{_B}m\.?eng{_E}", rf"{_B}mba{_E}", rf"{_B}m\.a\.?{_E}")]),
    (EducationLevel.BACHELORS, [re.compile(p) for p in (
        rf"{_B}bachelor(?:'s|’s|s)?{_E}", rf"{_B}b\.?sc{_E}", rf"{_B}b\.s\.?{_E}", rf"{_B}b\.e\.?{_E}",
        rf"{_B}b\.?tech{_E}", rf"{_B}b\.?eng{_E}", rf"{_B}b\.a\.?{_E}")]),
    (EducationLevel.ASSOCIATE, [re.compile(p) for p in (
        rf"{_B}associate(?:'s|’s|s)?\s+(?:degree|of){_E}", rf"{_B}a\.[sa]\.?{_E}")]),
    (EducationLevel.DIPLOMA, [re.compile(p) for p in (
        rf"{_B}diploma{_E}", rf"{_B}hnd{_E}", rf"{_B}ond{_E}")]),
]

_EDU_CONTEXT = re.compile(
    r"universit|college|polytechnic|school|institute|bachelor|master|ph\.?d|b\.?sc|m\.?sc|degree|diploma|\bhnd\b|\bond\b",
    re.IGNORECASE,
)

# ── Year-range pattern: "2018 - 2022", "Jan 2020 – Present", "2019–2021" ─────
_YEAR_RANGE = re.compile(
    r"((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)?\s*"
    r"\d{4})\s*(?:-|–|—|to)\s*(present|current|now|\d{4})",
    re.IGNORECASE,
)
_YEAR_SINGLE = re.compile(r"\b(?:19|20)\d{2}\b")      # (?:…) so group(0) is the full year

_EXPLICIT_YEARS = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience", re.IGNORECASE
)

# ── Email / phone ──────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d \t\-().]{7,}\d)")          # no newlines: must stay on one line
_YEAR_RANGE_ONLY = re.compile(r"^\s*(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present)\s*$", re.I)

_NAME_STOPWORDS = ("resume", "résumé", "curriculum", "vitae", "cv", "profile", "summary", "contact")


class ResumeParser:
    """Parses raw résumé text into a ParsedResume schema instance."""

    def parse(self, raw_text: str, filename: str) -> ParsedResume:
        lower = raw_text.lower()

        email = self._extract_email(raw_text)
        phone = self._extract_phone(raw_text)
        name = self._extract_name(raw_text)
        skills = extract_skills_from_text(raw_text)
        exp_entries, total_years = self._extract_experience(raw_text)
        edu_entries, highest_edu = self._extract_education(lower)

        logger.debug(
            "Parsed '%s' — skills=%d, exp_years=%.1f, edu=%s",
            filename, len(skills), total_years, highest_edu.value,
        )

        return ParsedResume(
            candidate_id=str(uuid.uuid4()),
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
        for m in _PHONE_RE.finditer(text):
            candidate = m.group(0).strip()
            if _YEAR_RANGE_ONLY.match(candidate):          # "2018-2022" is not a phone number
                continue
            phone = re.sub(r"[^\d+\-() ]", "", candidate).strip()
            if 7 <= len(re.sub(r"\D", "", phone)) <= 15:
                return phone
        return None

    def _extract_name(self, text: str) -> Optional[str]:
        """First line that looks like a proper name (2–4 capitalised words, no digits)."""
        for line in text.splitlines():
            line = line.strip()
            if not line or len(line) > 60 or "@" in line:
                continue
            if any(sw in line.lower() for sw in _NAME_STOPWORDS):
                continue
            words = line.split()
            alpha = [w for w in words if w.isalpha()]
            if 2 <= len(words) <= 4 and len(alpha) >= 2 and all(w[0].isupper() for w in alpha):
                if not any(c.isdigit() for c in line):
                    return line
        return None

    # ── Experience ─────────────────────────────────────────────────────────────

    def _extract_experience(self, text: str) -> Tuple[List[ExperienceEntry], float]:
        """
        Extract experience entries and estimate total years.
          1. An explicit "X years of experience" statement wins.
          2. Otherwise the union of all employment year-ranges (overlaps counted once;
             ranges that sit next to education keywords are ignored).
        """
        entries: list[ExperienceEntry] = []
        spans: list[Tuple[int, int]] = []
        current_year = datetime.now().year

        for m in _YEAR_RANGE.finditer(text):
            start_year = self._year_from_str(m.group(1))
            end_raw = m.group(2).lower()
            end_year = current_year if end_raw in ("present", "current", "now") else self._year_from_str(end_raw)

            if not (start_year and end_year and end_year >= start_year):
                continue
            if self._looks_like_education(text, m.start()):
                continue

            duration = end_year - start_year
            context = text[max(0, m.start() - 200): m.start()].strip()
            entries.append(ExperienceEntry(
                duration=f"{duration} year{'s' if duration != 1 else ''}",
                years=float(duration),
                title=self._infer_title(context),
            ))
            spans.append((start_year, end_year))

        explicit = _EXPLICIT_YEARS.search(text)
        if explicit:
            total_years = float(explicit.group(1))
        else:
            total_years = float(self._union_length(spans))

        return entries, round(min(total_years, 40.0), 1)

    @staticmethod
    def _union_length(spans: List[Tuple[int, int]]) -> int:
        total, cur_end = 0, None
        cur_start = None
        for s, e in sorted(spans):
            if cur_end is None or s > cur_end:
                if cur_end is not None:
                    total += cur_end - cur_start
                cur_start, cur_end = s, e
            else:
                cur_end = max(cur_end, e)
        if cur_end is not None:
            total += cur_end - cur_start
        return total

    @staticmethod
    def _looks_like_education(text: str, pos: int) -> bool:
        line_start = text.rfind("\n", 0, pos) + 1
        prev_start = text.rfind("\n", 0, max(line_start - 1, 0)) + 1
        line_end = text.find("\n", pos)
        window = text[prev_start: line_end if line_end != -1 else len(text)]
        return bool(_EDU_CONTEXT.search(window))

    def _year_from_str(self, s: str) -> Optional[int]:
        m = _YEAR_SINGLE.search(s)
        return int(m.group(0)) if m else None

    def _infer_title(self, context: str) -> Optional[str]:
        """Look for common job-title patterns in the context string (last match wins)."""
        title_re = re.compile(
            r"\b(senior|junior|lead|principal|staff|chief|head|director|manager|engineer|"
            r"developer|analyst|scientist|architect|consultant|intern|associate|specialist)\b[^,\n]{0,40}",
            re.IGNORECASE,
        )
        matches = list(title_re.finditer(context))
        return matches[-1].group(0).strip().title() if matches else None

    # ── Education ──────────────────────────────────────────────────────────────

    def _extract_education(self, lower_text: str) -> Tuple[List[EducationEntry], EducationLevel]:
        entries: list[EducationEntry] = []
        highest = EducationLevel.NONE

        for level, patterns in _EDUCATION_PATTERNS:
            if any(p.search(lower_text) for p in patterns):
                entries.append(EducationEntry(level=level))
                if EDUCATION_RANK[level] > EDUCATION_RANK[highest]:
                    highest = level

        return entries, highest


# ── Singleton ──────────────────────────────────────────────────────────────────
resume_parser = ResumeParser()
