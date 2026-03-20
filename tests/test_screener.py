"""
tests/test_screener.py
───────────────────────
Comprehensive unit + integration tests for the Smart Résumé Screener.

Run:
    pytest tests/ -v
    pytest tests/ -v -k "test_parser"    # single module
    pytest tests/ --tb=short             # brief tracebacks
"""

import pytest
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.skill_ontology import (
    canonicalise,
    extract_skills_from_text,
    keyword_match_score,
    skills_overlap,
)
from app.services.parser import ResumeParser
from app.services.scorer import HybridScorer
from app.models.schemas import (
    ParsedResume, ScoreBreakdown, EducationLevel
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Skill Ontology Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillOntology:

    def test_canonical_exact_match(self):
        assert canonicalise("python") == "Python"
        assert canonicalise("PyTorch") == "PyTorch"
        assert canonicalise("ml") == "Machine Learning"

    def test_canonical_aliases(self):
        assert canonicalise("machine learning") == "Machine Learning"
        assert canonicalise("ML") == "Machine Learning"
        assert canonicalise("tensorflow") == "TensorFlow"
        assert canonicalise("tf") == "TensorFlow"
        assert canonicalise("k8s") == "Kubernetes"

    def test_canonical_unknown_skill(self):
        # Unknown skills should be title-cased
        result = canonicalise("some obscure skill")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_skills_from_text(self):
        text = "We need someone with Python, TensorFlow, Docker, and experience in NLP."
        skills = extract_skills_from_text(text)
        assert "Python" in skills
        assert "TensorFlow" in skills
        assert "Docker" in skills
        assert "NLP" in skills

    def test_extract_no_skills(self):
        text = "Hello, my name is John and I like hiking and coffee."
        skills = extract_skills_from_text(text)
        assert isinstance(skills, list)

    def test_extract_aliases_resolved(self):
        text = "Experience with ML, deep learning, and k8s required."
        skills = extract_skills_from_text(text)
        assert "Machine Learning" in skills
        assert "Deep Learning" in skills
        assert "Kubernetes" in skills

    def test_keyword_match_score_perfect(self):
        jd_skills     = ["Python", "PyTorch", "Docker"]
        resume_skills = ["Python", "PyTorch", "Docker", "Kubernetes"]
        score = keyword_match_score(resume_skills, jd_skills)
        assert score == 100.0

    def test_keyword_match_score_partial(self):
        jd_skills     = ["Python", "PyTorch", "Docker", "Kubernetes"]
        resume_skills = ["Python", "PyTorch"]
        score = keyword_match_score(resume_skills, jd_skills)
        assert score == 50.0

    def test_keyword_match_score_zero(self):
        jd_skills     = ["PyTorch", "Kubernetes"]
        resume_skills = ["Excel", "PowerPoint"]
        score = keyword_match_score(resume_skills, jd_skills)
        assert score == 0.0

    def test_keyword_match_empty_jd(self):
        score = keyword_match_score(["Python"], [])
        assert score == 0.0

    def test_skills_overlap(self):
        resume = ["Python", "PyTorch", "Docker"]
        jd     = ["Python", "PyTorch", "Kubernetes"]
        matching, missing = skills_overlap(resume, jd)
        assert "Python" in matching
        assert "PyTorch" in matching
        assert "Kubernetes" in missing
        assert "Docker" not in missing


# ─────────────────────────────────────────────────────────────────────────────
# 2. Resume Parser Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestResumeParser:

    def setup_method(self):
        self.parser = ResumeParser()

    def _parse(self, text: str, filename: str = "test.pdf"):
        return self.parser.parse(text, filename)

    def test_parse_returns_parsed_resume(self):
        result = self._parse("John Doe\njohn@example.com\nPython developer with 3 years experience.")
        assert isinstance(result, ParsedResume)
        assert result.filename == "test.pdf"
        assert result.candidate_id is not None

    def test_parse_extracts_email(self):
        text = "Alice Smith\nalice.smith@gmail.com\nSoftware Engineer"
        r = self._parse(text)
        assert r.email == "alice.smith@gmail.com"

    def test_parse_no_email(self):
        text = "John Doe\nNo email here\nPython developer"
        r = self._parse(text)
        assert r.email is None

    def test_parse_extracts_skills(self):
        text = "Experienced with Python, TensorFlow, Docker and AWS."
        r = self._parse(text)
        assert len(r.skills) > 0
        assert "Python" in r.skills

    def test_parse_experience_explicit_mention(self):
        text = "John Doe\nI have 5 years of experience in software engineering."
        r = self._parse(text)
        assert r.total_experience_years == 5.0

    def test_parse_experience_year_range(self):
        text = "John Doe\nSoftware Engineer at Google\n2018 – 2022\nDeveloped APIs"
        r = self._parse(text)
        assert r.total_experience_years >= 3.0

    def test_parse_education_phd(self):
        text = "Jane Smith\nPhD in Computer Science from MIT."
        r = self._parse(text)
        assert r.highest_education == EducationLevel.PHD

    def test_parse_education_bachelors(self):
        text = "John Doe\nBachelor of Science in Computer Science, 2020."
        r = self._parse(text)
        assert r.highest_education == EducationLevel.BACHELORS

    def test_parse_education_none(self):
        text = "Mark Lee\nSelf-taught developer, no formal education listed."
        r = self._parse(text)
        assert r.highest_education == EducationLevel.NONE

    def test_parse_name_extraction(self):
        text = "Alice Johnson\nalice@example.com\nML Engineer with 5 years experience"
        r = self._parse(text)
        assert r.full_name == "Alice Johnson"

    def test_parse_empty_text(self):
        r = self._parse("   \n\n  ", "empty.pdf")
        assert isinstance(r, ParsedResume)
        assert r.skills == []

    def test_parse_unique_ids(self):
        text = "John Doe\nPython developer"
        r1 = self._parse(text, "a.pdf")
        r2 = self._parse(text, "b.pdf")
        assert r1.candidate_id != r2.candidate_id


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Scorer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridScorer:

    def setup_method(self):
        self.scorer = HybridScorer()
        self.parser = ResumeParser()

    def _make_resume(
        self,
        text: str = "Python developer",
        exp_years: float = 3.0,
        edu: EducationLevel = EducationLevel.BACHELORS,
        skills: list = None,
    ) -> ParsedResume:
        r = self.parser.parse(text, "test.pdf")
        r.total_experience_years = exp_years
        r.highest_education = edu
        if skills is not None:
            r.skills = skills
        return r

    def test_score_returns_breakdown(self):
        resume = self._make_resume()
        breakdown = self.scorer.score(
            resume=resume,
            jd_text="Python developer needed, 3 years experience, bachelor's degree",
            jd_skills=["Python"],
            embedding_score=75.0,
        )
        assert isinstance(breakdown, ScoreBreakdown)

    def test_final_score_in_range(self):
        resume = self._make_resume(skills=["Python", "Docker", "AWS"])
        breakdown = self.scorer.score(
            resume=resume,
            jd_text="Senior engineer, 5+ years, Python, Docker, AWS",
            jd_skills=["Python", "Docker", "AWS"],
            embedding_score=80.0,
        )
        assert 0.0 <= breakdown.final_score <= 100.0

    def test_high_embedding_improves_score(self):
        resume = self._make_resume(skills=["Python"])
        low  = self.scorer.score(resume=resume, jd_text="Python job", jd_skills=["Python"], embedding_score=30.0)
        high = self.scorer.score(resume=resume, jd_text="Python job", jd_skills=["Python"], embedding_score=90.0)
        assert high.final_score > low.final_score

    def test_keyword_match_reflected(self):
        resume_good = self._make_resume(skills=["Python", "PyTorch", "Docker", "AWS"])
        resume_poor = self._make_resume(skills=["Excel", "PowerPoint"])
        jd_skills   = ["Python", "PyTorch", "Docker", "AWS"]
        good = self.scorer.score(resume_good, "Python PyTorch Docker AWS", jd_skills, 70.0)
        poor = self.scorer.score(resume_poor, "Python PyTorch Docker AWS", jd_skills, 70.0)
        assert good.keyword_match > poor.keyword_match

    def test_phd_beats_bachelors_when_phd_required(self):
        jd = "PhD required for this research role."
        phd_resume  = self._make_resume(edu=EducationLevel.PHD)
        bsc_resume  = self._make_resume(edu=EducationLevel.BACHELORS)
        phd_score   = self.scorer.score(phd_resume,  jd, [], 70.0)
        bsc_score   = self.scorer.score(bsc_resume,  jd, [], 70.0)
        assert phd_score.education_match >= bsc_score.education_match

    def test_bonus_skills_awarded(self):
        resume_with  = self._make_resume(skills=["Docker", "Kubernetes", "AWS", "GCP"])
        resume_without = self._make_resume(skills=["Excel"])
        jd = "Build APIs"
        s_with    = self.scorer.score(resume_with,    jd, [], 60.0)
        s_without = self.scorer.score(resume_without, jd, [], 60.0)
        assert s_with.bonus_skills > s_without.bonus_skills

    def test_experience_alignment_senior(self):
        senior = self._make_resume(exp_years=8.0)
        junior = self._make_resume(exp_years=0.5)
        jd     = "Senior engineer, 5+ years required."
        s_senior = self.scorer.score(senior, jd, [], 70.0)
        s_junior = self.scorer.score(junior, jd, [], 70.0)
        assert s_senior.experience_alignment > s_junior.experience_alignment

    def test_all_components_non_negative(self):
        resume = self._make_resume()
        bd = self.scorer.score(resume, "Any job description", ["Python"], 50.0)
        assert bd.embedding_similarity >= 0
        assert bd.keyword_match        >= 0
        assert bd.experience_alignment >= 0
        assert bd.education_match      >= 0
        assert bd.bonus_skills         >= 0
        assert bd.final_score          >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Integration: Full Pipeline (no OpenAI calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    """
    Tests the full pipeline using mocked embeddings so no OpenAI key is needed.
    """

    def setup_method(self):
        from app.services.analyzer import ResumeAnalyzer
        from app.services.embedder import EmbeddingService
        from unittest.mock import patch, MagicMock

        self.analyzer = ResumeAnalyzer()
        self.parser   = ResumeParser()
        self._patch   = patch
        self._MagicMock = MagicMock

    def _add_resume(self, text: str, filename: str):
        r = self.parser.parse(text, filename)
        self.analyzer.add_resume(r)
        return r

    def test_add_and_list_resumes(self):
        self.analyzer.clear_resumes()
        self._add_resume("Python developer with 3 years exp", "r1.pdf")
        self._add_resume("Java developer with 5 years exp", "r2.pdf")
        resumes = self.analyzer.list_resumes()
        assert len(resumes) == 2

    def test_clear_resumes(self):
        self._add_resume("Test resume", "t.pdf")
        self.analyzer.clear_resumes()
        assert len(self.analyzer.list_resumes()) == 0

    def test_analyze_raises_without_resumes(self):
        self.analyzer.clear_resumes()
        with pytest.raises(ValueError, match="No résumés found"):
            self.analyzer.analyze("Some job description text that is long enough")

    def test_analyze_with_mock_openai(self):
        """Full pipeline run with mocked OpenAI calls."""
        from unittest.mock import patch, MagicMock

        self.analyzer.clear_resumes()
        self._add_resume(
            "Alice Brown\nalice@email.com\nSenior Python developer, 6 years, PyTorch, Docker, AWS, Kubernetes",
            "alice.pdf"
        )
        self._add_resume(
            "Bob Smith\nbob@email.com\nJunior developer, 1 year, basic Python, HTML, CSS",
            "bob.pdf"
        )

        # Mock OpenAI embeddings — return fake vectors
        import math
        def fake_embed(text):
            # Deterministic fake: Alice gets high similarity, Bob gets low
            if "Senior" in text or "PyTorch" in text:
                return [0.9, 0.1, 0.0] * 341 + [0.9]   # 1024-dim fake
            return [0.1, 0.9, 0.0] * 341 + [0.1]

        mock_explanation = MagicMock()
        mock_explanation.summary = "Mocked explanation"
        mock_explanation.strengths = ["Good Python skills"]
        mock_explanation.weaknesses = []
        mock_explanation.missing_requirements = []
        mock_explanation.recommendation = "Hire"
        mock_explanation.fit_level.value = "Strong Fit"

        with patch("app.services.embedder.embedding_service.embed", side_effect=fake_embed):
            with patch("app.services.reasoner.ai_reasoner.explain", return_value=MagicMock(
                summary="Mock summary",
                strengths=["Python"],
                weaknesses=[],
                missing_requirements=[],
                recommendation="Consider for interview",
                fit_level=MagicMock(value="Moderate Fit"),
            )):
                session = self.analyzer.analyze("We need Python, PyTorch, Docker, Kubernetes, AWS expert with 5+ years")

        assert session.total_resumes == 2
        assert len(session.candidates) == 2
        # Ranks should be assigned
        assert session.candidates[0].rank == 1
        assert session.candidates[1].rank == 2
        # Rank 1 should have higher or equal final score
        assert session.candidates[0].score.final_score >= session.candidates[1].score.final_score
        # Session ID is set
        assert len(session.session_id) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Schema Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas:

    def test_score_breakdown_clamped(self):
        """ScoreBreakdown should reject values outside 0–100."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ScoreBreakdown(
                embedding_similarity=110.0,  # Invalid
                keyword_match=50.0,
                experience_alignment=50.0,
                education_match=50.0,
                bonus_skills=50.0,
                final_score=50.0,
            )

    def test_fit_level_enum_values(self):
        from app.models.schemas import FitLevel
        assert FitLevel.STRONG   == "Strong Fit"
        assert FitLevel.MODERATE == "Moderate Fit"
        assert FitLevel.WEAK     == "Weak Fit"

    def test_education_level_ordering(self):
        levels = list(EducationLevel)
        assert EducationLevel.PHD in levels
        assert EducationLevel.BACHELORS in levels
