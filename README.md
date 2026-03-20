# 🎯 Smart Résumé Screener

> **AI-powered résumé ranking with OpenAI embeddings, GPT-4.1 reasoning, and a hybrid multi-signal scoring engine. Portfolio-grade. Production-ready.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

---

## 🔥 Project Overview

Smart Résumé Screener is a **production-grade Applicant Tracking System (ATS)** that uses OpenAI's `text-embedding-3-large` and GPT-4.1 to intelligently rank candidates against a job description with transparent, explainable scoring.

### Why this project stands out

| Feature | What makes it special |
|---|---|
| 🔗 **Semantic Matching** | OpenAI `text-embedding-3-large` vectors + cosine similarity — goes beyond keyword matching |
| 🧠 **Hybrid Scoring** | 5 weighted signals: embedding, keyword, experience, education, bonus skills |
| 🤖 **GPT-4.1 Reasoning** | Human-like explanations: strengths, weaknesses, hiring recommendation |
| 📚 **Skill Ontology** | 200+ alias mappings (`"ML"` → `"Machine Learning"`, `"k8s"` → `"Kubernetes"`) |
| 💾 **Embedding Cache** | SHA-256 keyed disk cache eliminates redundant API calls |
| 📦 **Batch Processing** | Upload and analyse unlimited résumés simultaneously |
| 📊 **Export** | Download ranked results as CSV or structured JSON |
| 🐳 **Docker Ready** | One-command deployment with `docker compose up` |

---

## 🧠 AI Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │           SMART RÉSUMÉ SCREENER              │
                        └─────────────────────────────────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
    ┌─────────▼──────────┐      ┌───────────▼──────────┐      ┌──────────▼──────────┐
    │  📄 Input Layer     │      │  🧹 Preprocessing    │      │  🔑 API Layer        │
    │                    │      │                      │      │                     │
    │ PDF → pdfplumber   │      │ Text extraction      │      │ POST /upload-resumes│
    │ DOCX → python-docx │      │ Lowercasing          │      │ POST /analyze       │
    │ Text → direct      │      │ Skill ontology NLP   │      │ GET  /results       │
    │ Drag & drop UI     │      │ Regex entity extract │      │ GET  /export        │
    └────────────────────┘      └──────────────────────┘      └─────────────────────┘
                                            │
                        ┌───────────────────┼───────────────────┐
                        │                   │                   │
            ┌───────────▼────────┐ ┌────────▼────────┐ ┌───────▼──────────┐
            │ 🔗 Embeddings      │ │ 🏷️ NLP Parsing  │ │ 📊 Hybrid Scorer │
            │                   │ │                 │ │                  │
            │ OpenAI             │ │ Name/Email/     │ │ [50%] Embedding  │
            │ text-embedding     │ │ Phone extract   │ │ [20%] Keywords   │
            │ -3-large           │ │ Skill extract   │ │ [15%] Experience │
            │                   │ │ Exp year detect │ │ [10%] Education  │
            │ Cosine similarity  │ │ Edu level parse │ │ [5%]  Bonus      │
            │ Disk cache (SHA256)│ │                 │ │                  │
            └────────────────────┘ └─────────────────┘ └──────────────────┘
                        │                                        │
                        └──────────────────┬─────────────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │  🤖 GPT-4.1 Reasoning  │
                               │                        │
                               │  System: Senior        │
                               │  Technical Recruiter   │
                               │                        │
                               │  → Summary             │
                               │  → Strengths           │
                               │  → Weaknesses          │
                               │  → Missing Skills      │
                               │  → Recommendation      │
                               │  → Fit Level           │
                               └────────────────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │  📊 Ranked Results     │
                               │                        │
                               │  Sorted by final score │
                               │  + CSV/JSON export     │
                               │  + Interactive UI      │
                               └────────────────────────┘
```

### Scoring Formula

```
Final Score = (Embedding × 0.50)
            + (Keyword Match × 0.20)
            + (Experience Alignment × 0.15)
            + (Education Match × 0.10)
            + (Bonus Skills × 0.05)
```

| Signal | Weight | Method |
|---|---|---|
| **Embedding Similarity** | 50% | Cosine similarity between JD and résumé OpenAI embeddings |
| **Keyword Match** | 20% | % of JD skills found in résumé via 200-node skill ontology |
| **Experience Alignment** | 15% | Candidate years vs. JD required seniority level |
| **Education Match** | 10% | Candidate degree vs. JD minimum education requirement |
| **Bonus Skills** | 5% | Premium in-demand skills (Docker, Kubernetes, LLMs, etc.) |

---

## 📁 Project Structure

```
smart-resume-screener/
│
├── main.py                         # FastAPI entrypoint
│
├── app/
│   ├── api/
│   │   ├── health.py               # GET /api/health
│   │   ├── resumes.py              # POST/GET/DELETE /api/resumes
│   │   ├── analysis.py             # POST /api/analysis/run
│   │   └── results.py              # GET /api/results/{id}
│   │
│   ├── core/
│   │   ├── config.py               # Pydantic settings (env-driven)
│   │   └── logger.py               # Structured logging
│   │
│   ├── models/
│   │   └── schemas.py              # All Pydantic v2 data models
│   │
│   ├── services/
│   │   ├── extractor.py            # PDF + DOCX text extraction
│   │   ├── parser.py               # NLP-based résumé structuring
│   │   ├── embedder.py             # OpenAI embeddings + disk cache
│   │   ├── scorer.py               # Hybrid 5-signal scoring engine
│   │   ├── reasoner.py             # GPT-4.1 explanation generator
│   │   └── analyzer.py             # Pipeline orchestrator
│   │
│   └── utils/
│       └── skill_ontology.py       # 200+ skill alias → canonical mappings
│
├── frontend/
│   └── index.html                  # Premium single-page UI (vanilla JS)
│
├── data/
│   ├── sample_resumes/             # 5 ready-to-use sample résumés
│   │   ├── aisha_okonkwo_ml_engineer.txt
│   │   ├── daniel_ferreira_backend_ml.txt
│   │   ├── priya_venkataraman_nlp_phd.txt
│   │   ├── marcus_johnson_entry_level.txt
│   │   └── yuki_tanaka_mlops_engineer.txt
│   │
│   └── job_descriptions/
│       └── senior_ml_engineer_jd.txt
│
├── tests/
│   └── test_screener.py            # 30+ unit + integration tests
│
├── exports/                        # CSV/JSON exports go here
├── demo.py                         # Rich terminal demo script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- ~300 MB disk space

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/smart-resume-screener.git
cd smart-resume-screener
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Optional: download spaCy English model for enhanced NLP
python -m spacy download en_core_web_sm
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-key-here
```

> 💡 **Cost note:** Embedding 5 résumés + 1 JD costs ~$0.001 with `text-embedding-3-large`. GPT-4.1 explanations cost ~$0.02–0.05 per run. Caching means repeated runs are free.

### 5. Create required directories

```bash
mkdir -p data/uploads data/cache exports
```

---

## ▶️ How to Run Locally

### Option A: Backend API only

```bash
python main.py
```

Open: **http://localhost:8000/api/docs** (Swagger UI)

### Option B: Run the demo script

```bash
python demo.py
```

This runs the full pipeline on the 5 sample résumés and prints a rich terminal ranking table.

### Option C: Use the web frontend

1. Start the backend: `python main.py`
2. Open `frontend/index.html` in your browser
3. Click **⚙️ API Settings** → set URL to `http://localhost:8000`
4. Upload résumés, paste the JD, click **Run AI Screening**

### Option D: Docker

```bash
# Copy your .env file first
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

docker compose up --build
```

---

## 🔑 API Reference

All endpoints are prefixed with `/api`. Full interactive docs at `/api/docs`.

### Upload Résumés

```http
POST /api/resumes/upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx, ...]
```

**Response:**
```json
[
  {
    "candidate_id": "uuid-here",
    "filename": "john_doe.pdf",
    "status": "success",
    "message": "Parsed successfully. Skills found: 24"
  }
]
```

### Run Analysis

```http
POST /api/analysis/run
Content-Type: application/json

{
  "job_description": "Senior ML Engineer with 5+ years...",
  "candidate_ids": null
}
```

**Response:** Full `AnalysisSession` with ranked `CandidateResult[]`

### Get Results

```http
GET /api/results/{session_id}
```

### Export Results

```http
GET /api/results/{session_id}/export?format=csv
GET /api/results/{session_id}/export?format=json
```

### List Résumés

```http
GET /api/resumes/
```

### Health Check

```http
GET /api/health
```

---

## 📊 Example Output

```
╭─────────────────────────────── Candidate Rankings ───────────────────────────────╮
│ Rank  Name                     Score   Emb%  KW%   Exp%  Edu%  Fit Level        │
│────────────────────────────────────────────────────────────────────────────────── │
│ #1    Aisha Okonkwo             87.3%   91.2  85.0  90.0  90.0  Strong Fit       │
│ #2    Priya Venkataraman        82.1%   88.5  75.0  80.0 100.0  Strong Fit       │
│ #3    Yuki Tanaka               74.6%   82.1  65.0  90.0  80.0  Strong Fit       │
│ #4    Daniel Ferreira           58.4%   65.3  50.0  60.0  80.0  Moderate Fit     │
│ #5    Marcus Johnson            24.1%   28.7  10.0  40.0  75.0  Weak Fit         │
╰──────────────────────────────────────────────────────────────────────────────────╯

#1 Aisha Okonkwo
  Score: 87.3% | Fit: Strong Fit

  "Senior ML Engineer with 7 years of production AI experience, deep expertise
  in NLP and MLOps. Skill overlap with JD is excellent across Python, PyTorch,
  LangChain, Kubernetes, and MLflow. Strong indicator of immediate productivity."

  ✓ Strengths: PyTorch · MLOps · Kubernetes · LangChain · Hugging Face
  ✗ Gaps: None significant
  → Strongly recommended for interview.
```

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test class
pytest tests/ -v -k "TestSkillOntology"
pytest tests/ -v -k "TestResumeParser"
pytest tests/ -v -k "TestHybridScorer"

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

The test suite includes **30+ tests** covering:
- Skill ontology: canonicalisation, alias resolution, overlap scoring
- Resume parser: email/phone/name extraction, education levels, experience years
- Hybrid scorer: all 5 scoring components
- Integration: full pipeline with mocked OpenAI calls (no API key required for tests)
- Schema validation: Pydantic model constraints

---

## 🚀 Future Improvements

| Feature | Status | Notes |
|---|---|---|
| Vector database (FAISS) | 🔲 Planned | Replace in-memory store with FAISS for scale |
| Authentication | 🔲 Planned | JWT + OAuth2 for multi-user support |
| PostgreSQL persistence | 🔲 Planned | Replace in-memory store |
| Bias detection | 🔲 Planned | Flag potentially biased scoring patterns |
| Real-time WebSocket updates | 🔲 Planned | Live progress during analysis |
| spaCy NER integration | 🔲 Planned | Deeper entity extraction (companies, titles) |
| Résumé anonymisation | 🔲 Planned | Strip PII for blind screening |
| Cloud deployment (AWS/GCP) | 🔲 Planned | Terraform IaC included |
| Multi-language support | 🔲 Planned | Non-English résumés |

---

## 🏗️ Design Decisions

**Why `text-embedding-3-large`?**
It's OpenAI's highest-quality embedding model, providing 3072-dimensional vectors that capture deep semantic nuance — critical for understanding that "ML Engineer" and "Machine Learning Scientist" are similar roles.

**Why a hybrid scorer instead of just embeddings?**
Embeddings alone can't reliably extract "5 years of Python experience" or "PhD in Computer Science." The hybrid approach combines semantic understanding with structured rule-based signals for more reliable, interpretable scoring.

**Why a skill ontology?**
Real résumés are inconsistent — "ML", "machine learning", "Machine Learning", and "deep learning" all appear in the wild. The ontology normalises these to canonical names before computing overlap, dramatically improving keyword match accuracy.

**Why disk caching for embeddings?**
The same résumé being re-analysed against a different JD doesn't need a new embedding. SHA-256 keyed JSON caching reduces API costs by ~80% for repeated runs.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

Built with [FastAPI](https://fastapi.tiangolo.com/), [OpenAI Python SDK](https://github.com/openai/openai-python), [pdfplumber](https://github.com/jsvine/pdfplumber), [python-docx](https://python-docx.readthedocs.io/), and [Pydantic](https://docs.pydantic.dev/).
