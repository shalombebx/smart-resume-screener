# Debug report — Smart Résumé Screener

## How this was verified
- Every `.py` file passes `python -m compileall`; the JavaScript in `frontend/index.html` passes `node --check`.
- `tests/test_screener.py` (38 original + 9 new regression tests = **47**) all pass.
- **Limitation:** the sandbox had no network, so `fastapi`, `pydantic`, `openai` and `pytest` could not be installed.
  The 47 tests ran against a small stand-in for Pydantic/pytest, so *real* Pydantic validation, `tests/test_api.py`
  (needs FastAPI) and live OpenAI calls were **not** executed. Run `pip install -r requirements.txt && pytest tests/ -v` once locally.
- Not provided in the upload, so not touched: `data/sample_resumes/*`, `data/job_descriptions/*`.

## Crash-level bugs (app could not start or run)
| File | Bug | Fix |
|---|---|---|
| `schemas.py` | Written for a different design: no `AnalysisSession`, `AnalyzeRequest`, `AIExplanation`, `FitLevel`, `EducationLevel`, `ExperienceEntry`, `EducationEntry`, `ExportRequest`; `ParsedResume`/`UploadResponse`/`HealthResponse` had different fields than every consumer → `ImportError` on startup | Rewritten as the one contract all modules/tests/frontend use |
| `main.py` | `serve_spa` mis-indented → `SyntaxError`; `HTTPException` never imported; served a non-existent `frontend/dist` | Serves `frontend/index.html` at `/`; lifespan instead of deprecated `on_event` |
| `config.py` | `UPLOAD_DIR`, `EXPORT_DIR`, `ALLOWED_EXTENSIONS` missing, but used by `resumes.py`/`results.py` → `AttributeError` | Added |
| `skill_ontology.py` | `extract_skills_from_text`, `keyword_match_score`, `skills_overlap`, `canonicalise` imported everywhere but not defined | Implemented (word-boundary regex scan) |
| `test_screener.py` | Mocked explanation was a `MagicMock`; `CandidateResult` validation rejects it | Uses a real `AIExplanation`; fresh analyzer per test |
| `requirements.txt` | `PyPDF2` (extractor fallback) missing; `rich`/`pytest` missing | Added; removed unused deps (nltk, scikit-learn, numpy, pandas, diskcache, aiofiles, tenacity, structlog) |
| repo | `Dockerfile` and `.env.example` referenced by README/compose but absent | Added |

## Logic bugs
- **parser**: education matched by raw substrings (`"ond"` in "second", `"master"` in "Scrum Master") → word-bounded regexes; study dates ("2014–2018, University…") were counted as work experience; overlapping jobs double-counted (now merged); "Present" hard-coded to 2025; phone regex matched year ranges and crossed newlines.
- **scorer**: a JD saying "mentor *junior* engineers" was read as entry-level (1 yr) even with "5+ years"; "Bachelor's **or** Master's" demanded a Master's; bonus list had names that can never match (`"GPT / OpenAI"`, `"Generative AI"`); `"1+"` matched inside `"11+"`.
- **analyzer/embedder**: JD re-embedded per résumé when cache is off; empty text sent to the API; `CACHE_TTL_SECONDS` was ignored; unused JD map removed.
- **ontology**: ambiguous aliases (`go`, `cv`, `ann`, `cache`, `express`, …) caused false skill hits when scanning text (they still work in `canonicalise()`). Use "Golang" to detect Go.
- **extractor**: fallback could run twice; `-\n` regex joined every hyphen at a line end.
- **resumes.py**: temp file leaked on error after rename; private `_store` poked from the router (added `remove_resume`); `MAX_RESUMES_PER_BATCH` never enforced.
- **analysis.py**: blocking OpenAI calls ran inside `async def` (froze the server) → plain `def`; raw exception text was returned to clients.
- **reasoner**: crashed on non-dict / non-list model output (now coerced); prompt now tells the model to ignore instructions inside résumés and protected traits.

## Security
- `index.html`: résumé-derived text (names, skills, filenames, GPT output) was injected via `innerHTML` → **stored XSS**. All dynamic values now pass through `esc()`.
- CSV export: **formula injection** (`=…`, `+…`, `@…` in names/summaries) neutralised; UTF-8 BOM added for Excel.
- `demo.py`: no longer prints the first 12 characters of the API key.

## Smaller items
`datetime.utcnow()` → timezone-aware; logger duplicate lines (`propagate=False`) and `DEBUG` level honoured; `.dockerignore` leading spaces; `.gitignore` now ignores `.cache/`; compose `version:` key removed; README corrections (FastAPI 0.111, GPT-4.1 → configurable model, endpoint names in the diagram, "unlimited" batch, "Terraform included", "~80% savings", skill counts, how to open the frontend). The README "Example Output" numbers are illustrative, not measured.
