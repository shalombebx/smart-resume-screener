"""
app/utils/skill_ontology.py
────────────────────────────
Canonical skill groups → normalises aliases so
"ML" and "Machine Learning" count as the same skill.
"""
from __future__ import annotations

# Each entry: canonical_name → set of known aliases (all lower-case)
SKILL_GROUPS: dict[str, set[str]] = {
    # ── Programming Languages ─────────────────────────────────────────────────
    "Python": {"python", "py"},
    "JavaScript": {"javascript", "js", "es6", "es2015"},
    "TypeScript": {"typescript", "ts"},
    "Java": {"java", "java 8", "java 11", "java 17"},
    "C++": {"c++", "cpp", "c plus plus"},
    "C#": {"c#", "csharp", "c sharp", ".net"},
    "Go": {"golang", "go"},
    "Rust": {"rust", "rust lang"},
    "Kotlin": {"kotlin"},
    "Swift": {"swift", "swiftui"},
    "R": {"r language", "rlang"},
    "SQL": {"sql", "t-sql", "pl/sql", "mysql", "postgresql", "postgres", "sqlite"},
    "Bash": {"bash", "shell", "shell scripting", "zsh"},

    # ── Machine Learning / AI ─────────────────────────────────────────────────
    "Machine Learning": {"machine learning", "ml", "supervised learning", "unsupervised learning"},
    "Deep Learning": {"deep learning", "dl", "neural networks", "neural nets", "ann", "dnn"},
    "NLP": {"nlp", "natural language processing", "text mining", "text analytics"},
    "Computer Vision": {"computer vision", "cv", "image recognition", "object detection"},
    "Reinforcement Learning": {"reinforcement learning", "rl", "q-learning"},
    "LLMs": {"llm", "llms", "large language models", "gpt", "chatgpt", "openai"},
    "Transformers": {"transformers", "bert", "gpt-2", "gpt-3", "t5", "roberta"},
    "MLOps": {"mlops", "ml ops", "model deployment", "model serving"},

    # ── ML Frameworks ─────────────────────────────────────────────────────────
    "TensorFlow": {"tensorflow", "tf", "keras"},
    "PyTorch": {"pytorch", "torch"},
    "Scikit-learn": {"scikit-learn", "sklearn", "scikit learn"},
    "XGBoost": {"xgboost", "xgb", "gradient boosting"},
    "LightGBM": {"lightgbm", "lgbm"},
    "Hugging Face": {"hugging face", "huggingface", "transformers library"},

    # ── Data Engineering ──────────────────────────────────────────────────────
    "Pandas": {"pandas", "dataframes"},
    "NumPy": {"numpy", "np"},
    "Spark": {"apache spark", "pyspark", "spark"},
    "Kafka": {"apache kafka", "kafka"},
    "Airflow": {"apache airflow", "airflow"},
    "dbt": {"dbt", "data build tool"},
    "ETL": {"etl", "elt", "data pipeline", "data pipelines"},

    # ── Cloud Platforms ───────────────────────────────────────────────────────
    "AWS": {"aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker"},
    "Azure": {"azure", "microsoft azure", "azure ml"},
    "GCP": {"gcp", "google cloud", "google cloud platform", "bigquery"},

    # ── DevOps / Infra ────────────────────────────────────────────────────────
    "Docker": {"docker", "containerization", "containers"},
    "Kubernetes": {"kubernetes", "k8s", "helm"},
    "CI/CD": {"ci/cd", "cicd", "github actions", "jenkins", "gitlab ci", "circleci"},
    "Terraform": {"terraform", "iac", "infrastructure as code"},

    # ── Databases ─────────────────────────────────────────────────────────────
    "MongoDB": {"mongodb", "mongo", "nosql"},
    "Redis": {"redis", "cache"},
    "Elasticsearch": {"elasticsearch", "elastic", "opensearch"},
    "Snowflake": {"snowflake"},
    "Databricks": {"databricks"},

    # ── Web Frameworks ────────────────────────────────────────────────────────
    "FastAPI": {"fastapi", "fast api"},
    "Django": {"django"},
    "Flask": {"flask"},
    "React": {"react", "reactjs", "react.js"},
    "Node.js": {"node", "nodejs", "node.js", "express"},

    # ── Soft / Domain Skills ──────────────────────────────────────────────────
    "Statistics": {"statistics", "statistical analysis", "probability", "bayesian"},
    "Data Visualization": {"data visualization", "tableau", "power bi", "matplotlib", "plotly", "seaborn"},
    "Agile": {"agile", "scrum", "kanban", "jira"},
    "Git": {"git", "github", "gitlab", "version control"},
}

# Inverted index: alias → canonical name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in SKILL_GROUPS.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical


def normalize_skill(raw: str) -> str:
    """Return canonical skill name, or the original (title-cased) if unknown."""
    return _ALIAS_TO_CANONICAL.get(raw.strip().lower(), raw.strip().title())


def normalize_skill_list(skills: list[str]) -> list[str]:
    """Normalize + deduplicate a list of raw skill strings."""
    seen: set[str] = set()
    result: list[str] = []
    for s in skills:
        canonical = normalize_skill(s)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result
