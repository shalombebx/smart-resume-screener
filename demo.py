#!/usr/bin/env python3
"""
demo.py — Smart Résumé Screener · Demo Script
═══════════════════════════════════════════════
Runs a complete résumé screening pipeline against all 5 sample résumés
and the included job description, then prints a rich terminal report.

Usage:
    python demo.py

Requirements:
    OPENAI_API_KEY must be set in .env

What this script does:
    1. Loads sample résumés from data/sample_resumes/
    2. Loads the sample job description
    3. Parses and extracts structured data from each résumé
    4. Runs the full hybrid scoring pipeline
    5. Calls OpenAI for embeddings + GPT-4.1 explanations
    6. Prints a formatted ranking table
    7. Exports results to exports/demo_results.csv
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime

# ── Ensure project root is on path ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# Load .env
from dotenv import load_dotenv
load_dotenv()

# ── Imports ────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.text    import Text
    from rich         import box
    RICH = True
except ImportError:
    RICH = False

from app.services.extractor import extractor
from app.services.parser    import resume_parser
from app.services.analyzer  import resume_analyzer
from app.core.config        import settings

console = Console() if RICH else None

def log(msg: str, style: str = ""):
    if RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


def main():
    log("\n[bold cyan]🎯  Smart Résumé Screener — Demo[/bold cyan]" if RICH else "\n🎯  Smart Résumé Screener — Demo")
    log("=" * 60)

    # ── 1. Verify OpenAI key ───────────────────────────────────────────────────
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("sk-your"):
        log("[bold red]❌ OPENAI_API_KEY is not set in .env[/bold red]" if RICH else "❌ OPENAI_API_KEY is not set in .env")
        log("   Add your key to .env:  OPENAI_API_KEY=sk-...")
        sys.exit(1)

    log(f"✅ OpenAI API key detected ({settings.OPENAI_API_KEY[:12]}…)")
    log(f"   Embedding model : {settings.OPENAI_EMBEDDING_MODEL}")
    log(f"   Chat model      : {settings.OPENAI_CHAT_MODEL}")

    # ── 2. Load résumés ────────────────────────────────────────────────────────
    resume_dir = Path("data/sample_resumes")
    resume_files = list(resume_dir.glob("*.txt")) + list(resume_dir.glob("*.pdf")) + list(resume_dir.glob("*.docx"))

    if not resume_files:
        log("❌ No sample résumés found in data/sample_resumes/")
        sys.exit(1)

    log(f"\n📂 Loading {len(resume_files)} résumé(s) from {resume_dir}/")

    for fpath in resume_files:
        try:
            # For .txt files, read directly; for PDF/DOCX use extractor
            if fpath.suffix == ".txt":
                raw_text = fpath.read_text(encoding="utf-8")
            else:
                raw_text = extractor.extract(fpath)

            parsed = resume_parser.parse(raw_text, fpath.name)
            resume_analyzer.add_resume(parsed)

            log(
                f"   ✓ [green]{fpath.name}[/green] — "
                f"{parsed.full_name or 'Unknown'} | "
                f"{len(parsed.skills)} skills | "
                f"{parsed.total_experience_years}yr exp | "
                f"{parsed.highest_education.value}"
                if RICH else
                f"   ✓ {fpath.name} — {parsed.full_name or 'Unknown'} | "
                f"{len(parsed.skills)} skills | {parsed.total_experience_years}yr exp"
            )
        except Exception as e:
            log(f"   ✗ Failed to load {fpath.name}: {e}")

    # ── 3. Load job description ───────────────────────────────────────────────
    jd_path = Path("data/job_descriptions/senior_ml_engineer_jd.txt")
    if not jd_path.exists():
        log(f"❌ Job description not found at {jd_path}")
        sys.exit(1)

    jd_text = jd_path.read_text(encoding="utf-8")
    log(f"\n📋 Job Description loaded: {jd_path.name} ({len(jd_text)} chars)")

    # ── 4. Run analysis ────────────────────────────────────────────────────────
    log("\n🔍 Running AI screening pipeline…")
    log("   This will call OpenAI for embeddings and GPT-4 explanations.")
    log("   (First run may take 30–60 seconds; subsequent runs use cache)\n")

    try:
        session = resume_analyzer.analyze(job_description=jd_text)
    except Exception as e:
        log(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 5. Print results ───────────────────────────────────────────────────────
    log(f"\n✅ Analysis complete in {session.processing_time_seconds}s")
    log(f"   Session ID: {session.session_id[:12]}…\n")

    if RICH and console:
        table = Table(
            title="📊 Candidate Rankings",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Rank",    style="bold", width=6)
        table.add_column("Name",    style="bold", width=24)
        table.add_column("Score",   style="bold yellow", width=8)
        table.add_column("Emb %",   width=7)
        table.add_column("KW %",    width=7)
        table.add_column("Exp %",   width=7)
        table.add_column("Edu %",   width=7)
        table.add_column("Fit Level", width=16)
        table.add_column("Matching Skills", width=30)

        rank_colors = {1: "gold1", 2: "grey74", 3: "orange3"}

        for c in session.candidates:
            fit   = c.ai_explanation.fit_level.value if c.ai_explanation else "—"
            color = "green" if "Strong" in fit else "yellow" if "Moderate" in fit else "red"
            rank_style = rank_colors.get(c.rank, "white")

            table.add_row(
                f"[{rank_style}]#{c.rank}[/{rank_style}]",
                c.full_name or c.filename[:20],
                f"{c.score.final_score:.1f}%",
                f"{c.score.embedding_similarity:.1f}",
                f"{c.score.keyword_match:.1f}",
                f"{c.score.experience_alignment:.1f}",
                f"{c.score.education_match:.1f}",
                f"[{color}]{fit}[/{color}]",
                ", ".join(c.matching_skills[:4]) or "—",
            )

        console.print(table)

        # Print detailed AI analysis for top 3
        log("\n[bold]🧠 AI Analysis — Top 3 Candidates[/bold]")
        for c in session.candidates[:3]:
            ai = c.ai_explanation
            if not ai:
                continue
            fit_color = "green" if "Strong" in ai.fit_level.value else "yellow" if "Moderate" in ai.fit_level.value else "red"
            panel_content = (
                f"[bold]Score:[/bold] {c.score.final_score:.1f}%  |  "
                f"[bold]Fit:[/bold] [{fit_color}]{ai.fit_level.value}[/{fit_color}]\n\n"
                f"{ai.summary}\n\n"
                f"[bold green]Strengths:[/bold green] {' · '.join(ai.strengths[:3])}\n"
                f"[bold red]Weaknesses:[/bold red] {' · '.join(ai.weaknesses[:2]) or 'None identified'}\n\n"
                f"[bold blue]→ {ai.recommendation}[/bold blue]"
            )
            console.print(Panel(
                panel_content,
                title=f"#{c.rank} {c.full_name or c.filename}",
                border_style="cyan",
                padding=(1, 2),
            ))
    else:
        # Plain text fallback
        print(f"\n{'Rank':<6} {'Name':<25} {'Score':>7} {'Fit Level':<16}")
        print("-" * 60)
        for c in session.candidates:
            fit = c.ai_explanation.fit_level.value if c.ai_explanation else "—"
            print(f"#{c.rank:<5} {(c.full_name or c.filename):<25} {c.score.final_score:>6.1f}% {fit:<16}")

    # ── 6. Export to CSV ───────────────────────────────────────────────────────
    export_dir  = Path("exports")
    export_dir.mkdir(exist_ok=True)
    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    export_path = export_dir / f"demo_results_{timestamp}.csv"

    with export_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank","Name","Email","Score","Embedding","Keyword","Experience","Education","Bonus","Fit","Recommendation"])
        for c in session.candidates:
            writer.writerow([
                c.rank,
                c.full_name or "",
                c.email or "",
                f"{c.score.final_score:.1f}",
                f"{c.score.embedding_similarity:.1f}",
                f"{c.score.keyword_match:.1f}",
                f"{c.score.experience_alignment:.1f}",
                f"{c.score.education_match:.1f}",
                f"{c.score.bonus_skills:.1f}",
                c.ai_explanation.fit_level.value if c.ai_explanation else "",
                c.ai_explanation.recommendation  if c.ai_explanation else "",
            ])

    log(f"\n💾 Results exported to: {export_path}")
    log("\n[bold green]Demo complete! 🎉[/bold green]" if RICH else "\nDemo complete! 🎉")


if __name__ == "__main__":
    main()
