# agents/final_report_agent.py

from pathlib import Path
import ollama


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "agent_reports"
FINAL_REPORT = PROJECT_ROOT / "outputs" / "rapport_global_agentic_ai.md"


def main():
    reports = []

    for path in sorted(REPORTS_DIR.glob("report_*.md")):
        reports.append(f"\n\n# {path.stem}\n")
        reports.append(path.read_text(encoding="utf-8"))

    content = "\n".join(reports)

    prompt = f"""
Tu es un agent scientifique chargé de synthétiser un pipeline de thèse

Voici les rapports des agents RF :

{content}

Rédige un rapport global structuré avec :
1. la vision générale du système,
2. le rôle de chaque agent,
3. la chaîne RF complète,
4. les résultats principaux,
5. l'intérêt de l'approche Agentic AI + Ollama,
6. une conclusion adaptée à une thèse de doctorat.
"""

    response = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    FINAL_REPORT.write_text(
        response["message"]["content"],
        encoding="utf-8"
    )

    print("Rapport global généré :", FINAL_REPORT)


if __name__ == "__main__":
    main()