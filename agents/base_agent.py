# agents/base_agent.py

import sys
import subprocess
from pathlib import Path

import pandas as pd
import ollama


class RFPhaseAgent:
    """
    Agent générique pour exécuter une phase du pipeline RF.
    """

    def __init__(self, name, model, input_path, output_path, scripts):
        self.name = name
        self.model = model
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.scripts = scripts

    def check_input(self):
        """
        Vérifie que l'entrée nécessaire à la phase existe.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"[{self.name}] Entrée introuvable : {self.input_path}"
            )

    def run_scripts(self):
        """
        Exécute les scripts associés à la phase avec le même Python
        que celui utilisé par l'environnement actif.
        """

        print(f"[{self.name}] Python utilisé : {sys.executable}")

        for script in self.scripts:
            script_path = Path(script)

            if not script_path.exists():
                raise FileNotFoundError(
                    f"[{self.name}] Script introuvable : {script_path}"
                )

            print(f"\n[{self.name}] Exécution du script : {script_path}")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(result.stdout)

            if result.returncode != 0:
                raise RuntimeError(
                    f"[{self.name}] Erreur dans {script_path}\n{result.stderr}"
                )

    def summarize_output(self):
        """
        Prépare un résumé simple de la sortie produite par la phase.
        """

        if self.output_path.suffix == ".csv" and self.output_path.exists():
            df = pd.read_csv(self.output_path, sep=";", decimal=",")

            return {
                "shape": df.shape,
                "columns": list(df.columns),
                "preview": df.head(5).to_string()
            }

        return {
            "shape": "N/A",
            "columns": [],
            "preview": "Sortie non tabulaire ou modèle sauvegardé."
        }

    def analyze_with_ollama(self):
        """
        Demande au modèle Ollama d'interpréter scientifiquement la sortie.
        """

        summary = self.summarize_output()

        prompt = f"""
Tu es {self.name}, un agent spécialisé en cybersécurité proactive RF.

Analyse la sortie de cette phase.

Fichier de sortie :
{self.output_path}

Dimensions :
{summary["shape"]}

Colonnes :
{summary["columns"]}

Aperçu :
{summary["preview"]}

Donne une analyse structurée avec :
1. le rôle scientifique de cette phase,
2. les résultats produits,
3. les éléments RF importants,
4. les éventuels risques ou anomalies,
5. la transition vers la phase suivante.

Réponds en français, avec un style clair et adapté à un rapport de thèse.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]

    def save_report(self, report):
        """
        Sauvegarde le rapport généré par l'agent.
        """

        reports_dir = Path("outputs") / "agent_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"report_{self.name}.md"
        report_path.write_text(report, encoding="utf-8")

        print(f"[{self.name}] Rapport sauvegardé : {report_path}")

    def run(self):
        """
        Exécution complète d'un agent.
        """

        print("\n" + "=" * 80)
        print(f"AGENT EN COURS : {self.name}")
        print("=" * 80)

        self.check_input()
        self.run_scripts()

        report = self.analyze_with_ollama()
        self.save_report(report)

        return report