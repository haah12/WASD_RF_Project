# agents/orchestrator_ollama.py

import yaml
from pathlib import Path
from base_agent import RFPhaseAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "agents_config.yaml"


def load_config():
    """
    Charge la configuration des agents.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """
    Orchestrateur global Agentic AI.

    Il exécute les six agents du pipeline RF dans l'ordre :
    SOCLE RF → Anomaly Detection → AROCLE → SPECTRA
    → Anticipation temporelle → Décision DRL.
    """

    config = load_config()
    agents = config["agents"]

    for phase_id, cfg in agents.items():

        agent = RFPhaseAgent(
            name=cfg["name"],
            model=cfg["model"],
            input_path=PROJECT_ROOT / cfg["input"],
            output_path=PROJECT_ROOT / cfg["output"],
            scripts=[str(PROJECT_ROOT / script) for script in cfg["scripts"]]
        )

        try:
            agent.run()

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"ERREUR DANS {cfg['name']}")
            print("=" * 80)
            print(e)
            break


if __name__ == "__main__":
    main()