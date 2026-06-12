# phase5_compute_anticipation_metrics.py

import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_anticipation_v5.csv"


# Durée moyenne d’une fenêtre RF en millisecondes.
# Cette valeur permet de convertir le délai d’anticipation
# depuis un nombre de fenêtres vers une durée physique.
TIME_WINDOW_MS = 5.2897


def main():
    """
    Calcule le délai d’anticipation entre une alerte pré-attaque
    et l’apparition réelle d’un état d’attaque.

    La métrique principale est :
    délai = fenêtre_attack - fenêtre_pre_attack
    """

    # =========================
    # Chargement des résultats Phase 5
    # =========================

    df = pd.read_csv(CSV_IN, sep=";", decimal=",")
    df.columns = df.columns.str.strip()

    delays = []

    # =========================
    # Analyse fichier par fichier
    # =========================

    for file_id, group in df.groupby("file_id"):

        group = group.sort_values("window_id").reset_index(drop=True)

        # Fenêtres où le modèle prédit une situation de pré-attaque
        pre_attack_idx = np.where(group["pred_state"] == 1)[0]

        # Fenêtres où l’état réel correspond à une attaque
        attack_idx = np.where(group["true_state"] == 2)[0]

        for t_attack in attack_idx:

            # On cherche uniquement les alertes apparues avant l’attaque.
            valid_alerts = pre_attack_idx[pre_attack_idx < t_attack]

            if len(valid_alerts) == 0:
                continue

            # Dernière alerte pré-attaque avant l’attaque réelle
            t_alert = valid_alerts[-1]
            delay_steps = t_attack - t_alert

            if delay_steps > 0:
                delays.append(delay_steps)

    # =========================
    # Résumé des résultats
    # =========================

    print("\n===== Anticipation PRE-ATTACK -> ATTACK =====")
    if len(delays) == 0:
        print("Aucune anticipation détectée.")
        return

    delays = np.array(delays)

    print("Délai moyen (fenêtres) :", np.mean(delays))
    print("Délai médian (fenêtres) :", np.median(delays))
    print("Délai maximal (fenêtres) :", np.max(delays))

    print("\nDélai moyen d’anticipation (ms) :", np.mean(delays) * TIME_WINDOW_MS)

    print("\nDistribution des délais :")
    print(pd.Series(delays).value_counts().sort_index())


if __name__ == "__main__":
    main()