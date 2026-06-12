# phase4_plot_spectra_timeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_risk_scored.csv"
PLOTS_DIR = OUT_DIR / "plots_spectra"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Paramètres graphiques
# =========================

# Couleurs associées aux niveaux de priorité.
PRIORITY_COLORS = {
    "low": "green",
    "significant": "orange",
    "critical": "red"
}

# Couleurs utilisées en arrière-plan pour visualiser l’état RF.
STATE_COLORS = {
    "normal": "gray",
    "pre_attack": "orange",
    "attack": "red"
}


def safe_numeric(series):
    """
    Convertit une série en valeurs numériques propres.

    Les valeurs infinies sont remplacées par NaN afin d’éviter
    les erreurs lors du tri temporel ou du tracé.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def plot_file(df, fid):
    """
    Génère la timeline SPECTRA-RF pour un fichier donné.

    La courbe principale représente le RiskScore_RF, tandis que les points
    indiquent le niveau de priorité associé à chaque fenêtre temporelle.
    """

    d = df[df["file_id"] == fid].copy()

    # Tri temporel à partir de l’identifiant de fenêtre
    d["window_id_num"] = safe_numeric(d["window_id"])
    d = d.dropna(subset=["window_id_num"])
    d = d.sort_values("window_id_num")

    plt.figure(figsize=(12, 5))

    # =========================
    # Courbe principale du risque
    # =========================

    plt.plot(
        d["window_id_num"],
        d["RiskScore_RF"],
        linewidth=2,
        color="black",
        label="RiskScore_RF"
    )

    # =========================
    # Zones temporelles selon l’état RF
    # =========================
    # Ces bandes rendent la transition normal → pré-attaque → attaque
    # plus facile à interpréter visuellement.

    for _, row in d.iterrows():
        state = str(row.get("rf_state", "normal"))
        x = row["window_id_num"]

        if state == "pre_attack":
            plt.axvspan(x - 0.5, x + 0.5, color="orange", alpha=0.08)

        elif state == "attack":
            plt.axvspan(x - 0.5, x + 0.5, color="red", alpha=0.15)

    # =========================
    # Points colorés par priorité
    # =========================

    for priority in ["low", "significant", "critical"]:
        mask = d["priority"] == priority

        plt.scatter(
            d.loc[mask, "window_id_num"],
            d.loc[mask, "RiskScore_RF"],
            color=PRIORITY_COLORS[priority],
            s=45,
            label=priority
        )

    # =========================
    # Mise en forme
    # =========================

    plt.title(f"SPECTRA-RF — Évolution du risque — {fid}", fontsize=12)
    plt.xlabel("Temps (fenêtres)")
    plt.ylabel("RiskScore_RF")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Nom de fichier sécurisé pour éviter les caractères problématiques
    safe_name = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in fid
    )

    out_path = PLOTS_DIR / f"risk_timeline_{safe_name}.png"

    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    """
    Génère les courbes temporelles de risque pour tous les fichiers RF.

    Cette visualisation permet de suivre l’évolution du RiskScore_RF
    et d’identifier rapidement les fenêtres critiques.
    """

    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = ["file_id", "window_id", "RiskScore_RF", "priority", "rf_state"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante: {col}")

    print("Répartition des priorités :")
    print(df["priority"].value_counts())

    for fid in df["file_id"].unique():
        plot_file(df, fid)

    print("\nFigures SPECTRA-RF sauvegardées dans :", PLOTS_DIR)


if __name__ == "__main__":
    main()