# phase3_plot_arocle_timeline_presentation.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_arocle_scored.csv"
PLOTS_DIR = OUT_DIR / "plots_arocle_timeline_presentation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Colonnes utilisées
# =========================

KEYS = ["file_id", "window_id"]
SCORE_COL = "anomaly_score"
PATTERN_COL = "arocle_rf_pattern"
STATE_COL = "rf_state"


# Couleurs choisies pour rendre la progression visuelle plus lisible :
# normal → gris, pré-attaque → orange, attaque → rouge.
STATE_COLORS = {
    "normal": "gray",
    "pre_attack": "orange",
    "attack": "red",
}


def safe_numeric(series):
    """
    Convertit une série en valeurs numériques exploitables.
    Les valeurs infinies sont remplacées par NaN afin d’éviter
    des erreurs lors du tracé.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def plot_one_file(df_file, file_id):
    """
    Génère une timeline AROCLE-RF pour un fichier RF donné.

    La figure montre l'évolution du score d'anomalie et met en évidence
    les zones normales, pré-attaque et attaque.
    """

    d = df_file.copy()

    # Préparation de l’axe temporel
    d["window_id_num"] = safe_numeric(d["window_id"])
    d = d.dropna(subset=["window_id_num"])
    d = d.sort_values("window_id_num")

    # Nettoyage du score d’anomalie
    d[SCORE_COL] = safe_numeric(d[SCORE_COL])

    # Seuil local basé sur le 95e percentile du fichier courant
    thr = float(d[SCORE_COL].quantile(0.95)) if d[SCORE_COL].notna().any() else np.nan

    plt.figure(figsize=(12, 5))

    # Courbe principale du score d’anomalie
    plt.plot(
        d["window_id_num"],
        d[SCORE_COL],
        linewidth=2,
        color="black",
        label="Anomaly score"
    )

    # Points associés aux différents états RF
    states = d[STATE_COL].fillna("normal").astype(str)

    for state in ["normal", "pre_attack", "attack"]:
        mask = states == state
        color = STATE_COLORS.get(state, "black")

        plt.scatter(
            d.loc[mask, "window_id_num"],
            d.loc[mask, SCORE_COL],
            s=50,
            alpha=0.9,
            label=state,
            color=color
        )

    # Zones colorées pour rendre la transition temporelle plus visible.
    # Cela permet de visualiser clairement le passage du comportement normal
    # vers la pré-attaque puis l’attaque.
    for _, row in d.iterrows():
        state = str(row[STATE_COL])
        x = row["window_id_num"]

        if state == "pre_attack":
            plt.axvspan(x - 0.5, x + 0.5, color="orange", alpha=0.08)

        elif state == "attack":
            plt.axvspan(x - 0.5, x + 0.5, color="red", alpha=0.15)

    # Seuil de référence
    if not np.isnan(thr):
        plt.axhline(
            y=thr,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Seuil P95"
        )

    # Mise en forme de la figure
    plt.title(f"AROCLE-RF — Anticipation des menaces — {file_id}", fontsize=12)
    plt.xlabel("Temps (fenêtres)")
    plt.ylabel("Anomaly Score RF")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=3, fontsize=9)
    plt.tight_layout()

    # Sauvegarde avec un nom compatible avec le système de fichiers
    safe_name = "".join(
        c if c.isalnum() or c in "._-" else "_"
        for c in file_id
    )

    out_path = PLOTS_DIR / f"timeline_arocle_{safe_name}.png"

    plt.savefig(out_path, dpi=220)
    plt.close()

    return out_path


def main():
    """
    Génère les timelines AROCLE-RF pour l'ensemble des fichiers disponibles.
    """

    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Vérification minimale des colonnes nécessaires au tracé
    for col in KEYS + [SCORE_COL, PATTERN_COL, STATE_COL]:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante: {col}")

    df["file_id"] = df["file_id"].astype(str).str.strip()

    print("Répartition des états RF :")
    print(df[STATE_COL].value_counts())

    generated_paths = []

    for file_id, group in df.groupby("file_id", sort=True):
        plot_path = plot_one_file(group, file_id)
        generated_paths.append(plot_path)

    print(f"\n{len(generated_paths)} figures générées dans : {PLOTS_DIR}")

    for path in generated_paths[:5]:
        print(" -", path)


if __name__ == "__main__":
    main()