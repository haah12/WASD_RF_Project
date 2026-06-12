# phase2_plot_timeseries.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
CSV_PATH = OUT_DIR / "WASD_RF_dataset_scored.csv"


# Chargement du dataset produit par la Phase 2
df = pd.read_csv(CSV_PATH, sep=";", decimal=",")


# =========================
# Paramètre de visualisation
# =========================

# Seuil fixe utilisé pour l’affichage du score d’anomalie.
# Il est gardé volontairement non normalisé pour rester cohérent
# avec les valeurs réelles calculées par l’autoencoder.
THRESHOLD = 0.12


# =========================
# Analyse statistique par fichier
# =========================

file_stats = []

for file_id in df["file_id"].unique():
    file_df = df[df["file_id"] == file_id]

    mean_score = file_df["anomaly_score"].mean()
    max_score = file_df["anomaly_score"].max()
    std_score = file_df["anomaly_score"].std()

    # Le ratio permet d’identifier les fichiers contenant un pic marqué.
    ratio = max_score / (mean_score + 1e-6)

    file_stats.append({
        "file_id": file_id,
        "mean": mean_score,
        "max": max_score,
        "std": std_score,
        "ratio": ratio
    })

stats_df = pd.DataFrame(file_stats)


# =========================
# Sélection de cas représentatifs
# =========================

# Cas le plus stable : score moyen minimal
file_normal = stats_df.sort_values("mean").iloc[0]["file_id"]

# Cas avec pic soudain : ratio max/moyenne le plus élevé
file_spike = stats_df.sort_values("ratio", ascending=False).iloc[0]["file_id"]

# Cas de dérive : forte variabilité, mais différent des deux premiers cas
drift_candidates = stats_df.sort_values("std", ascending=False)

file_drift = None
for _, row in drift_candidates.iterrows():
    if row["file_id"] not in [file_normal, file_spike]:
        file_drift = row["file_id"]
        break

selected_files = [
    ("NORMAL", file_normal),
    ("SPIKE", file_spike),
    ("DRIFT", file_drift)
]


# =========================
# Génération des courbes temporelles
# =========================

for label, file_id in selected_files:
    file_df = df[df["file_id"] == file_id].copy()

    # Reconstruction de l’axe temporel à partir de l’indice de fenêtre.
    file_df["t_ms"] = file_df["window_id"] * file_df["time_window_ms"]

    plt.figure(figsize=(8, 4))

    # Affichage du score d’anomalie réel, sans normalisation.
    plt.plot(
        file_df["t_ms"],
        file_df["anomaly_score"],
        marker="o",
        linewidth=2,
        label="Anomaly score"
    )

    # Seuil fixe de référence.
    plt.axhline(
        THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2,
        label="threshold = 0.12"
    )

    # Ajustement simple de l’échelle verticale pour rendre la figure lisible.
    ymax = max(file_df["anomaly_score"].max(), THRESHOLD)
    plt.ylim(0, ymax * 1.2)

    plt.title(f"{label} - AnomalyScore(t)\n{file_id}")
    plt.xlabel("Temps (ms)")
    plt.ylabel("Anomaly score")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = OUT_DIR / f"timeseries_{label}_FINAL_FIXED_{file_id.replace('.json', '')}.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

print("Courbes temporelles générées avec un seuil fixe = 0.12, sans normalisation.")