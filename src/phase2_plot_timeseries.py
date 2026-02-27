import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
CSV = OUT_DIR / "WASD_RF_dataset_scored.csv"

df = pd.read_csv(CSV, sep=";", decimal=",")

# Choisir 3 fichiers : 1 normal, 1 avec anomalies, 1 mixte
files = df["file_id"].unique()[:3]

for f in files:
    d = df[df["file_id"] == f].copy()
    d["t_ms"] = d["window_id"] * d["time_window_ms"]

    plt.figure()
    plt.plot(d["t_ms"], d["anomaly_score"], marker="o")
    # seuil (si colonne threshold_p99 existe)
    if "threshold_p99" in d.columns:
        plt.axhline(d["threshold_p99"].iloc[0], linestyle="--")

    plt.title(f"AnomalyScore(t) - {f}")
    plt.xlabel("t (ms)")
    plt.ylabel("anomaly_score")
    plt.savefig(OUT_DIR / f"timeseries_{f.replace('.json','')}.png", dpi=200)
    plt.close()

print(" Timeseries saved in outputs/")
