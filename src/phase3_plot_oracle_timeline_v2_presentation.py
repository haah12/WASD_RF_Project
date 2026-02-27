# src/phase3_plot_oracle_timeline_v2_presentation.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_oracle_scored_v2.csv"  # contient anomaly_score + oracle_rf_pattern
PLOTS_DIR = OUT_DIR / "plots_oracle_timeline_v2_presentation"  # nouveau dossier
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["file_id", "window_id"]
SCORE_COL = "anomaly_score"
PATTERN_COL = "oracle_rf_pattern"

# Couleurs (points + bandes)
PATTERN_COLORS = {
    "normal": "gray",
    "L": "red",
    "C": "orange",
    "O": "blue",
    "A": "purple",
    "R": "green",
    "E": "brown",
}


def safe_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def plot_one_file(df_file: pd.DataFrame, file_id: str, max_windows: int | None = None) -> Path:
    d = df_file.copy()

    # Sort by time axis = window_id
    d["window_id_num"] = safe_numeric(d["window_id"])
    d = d.dropna(subset=["window_id_num"])
    d = d.sort_values("window_id_num")

    # Optionnel: limiter nombre de points si énorme
    if max_windows is not None and len(d) > max_windows:
        d = d.iloc[:max_windows].copy()

    # anomaly_score numérique
    d[SCORE_COL] = safe_numeric(d[SCORE_COL])

    # ======================================================
    #  2) Seuil visuel anomaly_score (P95 du fichier)
    # ======================================================
    # (si peu de points, P95 reste ok; sinon tu peux mettre 0.99)
    thr = float(d[SCORE_COL].quantile(0.95)) if d[SCORE_COL].notna().any() else np.nan

    # Figure
    plt.figure(figsize=(12, 4.8))

    # ======================================================
    #  1) Bandes verticales pour les anomalies (fond coloré)
    # ======================================================
    patterns = d[PATTERN_COL].fillna("unknown").astype(str)
    for _, row in d.iterrows():
        pat = str(row[PATTERN_COL])
        if pat != "normal":
            color = PATTERN_COLORS.get(pat, "black")
            plt.axvspan(
                row["window_id_num"] - 0.5,
                row["window_id_num"] + 0.5,
                color=color,
                alpha=0.15
            )

    # 1) Courbe anomaly_score(t)
    plt.plot(d["window_id_num"], d[SCORE_COL], linewidth=2.0)

    # 2) Points colorés par pattern
    for pat in sorted(patterns.unique()):
        mask = patterns == pat
        color = PATTERN_COLORS.get(pat, "black")
        plt.scatter(
            d.loc[mask, "window_id_num"],
            d.loc[mask, SCORE_COL],
            s=45,
            alpha=0.9,
            label=pat,
        )

    # Ligne seuil
    if not np.isnan(thr):
        plt.axhline(
            y=thr,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="P95 anomaly_score"
        )

    # ======================================================
    #  3) Lisibilité “soutenance”
    # ======================================================
    plt.title(f"ORACLE RF v2 — Temporal detection — {file_id}", fontsize=12)
    plt.xlabel("Time (window index)")
    plt.ylabel("Anomaly score (RF)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.95, ncol=6, fontsize=9)
    plt.tight_layout()

    # Nom de fichier safe
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file_id)
    out_path = PLOTS_DIR / f"timeline_pres_{safe_name}.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def main():
    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Checks
    for c in KEYS + [SCORE_COL, PATTERN_COL]:
        if c not in df.columns:
            raise ValueError(f" Colonne manquante: {c} dans {CSV_IN.name}")

    df["file_id"] = df["file_id"].astype(str).str.strip()

    print(" Répartition globale oracle_rf_pattern:")
    print(df[PATTERN_COL].value_counts(dropna=False))

    paths = []
    for file_id, g in df.groupby("file_id", sort=True):
        p = plot_one_file(g, file_id=file_id)
        paths.append(p)

    print(f"\n {len(paths)} figures générées dans: {PLOTS_DIR}")
    print("Exemples:")
    for p in paths[:5]:
        print(" -", p)


if __name__ == "__main__":
    main()
