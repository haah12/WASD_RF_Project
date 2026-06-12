# phase4_compute_spectra_scores.py

import pandas as pd
from pathlib import Path


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_arocle_scored.csv"
CSV_OUT = OUT_DIR / "WASD_RF_spectra_scores.csv"


def compute_spectra(row):
    """
    Calcule les sept critères SPECTRA-RF pour une fenêtre donnée.

    Chaque critère traduit un aspect différent de la gravité RF :
    dégradation du signal, propagation de l’anomalie, exploitabilité,
    couverture, probabilité de menace, visibilité et adaptabilité.
    """

    snr = row["SNR_dB"]
    evm = row["EVM_percent"]
    ber = row.get("BER_est", 0)
    anomaly = row["anomaly_score"]
    delta_snr = row.get("delta_snr", 0)
    cfo = abs(row.get("cfo_hz", 0))
    isr = row.get("ISR_dB", 0)

    pattern = str(row.get("arocle_rf_pattern", "normal"))

    # S — Severity
    # Mesure l’impact direct sur la qualité du signal RF.
    if snr < 16 or evm > 15 or ber > 0.01:
        S = 3
    elif 16 <= snr < 18 or 13 <= evm <= 15:
        S = 2
    else:
        S = 1

    # P — Propagation
    # Représente l’intensité globale de l’anomalie détectée.
    if anomaly > 0.04:
        P = 3
    elif anomaly > 0.02:
        P = 2
    else:
        P = 1

    # E — Exploitability
    # Certaines classes AROCLE sont considérées comme plus directement exploitables.
    if pattern in ["A", "O"]:
        E = 3
    elif pattern in ["R", "C"]:
        E = 2
    else:
        E = 1

    # C — Coverage
    # Approxime l’étendue de l’impact à partir de l’intensité de l’anomalie.
    if anomaly > 0.05:
        C = 3
    elif anomaly > 0.02:
        C = 2
    else:
        C = 1

    # T — Threat Likelihood
    # Une chute brutale du SNR augmente la probabilité d’une menace active.
    if delta_snr < -0.2:
        T = 3
    elif delta_snr < -0.05:
        T = 2
    else:
        T = 1

    # R — Recognizability
    # Mesure à quel point l’anomalie est visible dans les métriques RF.
    if evm > 15 or isr > 10:
        R = 3
    elif 13 <= evm <= 15:
        R = 2
    else:
        R = 1

    # A — Adaptability
    # Une forte dérive CFO peut indiquer un comportement radio dynamique.
    if cfo > 2e7:
        A = 3
    elif cfo > 1e7:
        A = 2
    else:
        A = 1

    score = S + P + E + C + T + R + A

    return pd.Series({
        "S": S,
        "P": P,
        "E": E,
        "C": C,
        "T": T,
        "R": R,
        "A": A,
        "Score_SPECTRA": score
    })


def main():
    """
    Calcule le score SPECTRA-RF et le niveau de gravité associé.

    Cette phase transforme la menace identifiée par AROCLE-RF
    en une mesure de gravité exploitable par la priorisation.
    """

    # =========================
    # Chargement des données
    # =========================

    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # =========================
    # Vérification des colonnes
    # =========================

    required = [
        "SNR_dB",
        "EVM_percent",
        "anomaly_score",
        "arocle_rf_pattern",
        "rf_state"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante: {col}")

    # =========================
    # Calcul des critères SPECTRA
    # =========================

    df = df.join(df.apply(compute_spectra, axis=1))

    # =========================
    # Interprétation du niveau SPECTRA
    # =========================
    # On donne la priorité à l’état RF issu de la Phase 3 afin de garder
    # une cohérence entre la classification AROCLE et la gravité SPECTRA.

    def interpret(row):
        state = row["rf_state"]
        score = row["Score_SPECTRA"]

        if state == "normal":
            return "faible"

        if state == "pre_attack":
            return "significatif"

        if state == "attack":
            return "critique"

        # Cas de repli si l’état RF n’est pas reconnu.
        if score >= 16:
            return "critique"

        if score >= 11:
            return "significatif"

        return "faible"

    df["SPECTRA_level"] = df.apply(interpret, axis=1)

    # =========================
    # Sauvegarde
    # =========================

    df.to_csv(CSV_OUT, index=False, sep=";", decimal=",")

    print("Scores SPECTRA-RF sauvegardés :", CSV_OUT)

    # =========================
    # Résumés de contrôle
    # =========================

    print("\nDistribution du Score_SPECTRA :")
    print(df["Score_SPECTRA"].describe())

    print("\nRépartition des niveaux SPECTRA :")
    print(df["SPECTRA_level"].value_counts())

    print("\nVérification cohérence Phase 3 / Phase 4 :")
    print(pd.crosstab(df["rf_state"], df["SPECTRA_level"]))


if __name__ == "__main__":
    main()