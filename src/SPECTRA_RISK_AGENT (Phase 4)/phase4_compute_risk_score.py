# phase4_compute_risk_score.py

import pandas as pd
from pathlib import Path


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_spectra_scores.csv"
CSV_OUT = OUT_DIR / "WASD_RF_risk_scored.csv"


def classify_priority(row):
    """
    Détermine le niveau de priorité final.

    La priorité reste cohérente avec l’état RF obtenu en Phase 3 :
    une fenêtre déjà considérée comme attaque doit être traitée
    comme critique, même si le score numérique reste modéré.
    """

    state = row["rf_state"]

    if state == "attack":
        return "critical"

    if state == "pre_attack":
        return "significant"

    return "low"


def main():
    """
    Calcule le score de risque RF global.

    Cette étape combine :
    - le score d’anomalie issu de l’autoencoder,
    - le score de gravité SPECTRA-RF,
    - l’état RF produit par AROCLE-RF.

    Le résultat obtenu sert ensuite à la priorisation et à l’anticipation.
    """

    # =========================
    # Chargement des données
    # =========================

    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # =========================
    # Vérification des colonnes
    # =========================

    required = ["anomaly_score", "Score_SPECTRA", "rf_state"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante: {col}")

    # =========================
    # Normalisation du score d’anomalie
    # =========================
    # On amplifie le score d’anomalie tout en le bornant à 1.
    # Cela évite qu’une valeur extrême domine tout le calcul.

    df["anomaly_score_scaled"] = (df["anomaly_score"] * 20).clip(upper=1)

    # =========================
    # Normalisation du score SPECTRA
    # =========================
    # Le score SPECTRA varie théoriquement entre 7 et 21.
    # La division par 21 le ramène dans une échelle comparable.

    df["Score_SPECTRA_norm"] = df["Score_SPECTRA"] / 21.0

    # =========================
    # Calcul du RiskScore RF
    # =========================
    # Ce score relie l’intensité de l’anomalie à la gravité de la menace.
    # Il devient l’indicateur principal utilisé par les phases suivantes.

    df["RiskScore_RF"] = (
        df["anomaly_score_scaled"] *
        df["Score_SPECTRA_norm"]
    )

    # =========================
    # Priorisation finale
    # =========================

    df["priority"] = df.apply(classify_priority, axis=1)

    # =========================
    # Sauvegarde
    # =========================

    df.to_csv(CSV_OUT, index=False, sep=";", decimal=",")

    print("Dataset RiskScore RF sauvegardé :", CSV_OUT)

    # =========================
    # Résumés de contrôle
    # =========================

    print("\nDistribution du RiskScore_RF :")
    print(df["RiskScore_RF"].describe())

    print("\nRépartition des priorités :")
    print(df["priority"].value_counts())

    print("\nVérification cohérence Phase 3 / priorité :")
    print(pd.crosstab(df["rf_state"], df["priority"]))


if __name__ == "__main__":
    main()