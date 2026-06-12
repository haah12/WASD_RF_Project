# phase3_infer_arocle.py

import pandas as pd
import numpy as np
from pathlib import Path
import joblib


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_arocle_features.csv"
MODEL_PATH = OUT_DIR / "arocle_rf_model.joblib"
CSV_OUT = OUT_DIR / "WASD_RF_arocle_scored.csv"


# =========================
# Features utilisées par AROCLE-RF
# =========================

FEATURES = [
    'RSSI_dB', 'SNR_dB', 'puissance_moyenne', 'variance_IQ',
    'kurtosis', 'skewness', 'PSD_mean', 'occupation_spectrale', 'ISR_dB',
    'RSRP_dB', 'RSRQ_dB', 'EVM_percent', 'BER_est',
    'anomaly_score',
    'delta_snr', 'delta_evm', 'cos_sim_prev',
    'cfo_hz', 'delta_cfo_hz',
    'cos_sim_0_2', 'cos_sim_1_3'
]


def main():
    """
    Applique le modèle AROCLE-RF entraîné sur les features enrichies.

    Cette étape transforme chaque fenêtre RF en une décision interprétable :
    type de menace estimé, niveau de confiance et état RF global.
    """

    # =========================
    # Vérification des fichiers
    # =========================

    if not CSV_IN.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable: {CSV_IN}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle AROCLE introuvable: {MODEL_PATH}")

    # =========================
    # Chargement des données et du modèle
    # =========================

    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes dans {CSV_IN.name}: {missing}")

    clf = joblib.load(MODEL_PATH)

    # =========================
    # Préparation des données
    # =========================

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Les valeurs manquantes peuvent apparaître sur certaines fenêtres,
    # surtout pour les dérivées temporelles ou les estimations CFO.
    if X.isna().any().any():
        print("Valeurs manquantes détectées : imputation par médiane")
        X = X.fillna(X.median())

    # =========================
    # Inférence AROCLE-RF
    # =========================

    predictions = clf.predict(X)

    df_out = df.copy()
    df_out["arocle_rf_pattern"] = predictions

    # Lorsque le classifieur le permet, on récupère aussi les probabilités.
    # Elles sont utiles pour mesurer l'incertitude du modèle.
    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(X)
        classes = list(clf.classes_)

        for i, cls in enumerate(classes):
            df_out[f"proba_{cls}"] = probabilities[:, i]

        df_out["arocle_rf_confidence"] = probabilities.max(axis=1)
    else:
        df_out["arocle_rf_confidence"] = np.nan

    # =========================
    # Interprétation RF de l’état
    # =========================
    # Ici, on ne se limite pas à la classe prédite.
    # On combine le score d’anomalie et la confiance du modèle pour distinguer :
    # - normal
    # - pre_attack
    # - attack

    anomaly_threshold = df["anomaly_score"].quantile(0.95)
    confidence_threshold = df_out["arocle_rf_confidence"].quantile(0.25)

    print(f"\nSeuil anomaly_score (Q95): {anomaly_threshold:.3f}")
    print(f"Seuil confiance modèle (Q25): {confidence_threshold:.3f}")

    is_anomaly = df["anomaly_score"] > anomaly_threshold
    is_low_confidence = df_out["arocle_rf_confidence"] < confidence_threshold

    df_out["rf_state"] = "normal"
    df_out.loc[is_anomaly | is_low_confidence, "rf_state"] = "pre_attack"
    df_out.loc[is_anomaly & ~is_low_confidence, "rf_state"] = "attack"

    # Par cohérence, une fenêtre considérée normale ne porte pas de menace AROCLE.
    df_out.loc[df_out["rf_state"] == "normal", "arocle_rf_pattern"] = "normal"

    # Score simple d’incertitude : plus il est élevé, moins le modèle est sûr.
    df_out["uncertainty_score"] = 1 - df_out["arocle_rf_confidence"]

    # =========================
    # Analyse des cas ambigus
    # =========================

    low_confidence_cases = df_out.sort_values("arocle_rf_confidence").head(15)

    print("\n15 prédictions les moins confiantes :")
    print(low_confidence_cases[[
        "file_id", "window_id",
        "arocle_rf_pattern",
        "arocle_rf_confidence"
    ]])

    # =========================
    # Résumés statistiques
    # =========================

    print("\nConfiance moyenne par classe AROCLE-RF :")
    print(df_out.groupby("arocle_rf_pattern")["arocle_rf_confidence"].mean())

    print("\nRépartition des classes AROCLE-RF :")
    print(df_out["arocle_rf_pattern"].value_counts())

    print("\nRépartition des états RF :")
    print(df_out["rf_state"].value_counts())

    print("\nRésumé de la confiance du modèle :")
    print(df_out["arocle_rf_confidence"].describe())

    print("\nTaux de faible confiance :")
    print((df_out["arocle_rf_confidence"] < confidence_threshold).mean())

    # =========================
    # Sauvegarde
    # =========================

    df_out.to_csv(CSV_OUT, index=False, sep=";", decimal=",")

    print("\nRésultat AROCLE-RF sauvegardé :", CSV_OUT)


if __name__ == "__main__":
    main()