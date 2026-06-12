# phase5_build_sequences.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_risk_scored.csv"
OUT_FILE = OUT_DIR / "WASD_RF_sequences_v5.npz"
SCALER_OUT = OUT_DIR / "scaler_phase5_v5.joblib"


# =========================
# Paramètres temporels
# =========================

SEQ_LEN = 15

STATE_MAP = {
    "normal": 0,
    "pre_attack": 1,
    "attack": 2
}


# Features issues des phases précédentes.
# Elles résument à la fois le comportement RF, le risque et la gravité.
BASE_FEATURES = [
    "anomaly_score", "RiskScore_RF", "Score_SPECTRA",
    "SNR_dB", "EVM_percent", "delta_snr", "delta_evm",
    "cfo_hz", "ISR_dB", "kurtosis", "skewness"
]


def add_temporal_features(df):
    """
    Ajoute des indicateurs temporels simples autour du RiskScore_RF.

    L’objectif est de donner au modèle séquentiel une information
    sur la tendance, la variation et l’accélération du risque.
    """

    df["rolling_risk"] = df["RiskScore_RF"].rolling(5, min_periods=1).mean()
    df["risk_diff"] = df["RiskScore_RF"].diff().fillna(0)
    df["risk_acc"] = df["risk_diff"].diff().fillna(0)

    df["risk_trend"] = df["rolling_risk"].diff().fillna(0)
    df["snr_trend"] = df["SNR_dB"].rolling(5).mean().diff().fillna(0)
    df["evm_trend"] = df["EVM_percent"].rolling(5).mean().diff().fillna(0)

    return df


def main():
    """
    Construit les séquences temporelles utilisées en Phase 5.

    Chaque séquence contient plusieurs fenêtres RF successives.
    Le label associé correspond à l’état futur du système, ce qui permet
    d’entraîner un modèle d’anticipation plutôt qu’un simple détecteur.
    """

    # =========================
    # Chargement et tri temporel
    # =========================

    df = pd.read_csv(CSV_IN, sep=";", decimal=",")
    df.columns = df.columns.str.strip()

    df = df.sort_values(["file_id", "window_id"]).reset_index(drop=True)

    # Ajout des informations temporelles dérivées
    df = add_temporal_features(df)

    # Conversion des états RF en labels numériques
    df["state_label"] = df["rf_state"].map(STATE_MAP)

    FEATURES = BASE_FEATURES + [
        "rolling_risk", "risk_diff", "risk_acc",
        "risk_trend", "snr_trend", "evm_trend"
    ]

    # =========================
    # Préparation de la matrice de features
    # =========================

    X_raw = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0).values

    y_raw = df["state_label"].values

    X_all = []
    y_all = []
    file_ids = []
    window_ids = []

    # =========================
    # Construction des séquences
    # =========================
    # Pour chaque position temporelle, on prend les SEQ_LEN fenêtres passées
    # et on demande au modèle de prédire l’état de la fenêtre suivante.

    for i in range(len(df) - SEQ_LEN):
        sequence = X_raw[i:i + SEQ_LEN]

        # Label futur : état RF après la séquence observée
        future_label = y_raw[i + SEQ_LEN]

        X_all.append(sequence)
        y_all.append(future_label)

        file_ids.append(df["file_id"].iloc[i + SEQ_LEN])
        window_ids.append(df["window_id"].iloc[i + SEQ_LEN])

    X = np.array(X_all)
    y = np.array(y_all)

    print("Shape des séquences :", X.shape)
    print("Distribution des classes :", np.bincount(y))

    # =========================
    # Normalisation
    # =========================
    # Le scaler est ajusté sur toutes les fenêtres mises à plat,
    # puis les données sont remises sous forme séquentielle.

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X.reshape(-1, X.shape[-1])
    )

    X = X_scaled.reshape(X.shape)

    joblib.dump(scaler, SCALER_OUT)

    # =========================
    # Sauvegarde
    # =========================

    np.savez(
        OUT_FILE,
        X=X,
        y=y,
        file_id=np.array(file_ids),
        window_id=np.array(window_ids)
    )

    print("Séquences Phase 5 sauvegardées :", OUT_FILE)


if __name__ == "__main__":
    main()