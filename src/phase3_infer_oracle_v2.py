# phase3_infer_oracle_v2.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_oracle_features_v2.csv"
MODEL = OUT_DIR / "oracle_rf_model_v2.joblib"
CSV_OUT = OUT_DIR / "WASD_RF_oracle_scored_v2.csv"

FEATURES = [
    'RSSI_dB','SNR_dB','puissance_moyenne','variance_IQ',
    'kurtosis','skewness','PSD_mean','occupation_spectrale','ISR_dB',
    'RSRP_dB','RSRQ_dB','EVM_percent','BER_est',
    'anomaly_score',
    'delta_snr','delta_evm','cos_sim_prev','cfo_hz','delta_cfo_hz','cos_sim_0_2','cos_sim_1_3'
]

def main():
    # 1) Load data (anti-BOM + strip)
    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # 2) Check features exist
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f" Features manquantes dans {CSV_IN.name}: {missing}\n"
            f" Vérifie que ton fichier d'entrée contient bien anomaly_score & features v2."
        )

    # 3) Load model
    clf = joblib.load(MODEL)

    # 4) Numeric conversion safe (keeps DataFrame with column names)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # 5) Predict
    pred = clf.predict(X)

    df_out = df.copy()
    df_out["oracle_rf_pattern"] = pred

    # 6) Optional: probabilities (if supported)
    # Useful to debug confidence / borderline cases
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        classes = list(clf.classes_)
        for i, cls in enumerate(classes):
            df_out[f"proba_{cls}"] = proba[:, i]
        df_out["oracle_rf_confidence"] = proba.max(axis=1)

    # 7) Save
    low = df_out.sort_values("oracle_rf_confidence").head(15)
    print("\n 15 prédictions les moins confiantes:")
    print(low[["file_id", "window_id", "oracle_rf_pattern", "oracle_rf_confidence"]])
    cols_show = ["file_id","window_id","oracle_rf_pattern","oracle_rf_confidence"] + FEATURES
    print(low[cols_show])

    df_out.to_csv(CSV_OUT, index=False, sep=";", decimal=",")
    print(" Saved:", CSV_OUT)

    print("\n Répartition oracle_rf_pattern:")
    print(df_out["oracle_rf_pattern"].value_counts(dropna=False))

    if "oracle_rf_confidence" in df_out.columns:
        print("\n Confidence (max proba) summary:")
        print(df_out["oracle_rf_confidence"].describe())

if __name__ == "__main__":
    main()
