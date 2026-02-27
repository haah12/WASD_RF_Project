# build_csv.py
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_final_csv(
    features_pkl: str = 'data/features_17.pkl',
    output_csv: str = str(OUTPUT_DIR / 'WASD_RF_dataset.csv')
) -> pd.DataFrame:
    """
    Génère le CSV final depuis features_17.pkl.
    - Export stable: sep="," et decimal="."
    - Labels simples (tu peux ajuster)
    """
    df = pd.read_pickle(features_pkl).copy()

    # --- rf_pattern (exemples raisonnables)
    df["rf_pattern"] = "normal"
    # Anomalie si SNR très faible (bruité)
    df.loc[df["SNR_dB"] < 5.0, "rf_pattern"] = "low_snr"
    # Tone si ISR très élevé (pic spectral fort)
    df.loc[df["ISR_dB"] > 15.0, "rf_pattern"] = "tone"
    # Pulse si EVM trop grande
    df.loc[df["EVM_percent"] > 15.0, "rf_pattern"] = "pulse"

    cols = [
    'time_window_ms','center_frequency_hz','bandwidth_hz',
    'RSSI_dB','SNR_dB','puissance_moyenne','variance_IQ',
    'kurtosis','skewness','PSD_mean','occupation_spectrale','ISR_dB',
    'RSRP_dB','RSRQ_dB','EVM_percent','BER_est',
    'file_id','window_id','rf_pattern'
]


    cols = [c for c in cols if c in df.columns]
    out_df = df[cols]

    # IMPORTANT: export stable (évite les formats FR qui cassent Excel / parsing)
    out_df.to_csv(output_csv, index=False, sep=";", decimal=",")

    print(f" CSV final créé: {output_csv} | lignes={len(out_df)} | colonnes={len(cols)}")

    return out_df
