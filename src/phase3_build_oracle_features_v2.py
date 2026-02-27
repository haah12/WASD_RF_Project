# phase3_build_oracle_features_v2.py
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from read_iq import read_iq_file
from windowing import get_windows

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

candidates = [
    PROJECT_ROOT / "data" / "metadata.pkl",
    PROJECT_ROOT / "src" / "data" / "metadata.pkl",
    PROJECT_ROOT / "src" / "data" / "metadata_sim.pkl",
]

META_PKL = None
for p in candidates:
    if p.exists():
        META_PKL = p
        break

if META_PKL is None:
    raise FileNotFoundError("metadata.pkl introuvable dans data/ ou src/data/")
print(" metadata used:", META_PKL)

CSV_IN = OUT_DIR / "WASD_RF_dataset_scored.csv"
CSV_OUT = OUT_DIR / "WASD_RF_oracle_features_v2.csv"

# Base features (celles de Phase 2)
BASE_FEATURES = [
    'RSSI_dB','SNR_dB','puissance_moyenne','variance_IQ',
    'kurtosis','skewness','PSD_mean','occupation_spectrale','ISR_dB',
    'RSRP_dB','RSRQ_dB','EVM_percent','BER_est',
    'center_frequency_hz','bandwidth_hz','time_window_ms'
]

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def estimate_cfo_hz(iq: np.ndarray, fs: float) -> float:
    """
    CFO proxy simple:
    phase_inc ≈ angle(mean(conj(x[n]) * x[n+1]))
    CFO_hz ≈ fs/(2π) * phase_inc
    """
    if len(iq) < 2:
        return 0.0
    ph = np.angle(np.mean(np.conj(iq[:-1]) * iq[1:]))
    return float(fs * ph / (2*np.pi))

def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_IN}")
    if not META_PKL.exists():
        raise FileNotFoundError(f"metadata.pkl not found: {META_PKL}")

    df = pd.read_csv(CSV_IN, sep=";", decimal=",")
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    # Merge metadata -> trouver bin_path, sample_rate_hz, iq_dataform
    meta_small = meta[["json_file", "bin_path", "sample_rate_hz", "iq_dataform"]].copy()
    df = df.merge(meta_small, left_on="file_id", right_on="json_file", how="left")

    if df["bin_path"].isna().any():
        missing = df[df["bin_path"].isna()]["file_id"].unique()[:5]
        raise ValueError(f"BIN path missing for some files, e.g.: {missing}")

    # ---- 1) Temporal features + CFO per window (par file)
    df["delta_snr"] = np.nan
    df["delta_evm"] = np.nan
    df["cos_sim_prev"] = np.nan
    df["cfo_hz"] = np.nan
    df["delta_cfo_hz"] = np.nan

    # replay proxies for 4 windows
    df["cos_sim_0_2"] = np.nan
    df["cos_sim_1_3"] = np.nan

    VEC_FEATURES = [
        'RSSI_dB','SNR_dB','puissance_moyenne','variance_IQ',
        'kurtosis','skewness','PSD_mean','occupation_spectrale','ISR_dB',
        'RSRP_dB','RSRQ_dB','EVM_percent','BER_est'
    ]

    for fid, g in df.groupby("file_id", sort=False):
        g = g.sort_values("window_id").copy()
        fs = float(g["sample_rate_hz"].iloc[0])
        bin_path = g["bin_path"].iloc[0]
        dataform = int(g["iq_dataform"].iloc[0])

        # Lire IQ et découper 4 fenêtres comme Phase 1
        iq = read_iq_file(bin_path, dataform)
        wins = get_windows(iq)  # par défaut n_windows=4 dans ton windowing.py

        # CFO proxy par fenêtre
        cfo_list = [estimate_cfo_hz(w, fs) for w in wins]

        # remplir CFO
        for idx, row in g.iterrows():
            wid = int(row["window_id"])
            if 0 <= wid < len(cfo_list):
                df.loc[idx, "cfo_hz"] = cfo_list[wid]

        # deltas + cos_sim_prev
        X = g[VEC_FEATURES].astype(float).values
        for i in range(len(g)):
            if i == 0:
                df.loc[g.index[i], "delta_snr"] = 0.0
                df.loc[g.index[i], "delta_evm"] = 0.0
                df.loc[g.index[i], "cos_sim_prev"] = 0.0
                df.loc[g.index[i], "delta_cfo_hz"] = 0.0
            else:
                df.loc[g.index[i], "delta_snr"] = float(g["SNR_dB"].iloc[i] - g["SNR_dB"].iloc[i-1])
                df.loc[g.index[i], "delta_evm"] = float(g["EVM_percent"].iloc[i] - g["EVM_percent"].iloc[i-1])
                df.loc[g.index[i], "cos_sim_prev"] = cosine_sim(X[i], X[i-1])
                df.loc[g.index[i], "delta_cfo_hz"] = float(cfo_list[i] - cfo_list[i-1])

        # replay: similarité non-adjacente (0↔2, 1↔3) si 4 fenêtres
        if len(g) >= 4:
            sim_0_2 = cosine_sim(X[0], X[2])
            sim_1_3 = cosine_sim(X[1], X[3])
            for idx in g.index:
                df.loc[idx, "cos_sim_0_2"] = sim_0_2
                df.loc[idx, "cos_sim_1_3"] = sim_1_3

    #  AJOUT ICI (après calcul complet du CFO et deltas)
    df["cfo_khz"] = df["cfo_hz"] / 1e3
    df["delta_cfo_khz"] = df["delta_cfo_hz"] / 1e3

    # Nettoyage colonnes temporaires merge
    df.drop(columns=["json_file"], inplace=True, errors="ignore")

    df.to_csv(CSV_OUT, index=False, sep=";", decimal=",")
    print(" Saved:", CSV_OUT)
    print(df[["delta_snr","delta_evm","cos_sim_prev","cfo_hz","cfo_khz","delta_cfo_hz","delta_cfo_khz","cos_sim_0_2","cos_sim_1_3"]].describe())

if __name__ == "__main__":
    main()
