# phase3_build_arocle_features.py

import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from read_iq import read_iq_file
from windowing import get_windows


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"


# =========================
# Chargement des métadonnées
# =========================
# On cherche le fichier metadata.pkl dans plusieurs emplacements possibles
# afin de rendre le script robuste aux différentes organisations du projet.

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
    raise FileNotFoundError("metadata.pkl introuvable")

print("Metadata utilisée :", META_PKL)


# =========================
# Entrées / sorties
# =========================

CSV_IN = OUT_DIR / "WASD_RF_dataset_scored.csv"
CSV_OUT = OUT_DIR / "WASD_RF_arocle_features.csv"


# =========================
# Colonnes nécessaires
# =========================

REQUIRED_COLUMNS = [
    "file_id",
    "window_id",
    "SNR_dB",
    "EVM_percent",
    "anomaly_score",
]

# Vecteur utilisé pour les similarités
VEC_FEATURES = [
    "RSSI_dB",
    "SNR_dB",
    "puissance_moyenne",
    "variance_IQ",
    "kurtosis",
    "skewness",
    "PSD_mean",
    "occupation_spectrale",
    "ISR_dB",
    "RSRP_dB",
    "RSRQ_dB",
    "EVM_percent",
    "BER_est",
]


# =========================
# Fonctions utilitaires
# =========================

def safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convertit une colonne en numérique en gérant proprement les erreurs.
    Les valeurs infinies sont transformées en NaN.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Similarité cosinus entre deux vecteurs de features.
    Permet de mesurer la stabilité ou la répétition du signal.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan

    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps

    return float(np.dot(a, b) / (na * nb))


def estimate_cfo_hz(iq: np.ndarray, fs: float) -> float:
    """
    Estimation simple du CFO (Carrier Frequency Offset).
    Permet de capturer les dérives de fréquence caractéristiques de certaines attaques RF.
    """
    if iq is None or len(iq) < 2 or fs <= 0:
        return np.nan

    prod = np.conj(iq[:-1]) * iq[1:]
    mean_prod = np.mean(prod)

    if np.isnan(mean_prod.real) or np.isnan(mean_prod.imag):
        return np.nan

    phase = np.angle(mean_prod)
    return float(fs * phase / (2 * np.pi))


# =========================
# Pipeline principal
# =========================

def main():

    if not CSV_IN.exists():
        raise FileNotFoundError(f"CSV introuvable: {CSV_IN}")

    # Chargement des données
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)

    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Vérification des colonnes minimales
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante: {col}")

    # Harmonisation des types
    df["file_id"] = df["file_id"].astype(str).str.strip()
    df["window_id"] = safe_numeric(df["window_id"]).astype("Int64")

    if df["window_id"].isna().any():
        raise ValueError("window_id invalide détecté")

    # =========================
    # Jointure avec les métadonnées
    # =========================

    meta_small = meta[["json_file", "bin_path", "sample_rate_hz", "iq_dataform"]].copy()
    meta_small["json_file"] = meta_small["json_file"].astype(str).str.strip()

    df = df.merge(
        meta_small,
        left_on="file_id",
        right_on="json_file",
        how="left",
        validate="many_to_one"
    )

    if df["bin_path"].isna().any():
        raise ValueError("Certains fichiers n'ont pas de BIN associé")

    # =========================
    # Nettoyage des données
    # =========================

    numeric_cols = list(set(VEC_FEATURES + ["SNR_dB", "EVM_percent", "anomaly_score"]))

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # =========================
    # Initialisation des nouvelles features AROCLE
    # =========================

    new_cols = [
        "delta_snr", "delta_evm", "delta_anomaly_score",
        "cos_sim_prev", "cfo_hz", "delta_cfo_hz",
        "cos_sim_0_2", "cos_sim_1_3",
        "window_start_ms", "window_center_ms",
    ]

    for col in new_cols:
        df[col] = np.nan

    # =========================
    # Traitement par fichier RF
    # =========================

    for fid, group in df.groupby("file_id", sort=False):

        group = group.sort_values("window_id").copy()

        fs = float(group["sample_rate_hz"].iloc[0])
        bin_path = group["bin_path"].iloc[0]
        dataform = int(group["iq_dataform"].iloc[0])

        # Lecture IQ + découpage
        iq = read_iq_file(bin_path, dataform)
        windows = list(get_windows(iq))

        if len(windows) == 0:
            print(f"[WARN] aucune fenêtre pour {fid}")
            continue

        # CFO par fenêtre
        cfo_list = [estimate_cfo_hz(w, fs) for w in windows]

        # Temps relatif
        time_window_ms = safe_numeric(group["time_window_ms"]).iloc[0] if "time_window_ms" in group.columns else np.nan

        for i, (idx, row) in enumerate(group.iterrows()):

            wid = int(row["window_id"])

            if 0 <= wid < len(cfo_list):
                df.loc[idx, "cfo_hz"] = cfo_list[wid]

            if not pd.isna(time_window_ms):
                start_ms = wid * float(time_window_ms)
                center_ms = start_ms + float(time_window_ms) / 2
                df.loc[idx, "window_start_ms"] = start_ms
                df.loc[idx, "window_center_ms"] = center_ms

        # Similarité entre fenêtres
        X = group[VEC_FEATURES].astype(float).values

        for i in range(len(group)):
            idx = group.index[i]

            if i == 0:
                df.loc[idx, ["delta_snr", "delta_evm", "delta_anomaly_score"]] = 0.0
                df.loc[idx, "cos_sim_prev"] = 1.0
                df.loc[idx, "delta_cfo_hz"] = 0.0
            else:
                df.loc[idx, "delta_snr"] = group["SNR_dB"].iloc[i] - group["SNR_dB"].iloc[i-1]
                df.loc[idx, "delta_evm"] = group["EVM_percent"].iloc[i] - group["EVM_percent"].iloc[i-1]
                df.loc[idx, "delta_anomaly_score"] = group["anomaly_score"].iloc[i] - group["anomaly_score"].iloc[i-1]

                df.loc[idx, "cos_sim_prev"] = cosine_sim(X[i], X[i-1])

                if i < len(cfo_list):
                    df.loc[idx, "delta_cfo_hz"] = cfo_list[i] - cfo_list[i-1]

        # Similarités longue portée
        if len(group) >= 4:
            df.loc[group.index, "cos_sim_0_2"] = cosine_sim(X[0], X[2])
            df.loc[group.index, "cos_sim_1_3"] = cosine_sim(X[1], X[3])

    # Conversion en kHz (plus lisible)
    df["cfo_khz"] = df["cfo_hz"] / 1e3
    df["delta_cfo_khz"] = df["delta_cfo_hz"] / 1e3

    # Nettoyage
    df.drop(columns=["json_file"], inplace=True, errors="ignore")

    # Sauvegarde
    df.to_csv(CSV_OUT, index=False, sep=";", decimal=",", encoding="utf-8-sig")

    print("Features AROCLE sauvegardées :", CSV_OUT)


if __name__ == "__main__":
    main()