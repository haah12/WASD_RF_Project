# features.py

import numpy as np
import pandas as pd
from scipy import signal, stats
import pickle
from pathlib import Path

from read_iq import read_iq_file
from windowing import get_windows


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

METADATA_PKL = DATA_DIR / "metadata.pkl"
FEATURES_PKL = DATA_DIR / "features_17.pkl"


def welch_psd_db(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Calcule la densite spectrale de puissance en dB.
    """

    if len(x) < 512:
        psd = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
        psd = psd / (len(x) + 1e-12)
        return 10 * np.log10(psd + 1e-18)

    nperseg = min(4096, max(512, len(x) // 8))

    f, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        return_onesided=False,
        scaling="density"
    )

    psd = np.abs(psd) + 1e-18
    idx = np.argsort(f)

    return 10 * np.log10(psd[idx])


def compute_17_features(
    window_iq: np.ndarray,
    fs: float,
    center_freq: float,
    bandwidth: float,
    total_samples: int = None,
    num_windows: int = 4
) -> dict:
    """
    Extrait les principales features RF d'une fenetre IQ.
    """

    eps = 1e-12
    x = window_iq

    if total_samples is not None and fs > 0:
        total_duration_ms = (float(total_samples) / float(fs)) * 1000.0
        time_window_ms = total_duration_ms / float(num_windows)
    else:
        time_window_ms = (len(x) / float(fs)) * 1000.0 if fs > 0 else 0.0

    power = float(np.mean(np.abs(x) ** 2))
    rssi_db = float(10.0 * np.log10(power + eps))
    variance_iq = float(np.var(x))

    kurtosis_val = float(stats.kurtosis(np.real(x), fisher=True, bias=False))
    skewness_val = float(stats.skew(np.real(x), bias=False))

    psd_db = welch_psd_db(x, fs)
    psd_lin = 10 ** (psd_db / 10.0)
    psd_mean = float(np.mean(psd_lin))

    noise_floor_db = float(np.percentile(psd_db, 20))
    signal_level_db = float(np.percentile(psd_db, 95))

    snr_db = float(signal_level_db - noise_floor_db)
    snr_db = float(np.clip(snr_db, -10.0, 60.0))

    spectral_occ = float(np.mean(psd_db > (noise_floor_db + 6.0)))
    spectral_occ = float(np.clip(spectral_occ, 0.0, 1.0))

    isr_db = float(np.max(psd_db) - np.median(psd_db))
    isr_db = float(np.clip(isr_db, 0.0, 80.0))

    n_rb = max(float(bandwidth) / 180e3, 1.0)
    rsrp_db = float(rssi_db - 10.0 * np.log10(n_rb))

    occ_safe = max(spectral_occ, 1e-3)
    rsrq_db = float(rsrp_db - 10.0 * np.log10(occ_safe))

    evm_percent = float(100.0 * (10.0 ** (-snr_db / 20.0)))
    ber_est = float(np.clip(0.5 * np.exp(-snr_db / 10.0), 1e-9, 0.5))

    return {
        "time_window_ms": float(round(time_window_ms, 4)),
        "center_frequency_hz": float(center_freq),
        "bandwidth_hz": float(bandwidth),

        "RSSI_dB": float(round(rssi_db, 4)),
        "SNR_dB": float(round(snr_db, 4)),

        "puissance_moyenne": float(power),
        "variance_IQ": float(variance_iq),

        "kurtosis": float(round(kurtosis_val, 6)),
        "skewness": float(round(skewness_val, 6)),

        "PSD_mean": float(psd_mean),
        "occupation_spectrale": float(round(spectral_occ, 6)),
        "ISR_dB": float(round(isr_db, 4)),

        "RSRP_dB": float(round(rsrp_db, 4)),
        "RSRQ_dB": float(round(rsrq_db, 4)),

        "EVM_percent": float(round(evm_percent, 4)),
        "BER_est": float(ber_est),

        "rf_pattern": "normal"
    }


def extract_all_features(max_files: int = 505, num_windows: int = 4) -> pd.DataFrame:
    """
    Parcourt les fichiers IQ disponibles et construit le fichier features_17.pkl.
    """

    if not METADATA_PKL.exists():
        raise FileNotFoundError(f"metadata.pkl introuvable : {METADATA_PKL}")

    with open(METADATA_PKL, "rb") as f:
        metadata_df = pickle.load(f)

    metadata_df = metadata_df[metadata_df["bin_exists"] == True].head(max_files)

    print(
        f"{len(metadata_df)} fichiers BIN valides "
        f"(environ {len(metadata_df) * num_windows} fenetres)"
    )

    all_features = []

    for _, row in metadata_df.iterrows():
        try:
            iq = read_iq_file(
                row["bin_path"],
                row["iq_dataform"],
                remove_dc=True
            )

            windows = get_windows(iq, n_windows=num_windows)

            for win_id, win_iq in enumerate(windows):
                feats = compute_17_features(
                    win_iq,
                    fs=float(row["sample_rate_hz"]),
                    center_freq=float(row["center_freq_hz"]),
                    bandwidth=float(row["bandwidth_hz"]),
                    total_samples=int(row["num_samples"]),
                    num_windows=num_windows
                )

                feats.update({
                    "file_id": row["json_file"],
                    "window_id": int(win_id),
                    "datetime": row.get("datetime", ""),
                    "nanosec": int(row.get("nanosec", 0))
                })

                all_features.append(feats)

        except Exception as e:
            print(f"Erreur sur {row['json_file']} : {e}")

    df = pd.DataFrame(all_features)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_pickle(FEATURES_PKL)

    print(f"features_17.pkl cree : {len(df)} lignes")

    return df