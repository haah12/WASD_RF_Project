# features.py
import numpy as np
import pandas as pd
from scipy import signal, stats
import pickle
import os
from pathlib import Path

# -----------------------------
# Lecture IQ (utilise ton BIN)
# -----------------------------
def read_iq_file(bin_path: str, dataform: int = 2, remove_dc: bool = True) -> np.ndarray:
    """
    Lecture IQ interleavé I,Q :
    - intX -> float32 en [-1, 1)
    - suppression DC optionnelle
     PAS de normalisation RMS ici (sinon RSSI devient artificiel)
    """
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(int(dataform), np.int16)

    if not os.path.exists(bin_path):
        raise FileNotFoundError(f" BIN manquant: {bin_path}")

    raw = np.fromfile(bin_path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f" BIN vide: {bin_path}")
    if raw.size % 2 != 0:
        raise ValueError(f" Taille impaire (I/Q): {raw.size}")

    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)

    full_scale = float(2 ** (np.iinfo(dtype).bits - 1))  # 32768 pour int16
    iq = (i / full_scale) + 1j * (q / full_scale)

    if remove_dc:
        iq = iq - np.mean(iq)

    return iq


# -----------------------------
# Fenêtrage robuste
# -----------------------------
def window_signal(iq: np.ndarray, num_windows: int = 4) -> list:
    n = len(iq)
    if num_windows <= 1 or n < num_windows:
        return [iq]
    chunk = n // num_windows
    wins = []
    for k in range(num_windows):
        start = k * chunk
        end = (k + 1) * chunk if k < num_windows - 1 else n
        wins.append(iq[start:end])
    return wins


# -----------------------------
# PSD helper
# -----------------------------
def welch_psd_db(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Retourne PSD en dB (vector) trié en fréquence.
    """
    if len(x) < 512:
        # fallback très court
        psd = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
        psd = psd / (len(x) + 1e-12)
        return 10 * np.log10(psd + 1e-18)

    nperseg = min(4096, max(512, len(x)//8))
    f, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        return_onesided=False,
        scaling="density"
    )
    psd = np.abs(psd) + 1e-18
    idx = np.argsort(f)
    psd = psd[idx]
    return 10 * np.log10(psd)


# -----------------------------
# 17 features (corrigées)
# -----------------------------
def compute_17_features(window_iq: np.ndarray, fs: float, center_freq: float,
                        bandwidth: float, total_samples: int = None,
                        num_windows: int = 4) -> dict:
    eps = 1e-12
    x = window_iq

    # Durée fenêtre (ms) — force float (évite format date)
    if total_samples is not None and fs > 0:
        total_duration_ms = (float(total_samples) / float(fs)) * 1000.0
        time_window_ms = total_duration_ms / float(num_windows)
    else:
        time_window_ms = (len(x) / float(fs)) * 1000.0 if fs > 0 else 0.0
    time_window_ms = float(time_window_ms)

    # Puissance et RSSI en dBFS (relatif ADC)
    power = float(np.mean(np.abs(x)**2))
    rssi_db = float(10.0 * np.log10(power + eps))
    variance_iq = float(np.var(x))

    # Moments
    kurtosis_val = float(stats.kurtosis(np.real(x), fisher=True, bias=False))
    skewness_val = float(stats.skew(np.real(x), bias=False))

    # PSD (en dB) + stats
    psd_db = welch_psd_db(x, fs)
    psd_lin = 10 ** (psd_db / 10.0)
    psd_mean = float(np.mean(psd_lin))  # moyenne en linéaire

    # --- SNR robuste (évite -120 constant)
    # Noise floor = percentile 20%
    # Signal level = percentile 95%
    noise_floor_db = float(np.percentile(psd_db, 20))
    signal_level_db = float(np.percentile(psd_db, 95))
    snr_db = float(signal_level_db - noise_floor_db)

    # Clamp raisonnable pour stabilité
    snr_db = float(np.clip(snr_db, -10.0, 60.0))

    # --- Occupation spectrale (robuste)
    # bins dont PSD > noise + 6 dB
    spectral_occ = float(np.mean(psd_db > (noise_floor_db + 6.0)))
    spectral_occ = float(np.clip(spectral_occ, 0.0, 1.0))

    # --- ISR (pic vs médiane), en dB
    isr_db = float(np.max(psd_db) - np.median(psd_db))
    isr_db = float(np.clip(isr_db, 0.0, 80.0))

    # --- RSRP/RSRQ (approx cohérente)
    # N_RB ~ BW/180kHz
    n_rb = max(float(bandwidth) / 180e3, 1.0)
    rsrp_db = float(rssi_db - 10.0 * np.log10(n_rb))

    # Evite explosion si occ ~ 0
    occ_safe = max(spectral_occ, 1e-3)
    rsrq_db = float(rsrp_db - 10.0 * np.log10(occ_safe))

    # --- EVM (%) cohérente avec SNR
    # EVM% = 100 * 10^(-SNR/20)
    evm_percent = float(100.0 * (10.0 ** (-snr_db / 20.0)))

    # --- BER estimate (indicatif, borné)
    ber_est = float(np.clip(0.5 * np.exp(-snr_db / 10.0), 1e-9, 0.5))

    return {
        'time_window_ms': float(round(time_window_ms, 4)),
        'center_frequency_hz': float(center_freq),
        'bandwidth_hz': float(bandwidth),

        'RSSI_dB': float(round(rssi_db, 4)),
        'SNR_dB': float(round(snr_db, 4)),

        'puissance_moyenne': float(power),
        'variance_IQ': float(variance_iq),

        'kurtosis': float(round(kurtosis_val, 6)),
        'skewness': float(round(skewness_val, 6)),

        'PSD_mean': float(psd_mean),
        'occupation_spectrale': float(round(spectral_occ, 6)),
        'ISR_dB': float(round(isr_db, 4)),

        'RSRP_dB': float(round(rsrp_db, 4)),
        'RSRQ_dB': float(round(rsrq_db, 4)),

        'EVM_percent': float(round(evm_percent, 4)),
        'BER_est': float(ber_est),

        # label laissé à normal ici (build_csv décidera)
        'rf_pattern': 'normal'

    }


def extract_all_features(max_files: int = 505, num_windows: int = 4) -> pd.DataFrame:
    """
    Extrait features à partir de data/metadata.pkl (généré par ingest.py).
    """
    with open('data/metadata.pkl', 'rb') as f:
        metadata_df = pickle.load(f)

    metadata_df = metadata_df[metadata_df['bin_exists'] == True].head(max_files)
    print(f"🔬 {len(metadata_df)} fichiers BIN OK (≈ {len(metadata_df)*num_windows} fenêtres)")

    all_features = []
    for idx, row in metadata_df.iterrows():
        try:
            iq = read_iq_file(row['bin_path'], row['iq_dataform'], remove_dc=True)
            windows = window_signal(iq, num_windows=num_windows)

            for win_id, win_iq in enumerate(windows):
                feats = compute_17_features(
                    win_iq,
                    fs=float(row['sample_rate_hz']),
                    center_freq=float(row['center_freq_hz']),
                    bandwidth=float(row['bandwidth_hz']),
                    total_samples=int(row['num_samples']),
                    num_windows=num_windows
                )
                feats.update({
                    'file_id': row['json_file'],
                    'window_id': int(win_id),
                    'datetime': row.get('datetime', ''),
                    'nanosec': int(row.get('nanosec', 0))
                })
                all_features.append(feats)

        except Exception as e:
            print(f" Erreur {row['json_file']}: {e}")

    df = pd.DataFrame(all_features)
    Path('data').mkdir(exist_ok=True)
    df.to_pickle('data/features_17.pkl')
    print(f" features_17.pkl créé: {len(df)} lignes")
    return df
