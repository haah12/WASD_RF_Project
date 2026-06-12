# build_csv.py

import pandas as pd
from pathlib import Path


# Dossier où seront sauvegardées les sorties du pipeline
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_final_csv(
    features_pkl: str = "data/features_17.pkl",
    output_csv: str = str(OUTPUT_DIR / "WASD_RF_dataset.csv")
) -> pd.DataFrame:
    """
    Construit le dataset RF final à partir des features extraites.

    Ce fichier représente la sortie principale de la phase 1.
    Il regroupe les descripteurs physiques du signal RF et ajoute
    un premier étiquetage simple du comportement observé.
    """

    # Chargement des features calculées précédemment à partir des fenêtres IQ
    df = pd.read_pickle(features_pkl).copy()

    # Par défaut, chaque fenêtre est considérée comme un comportement normal.
    # Les règles suivantes permettent ensuite d'identifier quelques patterns RF simples.
    df["rf_pattern"] = "normal"

    # Signal fortement bruité : le rapport signal/bruit est trop faible.
    df.loc[df["SNR_dB"] < 5.0, "rf_pattern"] = "low_snr"

    # Chirp : occupation spectrale importante, mais sans pic spectral dominant.
    df.loc[
        (df["occupation_spectrale"] > 0.4) &
        (df["ISR_dB"] < 10.0) &
        (df["SNR_dB"] > 5.0),
        "rf_pattern"
    ] = "chirp"

    # Tone : présence d'un pic spectral marqué.
    df.loc[df["ISR_dB"] > 15.0, "rf_pattern"] = "tone"

    # Pulse : instabilité ou distorsion importante du signal.
    df.loc[df["EVM_percent"] > 15.0, "rf_pattern"] = "pulse"

    # Colonnes gardées dans le dataset final.
    # Certaines colonnes peuvent être absentes selon les fichiers traités,
    # donc on filtre la liste avant l'export.
    cols = [
        "time_window_ms", "center_frequency_hz", "bandwidth_hz",
        "RSSI_dB", "SNR_dB", "puissance_moyenne", "variance_IQ",
        "kurtosis", "skewness", "PSD_mean", "occupation_spectrale", "ISR_dB",
        "RSRP_dB", "RSRQ_dB", "EVM_percent", "BER_est",
        "file_id", "window_id", "rf_pattern"
    ]

    cols = [col for col in cols if col in df.columns]
    out_df = df[cols]

    # Export du dataset structuré.
    # Le séparateur ';' et la virgule décimale facilitent l'ouverture sous Excel.
    out_df.to_csv(output_csv, index=False, sep=";", decimal=",")

    print(
        f"CSV final créé : {output_csv} | "
        f"lignes = {len(out_df)} | colonnes = {len(cols)}"
    )

    return out_df


if __name__ == "__main__":
    build_final_csv()