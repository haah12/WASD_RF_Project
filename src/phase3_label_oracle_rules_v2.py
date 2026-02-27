# phase3_label_oracle_rules_v2.py
import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_ORACLE = OUT_DIR / "WASD_RF_oracle_features_v2.csv"
CSV_SCORED = OUT_DIR / "WASD_RF_dataset_scored.csv"
CSV_OUT = OUT_DIR / "WASD_RF_oracle_labeled_v2.csv"


def main():
    # 1) Charger features ORACLE v2
    df_oracle = pd.read_csv(CSV_ORACLE, sep=";", decimal=",", encoding="utf-8-sig")
    df_oracle.columns = df_oracle.columns.str.strip()

    # 2) Charger CSV Phase 2 scoré
    df_scored = pd.read_csv(CSV_SCORED, sep=";", decimal=",", encoding="utf-8-sig")
    df_scored.columns = df_scored.columns.str.strip()

    # 3) Harmoniser clés
    df_oracle["file_id"] = df_oracle["file_id"].astype(str).str.strip()
    df_scored["file_id"] = df_scored["file_id"].astype(str).str.strip()

    df_oracle["window_id"] = pd.to_numeric(df_oracle["window_id"], errors="coerce").astype("Int64").astype(str)
    df_scored["window_id"] = pd.to_numeric(df_scored["window_id"], errors="coerce").astype("Int64").astype(str)

    # 4) Colonnes à merger
    keep_cols = ["file_id", "window_id", "is_anomaly", "anomaly_score"]
    if "threshold_p99" in df_scored.columns:
        keep_cols.append("threshold_p99")

    # éviter conflits
    df_oracle = df_oracle.drop(
        columns=[c for c in ["is_anomaly", "anomaly_score", "threshold_p99"] if c in df_oracle.columns],
        errors="ignore"
    )

    # 5) Merge
    df = df_oracle.merge(
        df_scored[keep_cols],
        on=["file_id", "window_id"],
        how="left",
        validate="many_to_one"
    )

    # 6) Vérifs
    if df["is_anomaly"].isna().any():
        raise ValueError(" Merge incomplet: clés file_id/window_id non alignées")

    # 7) Init labels
    df["rf_pattern_v2"] = "normal"
    anom = (df["is_anomaly"].astype(int) == 1)

    # --------------------------
    # Règles ORACLE RF v2
    # --------------------------

    # L (Replay)
    df.loc[
        anom & ((df["cos_sim_0_2"] >= 0.995) | (df["cos_sim_1_3"] >= 0.995)),
        "rf_pattern_v2"
    ] = "L"

    # A (Availability / Jamming)
    df.loc[
        anom &
        (df["rf_pattern_v2"] == "normal") &
        (df["occupation_spectrale"] >= 0.85) &
        (df["SNR_dB"] <= 8) &
        (df["delta_snr"] <= 0),
        "rf_pattern_v2"
    ] = "A"

    # R (Radio Distortion)
    df.loc[
        anom &
        (df["rf_pattern_v2"] == "normal") &
        (df["EVM_percent"] >= 20) &
        (df["ISR_dB"] >= 1.5) &
        ((df["kurtosis"].abs() >= 0.5) | (df["skewness"].abs() >= 0.2)),
        "rf_pattern_v2"
    ] = "R"

    # O (Spoofing)
    if "delta_cfo_khz" not in df.columns and "delta_cfo_hz" in df.columns:
        df["delta_cfo_khz"] = df["delta_cfo_hz"] / 1e3

    if "delta_cfo_khz" not in df.columns:
        df["delta_cfo_khz"] = np.nan

    df.loc[
        anom &
        (df["rf_pattern_v2"] == "normal") &
        (df["cos_sim_prev"] >= 0.995) &
        (df["delta_snr"].abs() <= 1.0) &
        (df["delta_evm"].abs() <= 2.0) &
        (df["delta_cfo_khz"].abs() <= 10.0),
        "rf_pattern_v2"
    ] = "O"

    # C (Covert leakage)
    df.loc[
        anom &
        (df["rf_pattern_v2"] == "normal") &
        (df["occupation_spectrale"].between(0.70, 0.90)) &
        (df["SNR_dB"] >= 10) &
        (df["EVM_percent"] <= 18),
        "rf_pattern_v2"
    ] = "C"

    # E (fallback)
    df.loc[
        anom & (df["rf_pattern_v2"] == "normal"),
        "rf_pattern_v2"
    ] = "E"

    # 8) Sauvegarde
    df.to_csv(CSV_OUT, index=False, sep=";", decimal=",")
    print(" Saved:", CSV_OUT)

    print("\n Répartition ORACLE RF v2:")
    print(df["rf_pattern_v2"].value_counts())

    print("\n Check is_anomaly:")
    print(df["is_anomaly"].value_counts())

    # ======================================================
    #  DEBUG STATS – POURQUOI A / R NE SORTENT PAS
    # ======================================================
    print("\n Stats sur les ANOMALIES uniquement (is_anomaly = 1):")

    debug_cols = [
        "occupation_spectrale",
        "SNR_dB",
        "delta_snr",
        "EVM_percent",
        "ISR_dB",
        "kurtosis",
        "skewness"
    ]

    existing_cols = [c for c in debug_cols if c in df.columns]

    stats = (
        df.loc[anom, existing_cols]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .T[["min", "5%", "25%", "50%", "75%", "95%", "max"]]
    )

    print(stats)


if __name__ == "__main__":
    main()
