from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_IN = PROJECT_ROOT / "outputs" / "WASD_RF_dataset.csv"

EXCLUDE = {
    "file_id", "window_id", "datetime", "nanosec",
    "rf_pattern", "label", "is_anomaly", "anomaly_score"
}

print("CSV path =", CSV_IN)
print("Exists ?", CSV_IN.exists())


df = pd.read_csv(CSV_IN, sep=";", decimal=",")

# 1) Colonnes numériques uniquement
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

# 2) Retirer celles qu’on ne veut pas
features = [c for c in numeric_cols if c not in EXCLUDE]

print(" Features candidates (numériques) =", len(features))
print(features)

# 3) Vérification NaN/Inf
bad = df[features].replace([float("inf"), float("-inf")], pd.NA).isna().sum()
bad = bad[bad > 0]
print("\nNaN/Inf par feature (si vide => OK):")
print(bad)
