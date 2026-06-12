# phase2_select_features.py

from pathlib import Path
import pandas as pd


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_IN = PROJECT_ROOT / "outputs" / "WASD_RF_dataset.csv"

print("Chemin du CSV :", CSV_IN)
print("Fichier présent ?", CSV_IN.exists())


# =========================
# Chargement des données
# =========================

df = pd.read_csv(CSV_IN, sep=";", decimal=",")


# =========================
# Sélection des features
# =========================

# Colonnes à exclure :
# - identifiants (file_id, window_id)
# - informations temporelles
# - labels ou sorties des phases suivantes
EXCLUDE = {
    "file_id", "window_id", "datetime", "nanosec",
    "rf_pattern", "label", "is_anomaly", "anomaly_score"
}

# On ne conserve que les colonnes numériques,
# car l’autoencoder ne traite que des valeurs continues
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

# Filtrage final des features utilisées pour l’apprentissage
features = [col for col in numeric_cols if col not in EXCLUDE]

print(f"\nNombre de features candidates : {len(features)}")
print(features)


# =========================
# Vérification qualité des données
# =========================

# On vérifie ici la présence éventuelle de valeurs invalides
# (NaN ou infini) qui pourraient perturber l'entraînement du modèle
invalid_counts = (
    df[features]
    .replace([float("inf"), float("-inf")], pd.NA)
    .isna()
    .sum()
)

invalid_counts = invalid_counts[invalid_counts > 0]

print("\nVérification NaN / Inf (vide => données propres) :")
print(invalid_counts)