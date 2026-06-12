# phase3_compare_arocle_rules_vs_model.py

import pandas as pd
from pathlib import Path


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_RULES = OUT_DIR / "WASD_RF_arocle_labeled.csv"
CSV_MODEL = OUT_DIR / "WASD_RF_arocle_scored.csv"

CSV_DIFFS = OUT_DIR / "arocle_rules_vs_model_diffs.csv"


def main():
    """
    Compare les labels produits par les règles expertes AROCLE-RF
    avec les prédictions du modèle appris.

    Cette étape sert à vérifier si le modèle ML reproduit correctement
    le comportement du système expert.
    """

    # =========================
    # Vérification des fichiers
    # =========================

    if not CSV_RULES.exists():
        raise FileNotFoundError(f"Fichier règles introuvable: {CSV_RULES}")

    if not CSV_MODEL.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable: {CSV_MODEL}")

    # =========================
    # Chargement des données
    # =========================

    df_rules = pd.read_csv(CSV_RULES, sep=";", decimal=",", encoding="utf-8-sig")
    df_model = pd.read_csv(CSV_MODEL, sep=";", decimal=",", encoding="utf-8-sig")

    df_rules.columns = df_rules.columns.str.strip()
    df_model.columns = df_model.columns.str.strip()

    # =========================
    # Vérification des clés
    # =========================

    for key in ["file_id", "window_id"]:
        if key not in df_rules.columns:
            raise ValueError(f"{key} manquant dans {CSV_RULES.name}")

        if key not in df_model.columns:
            raise ValueError(f"{key} manquant dans {CSV_MODEL.name}")

    # =========================
    # Harmonisation des types
    # =========================
    # Cette étape évite les problèmes de jointure dus aux espaces,
    # aux formats numériques ou aux différences d’encodage.

    df_rules["file_id"] = df_rules["file_id"].astype(str).str.strip()
    df_model["file_id"] = df_model["file_id"].astype(str).str.strip()

    df_rules["window_id"] = (
        pd.to_numeric(df_rules["window_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    df_model["window_id"] = (
        pd.to_numeric(df_model["window_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    # =========================
    # Vérification des colonnes métier
    # =========================

    if "rf_pattern_arocle" not in df_rules.columns:
        raise ValueError("Colonne rf_pattern_arocle manquante dans le CSV des règles")

    if "arocle_rf_pattern" not in df_model.columns:
        raise ValueError("Colonne arocle_rf_pattern manquante dans le CSV du modèle")

    if "arocle_rf_confidence" not in df_model.columns:
        raise ValueError("Colonne arocle_rf_confidence manquante : predict_proba requis")

    rules_cols = ["file_id", "window_id", "rf_pattern_arocle"]
    model_cols = ["file_id", "window_id", "arocle_rf_pattern", "arocle_rf_confidence"]

    # =========================
    # Alignement règles / modèle
    # =========================

    df = df_rules[rules_cols].merge(
        df_model[model_cols],
        on=["file_id", "window_id"],
        how="inner",
        validate="one_to_one"
    )

    # =========================
    # Comparaison des décisions
    # =========================

    df["is_match"] = (
        df["rf_pattern_arocle"].astype(str)
        == df["arocle_rf_pattern"].astype(str)
    )

    diffs = df[~df["is_match"]].copy()

    # =========================
    # Statistiques globales
    # =========================

    total = len(df)
    ndiff = len(diffs)
    accuracy = 1 - (ndiff / total) if total else 0

    print(f"\nTotal comparé : {total}")
    print(f"Différences : {ndiff} ({(ndiff / total * 100 if total else 0):.2f}%)")
    print(f"Fidélité du modèle : {accuracy * 100:.2f}%")

    if ndiff == 0:
        print("Le modèle reproduit exactement les règles AROCLE-RF.")
        return

    # =========================
    # Analyse par classe
    # =========================

    print("\nTaux de correspondance par classe AROCLE :")
    print(
        df.groupby("rf_pattern_arocle")["is_match"]
        .mean()
        .sort_values()
    )

    # =========================
    # Analyse des confusions
    # =========================

    print("\nConfusions observees (regles -> modele) :")
    print(
        diffs.groupby(["rf_pattern_arocle", "arocle_rf_pattern"])
        .size()
        .sort_values(ascending=False)
    )

    # =========================
    # Analyse de la confiance
    # =========================

    print("\nDistribution de confiance sur les erreurs :")
    print(diffs["arocle_rf_confidence"].describe())

    # Les erreurs à faible confiance sont particulièrement intéressantes :
    # elles indiquent souvent des frontières ambiguës entre deux classes RF.
    difficult_cases = diffs.sort_values(
        "arocle_rf_confidence",
        ascending=True
    ).head(30)

    print("\nTop 30 des erreurs les moins confiantes :")
    print(difficult_cases[[
        "file_id", "window_id",
        "rf_pattern_arocle",
        "arocle_rf_pattern",
        "arocle_rf_confidence"
    ]])

    # =========================
    # Sauvegarde des divergences
    # =========================

    diffs.to_csv(CSV_DIFFS, index=False, sep=";", decimal=",")

    print("\nDifférences sauvegardées :", CSV_DIFFS)


if __name__ == "__main__":
    main()