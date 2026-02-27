# src/phase3_compare_rules_vs_model_v2.py

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_RULES = OUT_DIR / "WASD_RF_oracle_labeled_v2.csv"   # règles (rf_pattern_v2)
CSV_MODEL = OUT_DIR / "WASD_RF_oracle_scored_v2.csv"    # modèle (oracle_rf_pattern + oracle_rf_confidence)

CSV_DIFFS = OUT_DIR / "oracle_rules_vs_model_diffs_v2.csv"


def main():
    # 1) Load
    df_rules = pd.read_csv(CSV_RULES, sep=";", decimal=",", encoding="utf-8-sig")
    df_model = pd.read_csv(CSV_MODEL, sep=";", decimal=",", encoding="utf-8-sig")

    # 2) Clean column names
    df_rules.columns = df_rules.columns.str.strip()
    df_model.columns = df_model.columns.str.strip()

    # 3) Ensure keys exist
    for k in ["file_id", "window_id"]:
        if k not in df_rules.columns:
            raise ValueError(f" {k} manquant dans {CSV_RULES.name}")
        if k not in df_model.columns:
            raise ValueError(f" {k} manquant dans {CSV_MODEL.name}")

    # 4) Harmonize key types (avoid merge mismatches)
    df_rules["file_id"] = df_rules["file_id"].astype(str).str.strip()
    df_model["file_id"] = df_model["file_id"].astype(str).str.strip()

    df_rules["window_id"] = pd.to_numeric(df_rules["window_id"], errors="coerce").astype("Int64").astype(str)
    df_model["window_id"] = pd.to_numeric(df_model["window_id"], errors="coerce").astype("Int64").astype(str)

    # 5) Select minimal cols from each side
    if "rf_pattern_v2" not in df_rules.columns:
        raise ValueError(" rf_pattern_v2 manquant dans le CSV règles")
    if "oracle_rf_pattern" not in df_model.columns:
        raise ValueError(" oracle_rf_pattern manquant dans le CSV modèle")
    if "oracle_rf_confidence" not in df_model.columns:
        raise ValueError(" oracle_rf_confidence manquant dans le CSV modèle (active predict_proba dans infer)")

    rules_cols = ["file_id", "window_id", "rf_pattern_v2"]
    model_cols = ["file_id", "window_id", "oracle_rf_pattern", "oracle_rf_confidence"]

    # 6) Merge
    df = df_rules[rules_cols].merge(
        df_model[model_cols],
        on=["file_id", "window_id"],
        how="inner",
        validate="one_to_one"
    )

    # 7) Filter diffs
    diffs = df[df["rf_pattern_v2"].astype(str) != df["oracle_rf_pattern"].astype(str)].copy()

    # 8) Summary
    total = len(df)
    ndiff = len(diffs)
    print(f"\n Total comparé: {total}")
    print(f" Différences (rules != model): {ndiff} ({(ndiff/total*100 if total else 0):.2f}%)")

    if ndiff == 0:
        print(" Aucune différence. Le modèle reproduit exactement les règles sur l'ensemble comparé.")
        return

    print("\n Types de confusions (rules -> model):")
    print(diffs.groupby(["rf_pattern_v2", "oracle_rf_pattern"]).size().sort_values(ascending=False))

    # 9) Top 30 lowest confidence among diffs
    top30 = diffs.sort_values("oracle_rf_confidence", ascending=True).head(30)

    print("\n Top 30 différences avec confidence la plus faible:")
    print(top30[["file_id", "window_id", "rf_pattern_v2", "oracle_rf_pattern", "oracle_rf_confidence"]])

    # 10) Save diffs for inspection
    diffs.to_csv(CSV_DIFFS, index=False, sep=";", decimal=",")
    print("\n Diffs saved:", CSV_DIFFS)


if __name__ == "__main__":
    main()
