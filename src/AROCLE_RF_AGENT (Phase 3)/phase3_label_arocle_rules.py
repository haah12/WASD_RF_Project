# phase3_label_arocle_rules.py

import pandas as pd
from pathlib import Path


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_AROCLE = OUT_DIR / "WASD_RF_arocle_features.csv"
CSV_SCORED = OUT_DIR / "WASD_RF_dataset_scored.csv"
CSV_OUT = OUT_DIR / "WASD_RF_arocle_labeled.csv"


def main():
    """
    Applique les règles expertes AROCLE-RF.

    Cette étape transforme les anomalies RF détectées en classes de menace
    interprétables : A, R, O, C, L ou E.
    """

    # =========================
    # Chargement des données
    # =========================

    df_arocle = pd.read_csv(CSV_AROCLE, sep=";", decimal=",", encoding="utf-8-sig")
    df_scored = pd.read_csv(CSV_SCORED, sep=";", decimal=",", encoding="utf-8-sig")

    df_arocle.columns = df_arocle.columns.str.strip()
    df_scored.columns = df_scored.columns.str.strip()

    # Harmonisation des clés de jointure
    df_arocle["file_id"] = df_arocle["file_id"].astype(str).str.strip()
    df_scored["file_id"] = df_scored["file_id"].astype(str).str.strip()

    df_arocle["window_id"] = (
        pd.to_numeric(df_arocle["window_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    df_scored["window_id"] = (
        pd.to_numeric(df_scored["window_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    # On récupère les informations issues de la Phase 2
    keep_cols = ["file_id", "window_id", "is_anomaly", "anomaly_score"]

    df_arocle = df_arocle.drop(
        columns=["is_anomaly", "anomaly_score"],
        errors="ignore"
    )

    df = df_arocle.merge(
        df_scored[keep_cols],
        on=["file_id", "window_id"],
        how="left"
    )

    # =========================
    # Score de risque intermédiaire
    # =========================
    # Ce score combine plusieurs signes précurseurs :
    # - intensité de l’anomalie,
    # - augmentation brutale du score,
    # - chute du SNR,
    # - augmentation de l’EVM.
    #
    # Il sert à distinguer les fenêtres normales, pré-attaque et attaque.

    df["risk_score"] = (
        0.35 * df["anomaly_score"].fillna(0)
        + 0.35 * df["delta_anomaly_score"].fillna(0).clip(lower=0)
        + 0.20 * (-df["delta_snr"].fillna(0).clip(upper=0))
        + 0.10 * df["delta_evm"].fillna(0).clip(lower=0)
    )

    # =========================
    # Détermination de l’état RF
    # =========================

    PRE_ATTACK_TH = 0.25
    ATTACK_TH = 0.60

    df["state"] = "normal"

    df.loc[df["risk_score"] >= PRE_ATTACK_TH, "state"] = "pre_attack"
    df.loc[df["risk_score"] >= ATTACK_TH, "state"] = "attack"

    # Une montée rapide du score d’anomalie peut être considérée
    # comme un signe précoce, même si le score global reste modéré.
    df.loc[
        (df["delta_anomaly_score"] > 0.02) &
        (df["state"] == "normal"),
        "state"
    ] = "pre_attack"

    cond_pre = df["state"].isin(["pre_attack", "attack"])

    # =========================
    # Seuils adaptatifs
    # =========================
    # Les seuils sont calculés à partir de la distribution réelle des données.
    # Cela évite de figer des valeurs trop dépendantes d’un seul scénario.

    snr_th = df["delta_snr"].quantile(0.15)
    evm_th = df["delta_evm"].quantile(0.70)
    occ_th = df["occupation_spectrale"].quantile(0.70)

    cos_prev_th = df["cos_sim_prev"].quantile(0.70)
    cos_02_th = df["cos_sim_0_2"].quantile(0.70)
    cos_13_th = df["cos_sim_1_3"].quantile(0.70)

    # =========================
    # Classification AROCLE-RF
    # =========================

    df["rf_pattern_arocle"] = "normal"

    # A — Availability Pressure
    # Chute du SNR : signe typique d’un brouillage ou d’une pression sur le canal.
    df.loc[
        cond_pre &
        (df["delta_snr"] < snr_th),
        "rf_pattern_arocle"
    ] = "A"

    # R — Radio Distortion
    # Hausse de l’EVM : le signal semble déformé ou moins propre.
    df.loc[
        cond_pre &
        (df["rf_pattern_arocle"] == "normal") &
        (df["delta_evm"] > evm_th),
        "rf_pattern_arocle"
    ] = "R"

    # O — Origin Deception
    # Similarité élevée avec la fenêtre précédente : comportement trop stable,
    # potentiellement compatible avec une imitation de source.
    df.loc[
        cond_pre &
        (df["rf_pattern_arocle"] == "normal") &
        (df["cos_sim_prev"] > cos_prev_th),
        "rf_pattern_arocle"
    ] = "O"

    # L — Latency Manipulation / Replay
    # Répétition temporelle entre fenêtres non consécutives.
    df.loc[
        cond_pre &
        (df["rf_pattern_arocle"] == "normal") &
        (
            (df["cos_sim_0_2"] > cos_02_th) |
            (df["cos_sim_1_3"] > cos_13_th)
        ),
        "rf_pattern_arocle"
    ] = "L"

    # C — Covert Leakage / activité spectrale inhabituelle
    # Occupation spectrale élevée sans autre signature dominante.
    df.loc[
        cond_pre &
        (df["rf_pattern_arocle"] == "normal") &
        (df["occupation_spectrale"] > occ_th),
        "rf_pattern_arocle"
    ] = "C"

    # R — fallback distorsion
    # Règle complémentaire pour capturer des distorsions plus faibles.
    df.loc[
        cond_pre &
        (df["rf_pattern_arocle"] == "normal") &
        (df["delta_evm"] > df["delta_evm"].quantile(0.60)),
        "rf_pattern_arocle"
    ] = "R"

    # E — Escalation Capability
    # Classe de repli pour les comportements suspects qui ne correspondent
    # pas clairement aux autres signatures AROCLE.
    df.loc[
        cond_pre &
        (df["rf_pattern_arocle"] == "normal"),
        "rf_pattern_arocle"
    ] = "E"

    # =========================
    # Résumé des résultats
    # =========================

    print("\nRépartition AROCLE-RF :")
    print(df["rf_pattern_arocle"].value_counts())

    print("\nRépartition des états RF :")
    print(df["state"].value_counts())

    print("\nCroisement classe AROCLE / état RF :")
    print(pd.crosstab(df["rf_pattern_arocle"], df["state"]))

    # =========================
    # Sauvegarde
    # =========================

    df.to_csv(CSV_OUT, index=False, sep=";", decimal=",")

    print("\nLabels AROCLE-RF sauvegardés :", CSV_OUT)


if __name__ == "__main__":
    main()