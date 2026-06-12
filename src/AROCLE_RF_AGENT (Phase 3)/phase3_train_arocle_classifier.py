# phase3_train_arocle_classifier.py
# Version anticipative

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import joblib
import matplotlib.pyplot as plt


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

CSV_IN = OUT_DIR / "WASD_RF_arocle_labeled.csv"

MODEL_OUT = OUT_DIR / "arocle_rf_model.joblib"
REPORT_OUT = OUT_DIR / "arocle_rf_report.txt"
CM_OUT = OUT_DIR / "arocle_rf_confusion.png"
IMP_OUT = OUT_DIR / "arocle_rf_feature_importance.png"


# =========================
# Features du modèle AROCLE-RF
# =========================

FEATURES = [
    'RSSI_dB', 'SNR_dB', 'puissance_moyenne', 'variance_IQ',
    'kurtosis', 'skewness', 'PSD_mean', 'occupation_spectrale', 'ISR_dB',
    'RSRP_dB', 'RSRQ_dB', 'EVM_percent', 'BER_est',
    'anomaly_score',
    'delta_snr', 'delta_evm', 'cos_sim_prev',
    'cfo_hz', 'delta_cfo_hz',
    'cos_sim_0_2', 'cos_sim_1_3'
]


def main():
    """
    Entraîne le classifieur AROCLE-RF à partir des labels générés
    par les règles expertes.

    Dans cette version, le modèle apprend uniquement sur les fenêtres
    suspectes ou confirmées comme attaque, afin de renforcer la logique
    anticipative de la Phase 3.
    """

    # =========================
    # Chargement des données
    # =========================

    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # =========================
    # Filtrage anticipatif
    # =========================
    # On retire les fenêtres normales pour entraîner le modèle uniquement
    # sur les situations intéressantes du point de vue cybersécurité :
    # pré-attaque et attaque.

    df_model = df[df["state"].isin(["pre_attack", "attack"])].copy()

    # Cible : classe de menace AROCLE-RF
    y = df_model["rf_pattern_arocle"].astype(str)

    # Features numériques
    Xdf = df_model[FEATURES].apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    # =========================
    # Découpage apprentissage / test
    # =========================
    # Le split stratifié conserve la répartition des classes AROCLE
    # dans les deux sous-ensembles.

    Xtr, Xte, ytr, yte = train_test_split(
        Xdf,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # =========================
    # Modèle de classification
    # =========================
    # Le pipeline contient :
    # - une imputation par médiane pour les valeurs manquantes,
    # - un Random Forest robuste aux données bruitées et non linéaires.

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    clf.fit(Xtr, ytr)

    # =========================
    # Prédiction et évaluation
    # =========================

    pred = clf.predict(Xte)

    # predict_proba sera utilisé ensuite par la phase d’inférence
    # pour calculer la confiance du modèle.
    _ = clf.predict_proba(Xte)

    labels = sorted(y.unique().tolist())

    report = classification_report(
        yte,
        pred,
        labels=labels,
        digits=4,
        zero_division=0
    )

    print(report)
    REPORT_OUT.write_text(report, encoding="utf-8")

    # =========================
    # Matrice de confusion
    # =========================

    cm = confusion_matrix(yte, pred, labels=labels)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matrice de confusion AROCLE-RF anticipative")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=200)
    plt.close()

    # =========================
    # Importance des features
    # =========================
    # Cette figure aide à interpréter les critères RF qui influencent
    # le plus la décision du classifieur.

    rf = clf.named_steps["rf"]
    importances = rf.feature_importances_

    idx = np.argsort(importances)[::-1]
    ordered_features = [FEATURES[i] for i in idx]
    ordered_importances = importances[idx]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(ordered_features)), ordered_importances)
    plt.xticks(range(len(ordered_features)), ordered_features, rotation=90)
    plt.title("Importance des features - AROCLE-RF anticipatif")
    plt.tight_layout()
    plt.savefig(IMP_OUT, dpi=200)
    plt.close()

    # =========================
    # Sauvegarde du modèle
    # =========================

    joblib.dump(clf, MODEL_OUT)

    print("\nModèle AROCLE-RF sauvegardé :", MODEL_OUT)


if __name__ == "__main__":
    main()