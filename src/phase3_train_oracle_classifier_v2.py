# phase3_train_oracle_classifier_v2.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
CSV_IN = OUT_DIR / "WASD_RF_oracle_labeled_v2.csv"

MODEL_OUT = OUT_DIR / "oracle_rf_model_v2.joblib"
REPORT_OUT = OUT_DIR / "oracle_rf_report_v2.txt"
CM_OUT = OUT_DIR / "oracle_rf_confusion_v2.png"
IMP_OUT = OUT_DIR / "oracle_rf_feature_importance_v2.png"

FEATURES = [
    # base RF
    'RSSI_dB','SNR_dB','puissance_moyenne','variance_IQ',
    'kurtosis','skewness','PSD_mean','occupation_spectrale','ISR_dB',
    'RSRP_dB','RSRQ_dB','EVM_percent','BER_est',
    # phase 2
    'anomaly_score',
    # v2 temporal + CFO proxy
    'delta_snr','delta_evm','cos_sim_prev','cfo_hz','delta_cfo_hz','cos_sim_0_2','cos_sim_1_3'
]

def main():
    df = pd.read_csv(CSV_IN, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Vérifier colonnes nécessaires
    missing = [c for c in FEATURES + ["rf_pattern_v2"] if c not in df.columns]
    if missing:
        raise ValueError(f" Colonnes manquantes dans {CSV_IN.name}: {missing}")

    y = df["rf_pattern_v2"].astype(str)

    # Convertir en numérique proprement
    Xdf = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    # Remplacer inf/-inf par NaN
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    Xtr, Xte, ytr, yte = train_test_split(
        Xdf, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    labels = sorted(y.unique().tolist())

    rep = classification_report(
        yte, pred,
        labels=labels,
        digits=4,
        zero_division=0
    )
    print(rep)
    REPORT_OUT.write_text(rep, encoding="utf-8")
    print(" Report saved:", REPORT_OUT)

    cm = confusion_matrix(yte, pred, labels=labels)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("ORACLE RF v2 confusion matrix")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()

    # annotation des valeurs (optionnel mais utile)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=200)
    plt.close()
    print(" Confusion matrix saved:", CM_OUT)

    rf = clf.named_steps["rf"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]

    ordered_features = [FEATURES[i] for i in idx]
    ordered_importances = importances[idx]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(ordered_features)), ordered_importances)
    plt.xticks(range(len(ordered_features)), ordered_features, rotation=90)
    plt.title("Feature importance (ORACLE RF v2)")
    plt.tight_layout()
    plt.savefig(IMP_OUT, dpi=200)
    plt.close()
    print(" Feature importance saved:", IMP_OUT)

    joblib.dump(clf, MODEL_OUT)
    print(" Model saved:", MODEL_OUT)

    # Petit diagnostic utile: combien de NaN imputés ?
    n_nan_train = int(Xtr.isna().sum().sum())
    n_nan_test = int(Xte.isna().sum().sum())
    print(f"\n  NaN avant imputation: train={n_nan_train}, test={n_nan_test}")

if __name__ == "__main__":
    main()
