# phase5_infer_anticipation.py

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

DATA_SEQ = OUT_DIR / "WASD_RF_sequences_v5.npz"
MODEL_PATH = OUT_DIR / "transformer_multiclass_v5.keras"
CSV_OUT = OUT_DIR / "WASD_RF_anticipation_v5.csv"


# États RF prédits par le modèle temporel
CLASS_NAMES = ["normal", "pre_attack", "attack"]


def main():
    """
    Applique le modèle temporel de Phase 5 sur les séquences RF.

    L’objectif est de prédire l’état futur du système :
    normal, pré-attaque ou attaque.
    Cette étape matérialise la partie anticipative du pipeline.
    """

    # =========================
    # Chargement des séquences
    # =========================

    data = np.load(DATA_SEQ)

    X = data["X"]
    y_true = data["y"]
    file_ids = data["file_id"]
    window_ids = data["window_id"]

    # =========================
    # Chargement du modèle
    # =========================

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # =========================
    # Inférence temporelle
    # =========================

    probabilities = model.predict(X)
    y_pred = np.argmax(probabilities, axis=1)

    # =========================
    # Évaluation du modèle
    # =========================

    print("\n===== Performance multi-classe =====")

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print(report)

    cm = confusion_matrix(y_true, y_pred)

    print("Matrice de confusion :")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    # =========================
    # Export des prédictions
    # =========================
    # Le CSV produit ici servira ensuite au calcul du délai d’anticipation
    # et à la Phase 6 de décision proactive.

    df_out = pd.DataFrame({
        "file_id": file_ids,
        "window_id": window_ids,
        "true_state": y_true,
        "pred_state": y_pred,
        "proba_normal": probabilities[:, 0],
        "proba_pre_attack": probabilities[:, 1],
        "proba_attack": probabilities[:, 2]
    })

    df_out.to_csv(CSV_OUT, index=False, sep=";", decimal=",")

    print("\nRésultats d’anticipation sauvegardés :", CSV_OUT)


if __name__ == "__main__":
    main()