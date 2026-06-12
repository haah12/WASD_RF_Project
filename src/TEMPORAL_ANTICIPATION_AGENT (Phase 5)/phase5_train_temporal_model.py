# phase5_train_temporal_model.py

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"

DATA = OUT_DIR / "WASD_RF_sequences_v5.npz"
MODEL_OUT = OUT_DIR / "transformer_multiclass_v5.keras"


def transformer_block(x):
    """
    Bloc Transformer léger pour l’analyse temporelle des séquences RF.

    L’attention permet au modèle de repérer les fenêtres importantes
    dans une séquence, même lorsque le signal précurseur est faible.
    """

    # Mécanisme d’attention sur la séquence
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=2,
        key_dim=32
    )(x, x)

    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization()(x)

    # Petit réseau feed-forward appliqué après l’attention
    ff = tf.keras.layers.Dense(64, activation="relu")(x)
    ff = tf.keras.layers.Dense(x.shape[-1])(ff)

    x = tf.keras.layers.Add()([x, ff])
    x = tf.keras.layers.LayerNormalization()(x)

    return x


def main():
    """
    Entraîne le modèle temporel de Phase 5.

    Le modèle apprend à partir de séquences RF successives afin de prédire
    l’état futur du système : normal, pré-attaque ou attaque.
    """

    # =========================
    # Chargement des séquences
    # =========================

    data = np.load(DATA)

    X = data["X"]
    y = data["y"]

    # =========================
    # Découpage apprentissage / validation
    # =========================
    # Le split stratifié conserve la répartition des états RF dans les deux jeux.

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =========================
    # Architecture du modèle
    # =========================

    inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2]))

    # Deux blocs Transformer pour capter les dépendances temporelles
    x = transformer_block(inputs)
    x = transformer_block(x)

    # La couche GRU résume ensuite la dynamique globale de la séquence
    x = tf.keras.layers.GRU(64)(x)

    # Couche dense finale avant la classification
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # Trois classes : normal, pre_attack, attack
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # =========================
    # Compilation
    # =========================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # =========================
    # Entraînement
    # =========================

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # =========================
    # Sauvegarde
    # =========================

    model.save(MODEL_OUT)

    print("Modèle temporel Phase 5 sauvegardé :", MODEL_OUT)


if __name__ == "__main__":
    main()