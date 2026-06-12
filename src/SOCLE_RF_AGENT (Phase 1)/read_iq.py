# read_iq.py

import numpy as np
import os


def read_iq_file(bin_path: str, dataform: int = 2, remove_dc: bool = True) -> np.ndarray:
    """
    Lit un fichier binaire contenant des échantillons I/Q.

    Le format attendu est une alternance I, Q, I, Q, ...
    avec un type dépendant du paramètre dataform.
    """

    # =========================
    # Détection du type de données
    # =========================
    # dataform indique la taille des entiers utilisés dans le fichier.
    # Par défaut, on utilise du int16 (cas le plus courant en RF).

    dtype_map = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }

    dtype = dtype_map.get(int(dataform), np.int16)

    # =========================
    # Vérification du fichier
    # =========================

    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Fichier BIN introuvable : {bin_path}")

    raw = np.fromfile(bin_path, dtype=dtype)

    if raw.size == 0:
        raise ValueError(f"Fichier BIN vide : {bin_path}")

    # Un signal I/Q doit contenir un nombre pair d’échantillons
    # (un I et un Q pour chaque instant).
    if raw.size % 2 != 0:
        raise ValueError(f"Taille invalide (I/Q incomplet) : {raw.size}")

    # =========================
    # Séparation I et Q
    # =========================

    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)

    # =========================
    # Normalisation
    # =========================
    # Conversion vers [-1, 1] en fonction de la résolution du type.

    full_scale = float(2 ** (np.iinfo(dtype).bits - 1))

    iq = (i / full_scale) + 1j * (q / full_scale)

    # =========================
    # Suppression du DC offset
    # =========================
    # Le recentrage du signal permet d’éviter des biais dans les calculs
    # de features (PSD, CFO, etc.).

    if remove_dc:
        iq = iq - np.mean(iq)

    return iq