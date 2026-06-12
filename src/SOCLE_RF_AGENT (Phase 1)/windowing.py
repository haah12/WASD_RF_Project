# windowing.py

import numpy as np


def get_windows(iq: np.ndarray, n_windows: int = 4) -> list:
    """
    Découpe un signal IQ en plusieurs fenêtres de taille égale.

    Cette étape est utilisée pour analyser localement le signal
    plutôt que sur toute sa durée, ce qui permet de capter
    des variations temporelles fines.
    """

    n = len(iq)

    # Si le signal est trop court, on ne peut pas créer les fenêtres demandées
    if n < n_windows:
        return []

    # Taille de chaque segment
    chunk_size = n // n_windows

    windows = []

    for k in range(n_windows):
        start = k * chunk_size
        end = (k + 1) * chunk_size

        windows.append(iq[start:end])

    return windows


def get_4_windows(iq: np.ndarray) -> list:
    """
    Cas standard utilisé dans le projet :
    découpage du signal en 4 fenêtres.
    """
    return get_windows(iq, n_windows=4)