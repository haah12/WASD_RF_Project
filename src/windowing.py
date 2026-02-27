import numpy as np

def get_windows(iq: np.ndarray, n_windows: int = 4) -> list:
    n = len(iq)
    if n < n_windows:
        return []
    chunk = n // n_windows
    return [iq[k*chunk:(k+1)*chunk] for k in range(n_windows)]

# Alias compatibilité (si d'autres scripts utilisent get_4_windows)
def get_4_windows(iq: np.ndarray) -> list:
    return get_windows(iq, n_windows=4)
