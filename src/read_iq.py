# read_iq.py
import numpy as np
import os

def read_iq_file(bin_path: str, dataform: int = 2, remove_dc: bool = True) -> np.ndarray:
    """
    Lecture IQ interleavé I,Q :
    - intX -> float32 en [-1, 1)
    - suppression DC optionnelle
    """
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(int(dataform), np.int16)

    if not os.path.exists(bin_path):
        raise FileNotFoundError(f" BIN introuvable: {bin_path}")

    raw = np.fromfile(bin_path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f" BIN vide: {bin_path}")
    if raw.size % 2 != 0:
        raise ValueError(f" Taille impaire (I/Q): {raw.size}")

    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)

    full_scale = float(2 ** (np.iinfo(dtype).bits - 1))  # 32768 pour int16
    iq = (i / full_scale) + 1j * (q / full_scale)

    if remove_dc:
        iq = iq - np.mean(iq)

    return iq
