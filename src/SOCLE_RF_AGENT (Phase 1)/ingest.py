# ingest.py

import json
import pandas as pd
from pathlib import Path


def ingest_metadata(json_dir: str, bin_dir: str) -> pd.DataFrame:
    """
    Lit les fichiers JSON de métadonnées et les associe aux fichiers BIN correspondants.

    Cette étape prépare l'accès aux signaux IQ bruts avant l'extraction
    des features RF. Elle permet surtout de vérifier que chaque fichier JSON
    possède bien un fichier BIN exploitable.
    """

    json_path = Path(json_dir)
    bin_path = Path(bin_dir)

    # Vérification des dossiers d'entrée
    if not json_path.exists():
        raise FileNotFoundError(f"Dossier JSON introuvable: {json_dir}")

    if not bin_path.exists():
        raise FileNotFoundError(f"Dossier BIN introuvable: {bin_dir}")

    rows = []

    # Parcours ordonné des fichiers de métadonnées
    for json_file in sorted(json_path.glob("*.json")):
        try:
            meta = json.loads(json_file.read_text(encoding="utf-8"))

            # Le nom du fichier BIN est récupéré depuis le JSON.
            # Si le champ est absent, on suppose qu'il porte le même nom que le JSON.
            iq_file_path = str(meta.get("iq_file_path", ""))
            bin_filename = Path(iq_file_path).name if iq_file_path else f"{json_file.stem}.bin"
            bin_fullpath = bin_path / bin_filename

            rows.append({
                "json_file": json_file.name,
                "datetime": meta.get("datetime", ""),
                "nanosec": int(meta.get("nanosec", 0)),

                # Paramètres radio utiles pour l'analyse RF
                "center_freq_hz": float(meta.get("frequency", 0.0)),
                "bandwidth_hz": float(meta.get("bandwidth", 0.0)),
                "sample_rate_hz": float(meta.get("sampling_frequency", 0.0)),
                "num_samples": int(meta.get("iq_sample_number", 0)),
                "iq_dataform": int(meta.get("iq_dataform", 2)),

                # Association JSON ↔ BIN
                "bin_file": bin_filename,
                "bin_path": str(bin_fullpath),
                "bin_exists": bin_fullpath.exists(),
            })

        except Exception as e:
            print(f"Fichier ignoré {json_file.name}: {e}")

    df = pd.DataFrame(rows)

    print(
        f"Métadonnées ingérées : {len(df)} fichiers JSON | "
        f"BIN valides : {df['bin_exists'].sum()}/{len(df)}"
    )

    return df