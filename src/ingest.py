# ingest.py
import json
import pandas as pd
from pathlib import Path

def ingest_metadata(json_dir: str, bin_dir: str) -> pd.DataFrame:
    json_path = Path(json_dir)
    bin_path = Path(bin_dir)

    if not json_path.exists():
        raise FileNotFoundError(f" json_dir introuvable: {json_dir}")
    if not bin_path.exists():
        raise FileNotFoundError(f" bin_dir introuvable: {bin_dir}")

    rows = []
    for jf in sorted(json_path.glob("*.json")):
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))

            iq_file_path = str(meta.get("iq_file_path", ""))
            bin_filename = Path(iq_file_path).name if iq_file_path else f"{jf.stem}.bin"
            bin_fullpath = bin_path / bin_filename

            rows.append({
                "json_file": jf.name,
                "datetime": meta.get("datetime", ""),
                "nanosec": int(meta.get("nanosec", 0)),
                "center_freq_hz": float(meta.get("frequency", 0.0)),
                "bandwidth_hz": float(meta.get("bandwidth", 0.0)),
                "sample_rate_hz": float(meta.get("sampling_frequency", 0.0)),
                "num_samples": int(meta.get("iq_sample_number", 0)),
                "iq_dataform": int(meta.get("iq_dataform", 2)),
                "bin_file": bin_filename,
                "bin_path": str(bin_fullpath),
                "bin_exists": bin_fullpath.exists(),
            })
        except Exception as e:
            print(f" Skip {jf.name}: {e}")

    df = pd.DataFrame(rows)
    print(f" Metadata ingéré: {len(df)} JSON | BIN OK: {df['bin_exists'].sum()}/{len(df)}")
    return df
