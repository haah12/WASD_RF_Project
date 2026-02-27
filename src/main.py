# main.py
from pathlib import Path
from ingest import ingest_metadata
from features import extract_all_features
from build_csv import build_final_csv

def main(max_files: int = 505, n_windows: int = 4):
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data" / "36_LTE_1"

    json_dir = data_root / "json"   # <-- adapte ici si besoin
    bin_dir  = data_root / "bin"    # <-- adapte ici si besoin

    print("=" * 70)
    print(" PIPELINE WASD RF (GitHub IQ) → FEATURES → CSV")
    print("=" * 70)

    # 1) Ingest
    metadata_df = ingest_metadata(json_dir=str(json_dir), bin_dir=str(bin_dir))

    # 2) Features
    features_df = extract_all_features(max_files=max_files, num_windows=n_windows)


    # 3) CSV
    build_final_csv()

    print(" Terminé.")

if __name__ == "__main__":
    main(max_files=505, n_windows=4)
