# main.py

from pathlib import Path
from ingest import ingest_metadata
from features import extract_all_features
from build_csv import build_final_csv


def main(max_files: int = 505, n_windows: int = 4):

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data" / "36_LTE_1"

    json_dir = data_root / "json"
    bin_dir  = data_root / "bin"

    print("=" * 70)
    print("PIPELINE WASD-RF : IQ brut -> features RF -> dataset CSV")
    print("=" * 70)

    # =========================
    # 1) INGEST
    # =========================
    metadata_df = ingest_metadata(
        json_dir=str(json_dir),
        bin_dir=str(bin_dir)
    )

    # 🔥 AJOUT IMPORTANT ICI
    metadata_path = project_root / "data" / "metadata.pkl"
    metadata_df.to_pickle(metadata_path)
    print(f"Metadata sauvegardée : {metadata_path}")

    # =========================
    # 2) FEATURES
    # =========================
    features_df = extract_all_features(
        max_files=max_files,
        num_windows=n_windows
    )

    # =========================
    # 3) CSV FINAL
    # =========================
    build_final_csv()

    print("Pipeline terminé.")


if __name__ == "__main__":
    main(max_files=505, n_windows=4)