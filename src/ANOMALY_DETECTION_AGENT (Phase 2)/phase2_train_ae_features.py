# phase2_train_ae_features.py

import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CSV_IN = OUT_DIR / "WASD_RF_dataset.csv"
CSV_OUT = OUT_DIR / "WASD_RF_dataset_scored.csv"

MODEL_PATH = OUT_DIR / "ae_model.pt"
SCALER_PATH = OUT_DIR / "ae_scaler.joblib"
THRESH_PATH = OUT_DIR / "ae_threshold.txt"
NORMAL_ONLY_CSV = OUT_DIR / "normal_only_windows.csv"


# =========================
# Features utilisées
# =========================

# Ce sont les descripteurs physiques extraits en Phase 1.
# Ils servent de représentation du comportement RF.
FEATURES = [
    'time_window_ms', 'center_frequency_hz', 'bandwidth_hz',
    'RSSI_dB', 'SNR_dB', 'puissance_moyenne', 'variance_IQ',
    'kurtosis', 'skewness', 'PSD_mean', 'occupation_spectrale',
    'ISR_dB', 'RSRP_dB', 'RSRQ_dB', 'EVM_percent', 'BER_est'
]


# =========================
# Définition de l’autoencoder
# =========================

class AE(nn.Module):
    """
    Autoencoder simple entièrement connecté.

    L’objectif est d’apprendre une représentation compacte
    du comportement RF nominal, puis de mesurer l’erreur de reconstruction.
    """

    def __init__(self, d: int):
        super().__init__()

        # Compression progressive des features
        self.encoder = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Reconstruction à partir de l’espace latent
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, d)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =========================
# Reproductibilité
# =========================

def set_seeds(seed: int = 42):
    """
    Fixe les graines aléatoires pour garantir la reproductibilité
    des résultats expérimentaux.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Pipeline principal
# =========================

def main():
    set_seeds(42)

    # -------------------------
    # Chargement du dataset Phase 1
    # -------------------------
    if not CSV_IN.exists():
        raise FileNotFoundError(f"CSV introuvable: {CSV_IN}")

    df = pd.read_csv(CSV_IN, sep=";", decimal=",")
    print("Dataset chargé :", df.shape)

    # Vérification de la présence des features attendues
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Vérification des valeurs invalides
    Xcheck = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    if Xcheck.isna().any().any():
        raise ValueError("Présence de NaN/Inf dans les données")

    # -------------------------
    # Sélection du comportement normal
    # -------------------------
    # On entraîne l’autoencoder uniquement sur des données considérées comme saines.
    df_train = df[
        (df["SNR_dB"] >= 10) &
        (df["EVM_percent"] <= 15) &
        (df["ISR_dB"] <= 15)
    ].copy()

    print(f"Fenêtres normales retenues : {len(df_train)} / {len(df)}")

    df_train.to_csv(NORMAL_ONLY_CSV, index=False, sep=";", decimal=",")

    if len(df_train) < 50:
        raise ValueError("Dataset normal-only insuffisant")

    # -------------------------
    # Normalisation
    # -------------------------
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_train[FEATURES].values)

    Xtr, Xva = train_test_split(Xs, test_size=0.2, random_state=42)

    # -------------------------
    # Initialisation du modèle
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(d=Xs.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # -------------------------
    # Boucle d’entraînement
    # -------------------------
    def run_epoch(X, train=True):
        model.train(train)
        losses = []

        for i in range(0, len(X), 256):
            xb = torch.tensor(X[i:i+256], dtype=torch.float32, device=device)

            pred = model(xb)
            loss = loss_fn(pred, xb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    train_losses, val_losses = [], []

    for ep in range(50):
        tr_loss = run_epoch(Xtr, True)
        va_loss = run_epoch(Xva, False)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if ep % 5 == 0:
            print(f"Epoch {ep:02d} | train={tr_loss:.6f} | val={va_loss:.6f}")

    # -------------------------
    # Calcul du seuil d’anomalie
    # -------------------------
    model.eval()

    with torch.no_grad():
        rec = model(torch.tensor(Xs, dtype=torch.float32, device=device)).cpu().numpy()

    err_norm = np.mean((rec - Xs) ** 2, axis=1)

    # Seuil basé sur le percentile 99 du comportement normal
    threshold = np.percentile(err_norm, 99)

    print("Seuil d’anomalie (p99) :", threshold)

    # -------------------------
    # Scoring global
    # -------------------------
    X_all = scaler.transform(df[FEATURES].values)

    with torch.no_grad():
        rec_all = model(torch.tensor(X_all, dtype=torch.float32, device=device)).cpu().numpy()

    score_all = np.mean((rec_all - X_all) ** 2, axis=1)

    # -------------------------
    # Construction du dataset Phase 2
    # -------------------------
    df_out = df.copy()

    df_out["anomaly_score"] = score_all
    df_out["is_anomaly"] = (score_all > threshold).astype(int)
    df_out["threshold_p99"] = threshold

    df_out.to_csv(CSV_OUT, index=False, sep=";", decimal=",")

    # Sauvegarde des éléments nécessaires à l’inférence
    joblib.dump(scaler, SCALER_PATH)
    torch.save(model.state_dict(), MODEL_PATH)
    THRESH_PATH.write_text(str(threshold))

    print("Dataset scoré et modèle sauvegardés")

    # -------------------------
    # Visualisation
    # -------------------------

    # Courbe de loss
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.legend()
    plt.title("Courbe de convergence")
    plt.savefig(OUT_DIR / "ae_loss_curve.png", dpi=200)
    plt.close()

    # Histogramme des scores (échelle log)
    plt.figure(figsize=(8, 5))
    plt.hist(score_all, bins=80, alpha=0.5, label="all data")
    plt.hist(err_norm, bins=80, alpha=0.5, label="normal-only")
    plt.axvline(threshold, color="red", linestyle="--", label="threshold")
    plt.xscale("log")
    plt.legend()
    plt.title("Distribution des anomaly scores (log)")
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / "ae_score_hist_log.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()