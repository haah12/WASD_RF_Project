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

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CSV_IN = OUT_DIR / "WASD_RF_dataset.csv"
CSV_OUT = OUT_DIR / "WASD_RF_dataset_scored.csv"

MODEL_PATH = OUT_DIR / "ae_model.pt"
SCALER_PATH = OUT_DIR / "ae_scaler.joblib"
THRESH_PATH = OUT_DIR / "ae_threshold.txt"
NORMAL_ONLY_CSV = OUT_DIR / "normal_only_windows.csv"

# ----------------------------
# Features (from phase2_select_features.py)
# ----------------------------
FEATURES = [
    'time_window_ms', 'center_frequency_hz', 'bandwidth_hz',
    'RSSI_dB', 'SNR_dB', 'puissance_moyenne', 'variance_IQ',
    'kurtosis', 'skewness', 'PSD_mean', 'occupation_spectrale',
    'ISR_dB', 'RSRP_dB', 'RSRQ_dB', 'EVM_percent', 'BER_est'
]

# ----------------------------
# Autoencoder model
# ----------------------------
class AE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, d)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 0) Reproducibility
    set_seeds(42)

    # 1) Lire CSV (Excel-FR export)
    if not CSV_IN.exists():
        raise FileNotFoundError(f" CSV introuvable: {CSV_IN}")
    df = pd.read_csv(CSV_IN, sep=";", decimal=",")
    print(" CSV loaded:", df.shape)

    # 2) Vérifier features présentes
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f" Colonnes manquantes dans CSV: {missing}")

    # 2bis) Vérifier NaN/Inf (sécurité)
    Xcheck = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    if Xcheck.isna().any().any():
        bad_cols = Xcheck.columns[Xcheck.isna().any()].tolist()
        raise ValueError(f" NaN/Inf détectés dans: {bad_cols}")

    # 3) Définir normal-only (qualité RF)
    #    (Seuils défendables scientifiquement)
    df_train = df[
        (df["SNR_dB"] >= 10) &
        (df["EVM_percent"] <= 15) &
        (df["ISR_dB"] <= 15)
    ].copy()

    print(f" Normal-only windows: {len(df_train)} / {len(df)}")

    # 3bis) Sauvegarde normal-only (traçabilité thèse)
    df_train.to_csv(NORMAL_ONLY_CSV, index=False, sep=";", decimal=",")
    print("Saved normal-only subset:", NORMAL_ONLY_CSV)

    # Stop si trop petit
    if len(df_train) < 50:
        raise ValueError(
            " Normal-only trop petit (<50). "
            "Assouplis les seuils (ex: SNR>=8, EVM<=18, ISR<=18)."
        )

    if len(df_train) < 200:
        print(" Peu d'exemples normal-only (<200). "
              "Tu peux assouplir légèrement les seuils si besoin.")

    # 4) Standardisation (fit sur normal-only uniquement)
    X_train = df_train[FEATURES].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    # 5) Split train/val
    Xtr, Xva = train_test_split(Xs, test_size=0.2, random_state=42)

    # 6) Entraînement AE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(d=Xs.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def epoch_pass(Xnp, train: bool):
        model.train(train)
        bs = 256
        losses = []
        for i in range(0, len(Xnp), bs):
            xb = torch.tensor(Xnp[i:i+bs], dtype=torch.float32, device=device)
            pred = model(xb)
            loss = loss_fn(pred, xb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    train_losses, val_losses = [], []
    for ep in range(50):
        tr = epoch_pass(Xtr, train=True)
        va = epoch_pass(Xva, train=False)
        train_losses.append(tr)
        val_losses.append(va)
        if ep % 5 == 0:
            print(f"epoch {ep:02d} | train={tr:.6f} | val={va:.6f}")

    # 7) Seuil p99 sur erreurs normal-only
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
        rec = model(Xt).cpu().numpy()
    err_norm = np.mean((rec - Xs) ** 2, axis=1)
    threshold = float(np.percentile(err_norm, 99))
    print(" Threshold p99 (normal-only):", threshold)

    # 8) Score sur tout le dataset (AnomalyScore)
    X_all = scaler.transform(df[FEATURES].astype(float).values)
    with torch.no_grad():
        Xall_t = torch.tensor(X_all, dtype=torch.float32, device=device)
        rec_all = model(Xall_t).cpu().numpy()
    score_all = np.mean((rec_all - X_all) ** 2, axis=1)

    # Score normal-only pour histogramme comparatif
    X_norm = scaler.transform(df_train[FEATURES].astype(float).values)
    with torch.no_grad():
        Xn_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
        rec_n = model(Xn_t).cpu().numpy()
    score_norm = np.mean((rec_n - X_norm) ** 2, axis=1)

    # 9) Dataset scoré + timeline
    df_out = df.copy()
    df_out["anomaly_score"] = score_all
    df_out["is_anomaly"] = (df_out["anomaly_score"] > threshold).astype(int)

    # Pour audit / traçabilité
    df_out["threshold_p99"] = threshold

    # Timeline relative (AnomalyScore(t))
    # t_ms = window_id * time_window_ms
    if "window_id" in df_out.columns and "time_window_ms" in df_out.columns:
        df_out["t_ms"] = df_out["window_id"].astype(float) * df_out["time_window_ms"].astype(float)
    else:
        df_out["t_ms"] = np.nan

    # 10) Sauvegarder modèle/scaler/seuil + CSV scoré
    joblib.dump(scaler, SCALER_PATH)
    torch.save(model.state_dict(), MODEL_PATH)
    THRESH_PATH.write_text(str(threshold), encoding="utf-8")
    df_out.to_csv(CSV_OUT, index=False, sep=";", decimal=",")
    print(" Saved scored dataset:", CSV_OUT)
    print(" Saved scaler:", SCALER_PATH)
    print(" Saved model:", MODEL_PATH)
    print(" Saved threshold:", THRESH_PATH)

    # 11) Figures (thèse-ready)
    # 11.1 Courbe loss train/val
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Autoencoder training/validation loss")
    plt.legend()
    plt.savefig(OUT_DIR / "ae_loss_curve.png", dpi=200)
    plt.close()

    # 11.2 Histogramme scores : all vs normal-only + seuil
    plt.figure()
    plt.hist(score_all, bins=80, alpha=0.5, label="all")
    plt.hist(score_norm, bins=80, alpha=0.5, label="normal-only")
    plt.axvline(threshold, linestyle="--", label="threshold p99")
    plt.xlabel("anomaly_score (MSE)")
    plt.ylabel("count")
    plt.title("Anomaly score distribution (all vs normal-only)")
    plt.legend()
    plt.savefig(OUT_DIR / "ae_score_hist.png", dpi=200)
    plt.close()

    print(" Figures saved: ae_loss_curve.png, ae_score_hist.png")

    # Petit résumé
    n_anom = int(df_out["is_anomaly"].sum())
    print(f"\n Résumé: anomalies détectées = {n_anom} / {len(df_out)} fenêtres "
          f"({100*n_anom/len(df_out):.2f}%)")

if __name__ == "__main__":
    main()
