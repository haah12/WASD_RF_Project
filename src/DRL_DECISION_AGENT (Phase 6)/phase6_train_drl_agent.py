# phase6_train_drl_agent.py

from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from phase6_env_rf import RFEnv


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FEATURES = PROJECT_ROOT / "outputs" / "WASD_RF_risk_scored.csv"
CSV_ANTICIPATION = PROJECT_ROOT / "outputs" / "WASD_RF_anticipation_v5.csv"

MODEL_OUT = PROJECT_ROOT / "outputs" / "drl_dqn_model_final"
LOG_DIR = PROJECT_ROOT / "outputs" / "drl_logs"


def make_env():
    """
    Crée l’environnement RF sous une forme compatible avec Stable-Baselines3.

    Le Monitor permet de suivre les récompenses et la durée des épisodes
    pendant l’apprentissage.
    """

    def _init():
        env = RFEnv(CSV_FEATURES, CSV_ANTICIPATION)
        env = Monitor(env)
        return env

    return _init


def main():
    """
    Entraîne un agent DQN pour la prise de décision proactive.

    L’agent apprend à choisir une action adaptée selon le niveau de risque RF,
    le score SPECTRA, l’anomalie détectée et la probabilité de pré-attaque.
    """

    # =========================
    # Initialisation de l’environnement
    # =========================

    env = DummyVecEnv([make_env()])

    # Normalisation des observations et des récompenses pour stabiliser l’apprentissage
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True
    )

    # Vérification rapide de la conformité Gymnasium
    check_env(
        RFEnv(CSV_FEATURES, CSV_ANTICIPATION),
        warn=True
    )

    # =========================
    # Définition de l’agent DQN
    # =========================
    # Les paramètres sont choisis pour encourager une exploration suffisante
    # au début, puis une convergence progressive vers une politique stable.

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,

        learning_rate=5e-4,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.995,

        exploration_fraction=0.6,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,

        target_update_interval=1000,
        train_freq=4,
        gradient_steps=2,

        tensorboard_log=str(LOG_DIR),
    )

    # =========================
    # Apprentissage
    # =========================

    model.learn(total_timesteps=200000)

    # =========================
    # Sauvegarde
    # =========================
    # On sauvegarde à la fois le modèle et les paramètres de normalisation,
    # car ils seront nécessaires pour une évaluation cohérente.

    model.save(MODEL_OUT)
    env.save(str(MODEL_OUT) + "_vecnormalize.pkl")

    print("\nModèle DQN sauvegardé :", MODEL_OUT)


if __name__ == "__main__":
    main()