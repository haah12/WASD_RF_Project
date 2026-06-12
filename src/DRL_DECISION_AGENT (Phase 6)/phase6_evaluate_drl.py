# phase6_evaluate_drl.py

from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from phase6_env_rf import RFEnv


# =========================
# Chemins du projet
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FEATURES = PROJECT_ROOT / "outputs" / "WASD_RF_risk_scored.csv"
CSV_ANTICIPATION = PROJECT_ROOT / "outputs" / "WASD_RF_anticipation_v5.csv"

MODEL_PATH = PROJECT_ROOT / "outputs" / "drl_dqn_model_final"


# Actions disponibles pour l’agent DRL
ACTIONS = ["IGNORE", "ALERT", "CHANGE_CHANNEL", "BOOST_POWER"]


def make_env():
    """
    Crée une instance de l’environnement RF utilisé pour l’évaluation.
    """
    return RFEnv(CSV_FEATURES, CSV_ANTICIPATION)


def main():
    """
    Évalue la politique apprise par l’agent DQN.

    L’objectif est d’observer les actions choisies par l’agent
    face aux différents niveaux de risque RF.
    """

    # =========================
    # Environnement et normalisation
    # =========================
    # On recharge la même normalisation que celle utilisée pendant
    # l'entraînement afin de garder des observations cohérentes.

    env = DummyVecEnv([make_env])

    env = VecNormalize.load(
        str(MODEL_PATH) + "_vecnormalize.pkl",
        env
    )

    env.training = False
    env.norm_reward = False

    # =========================
    # Chargement du modèle DQN
    # =========================

    model = DQN.load(MODEL_PATH)

    state = env.reset()

    total_reward = 0.0
    action_counts = {action_name: 0 for action_name in ACTIONS}
    risks = []

    print("\n===== Politique de décision DRL =====")

    # =========================
    # Boucle d’évaluation
    # =========================

    for step in range(200):

        # L’évaluation est déterministe pour observer la politique apprise,
        # sans exploration aléatoire.
        action, _ = model.predict(state, deterministic=True)
        action = int(action[0])

        # VecEnv attend une liste d’actions, même avec un seul environnement.
        next_state, reward, done, info = env.step([action])

        reward = float(reward[0])
        done = bool(done[0])
        info = info[0]

        risk = round(info["risk"], 3)
        risks.append(risk)

        action_name = ACTIONS[action]
        action_counts[action_name] += 1

        print(
            f"Step {step} | "
            f"Risk: {risk} | "
            f"Action: {action_name} | "
            f"Reward: {round(reward, 3)}"
        )

        total_reward += reward
        state = next_state

        if done:
            break

    # =========================
    # Résumé de l’évaluation
    # =========================

    print("\n===== Résumé =====")
    print("Reward total :", round(total_reward, 3))

    if risks:
        print("Risque moyen :", round(sum(risks) / len(risks), 3))
    else:
        print("Risque moyen : N/A")

    print("\nDistribution des actions :")
    for action_name, count in action_counts.items():
        print(f"{action_name}: {count}")


if __name__ == "__main__":
    main()