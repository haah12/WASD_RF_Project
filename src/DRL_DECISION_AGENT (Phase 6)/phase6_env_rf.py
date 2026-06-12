# phase6_env_rf.py

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class RFEnv(gym.Env):
    """
    Environnement RF utilisé pour entraîner l’agent DRL.

    L’objectif est de simuler une prise de décision proactive à partir
    des scores produits par les phases précédentes : anomalie, gravité,
    risque RF et anticipation temporelle.
    """

    def __init__(self, csv_features, csv_anticipation):
        super(RFEnv, self).__init__()

        # Chargement des résultats issus de la Phase 4 et de la Phase 5
        df_feat = pd.read_csv(csv_features, sep=";", decimal=",")
        df_pred = pd.read_csv(csv_anticipation, sep=";", decimal=",")

        df_feat.columns = df_feat.columns.str.strip()
        df_pred.columns = df_pred.columns.str.strip()

        # Fusion des informations de risque et d’anticipation
        self.df = pd.merge(
            df_feat,
            df_pred,
            on=["file_id", "window_id"],
            how="inner"
        )

        self.df = self.df.sort_values(
            ["file_id", "window_id"]
        ).reset_index(drop=True)

        # Normalisation simple des scores principaux afin de garder
        # un espace d’état borné entre 0 et 1.
        for col in ["Score_SPECTRA", "RiskScore_RF", "anomaly_score"]:
            if col in self.df.columns:
                max_val = self.df[col].max()
                if max_val > 0:
                    self.df[col] = self.df[col] / max_val

        # Probabilité de pré-attaque produite par le modèle temporel
        self.df["anticipation_score"] = self.df.get("proba_pre_attack", 0.0)

        # Variables observées par l’agent à chaque étape
        self.features = [
            "anomaly_score",
            "Score_SPECTRA",
            "RiskScore_RF",
            "anticipation_score"
        ]

        # État : vecteur continu décrivant la situation RF courante
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.features),),
            dtype=np.float32
        )

        # Actions possibles :
        # 0 : ignorer
        # 1 : déclencher une alerte
        # 2 : changer de canal
        # 3 : augmenter la puissance
        self.action_space = spaces.Discrete(4)

        self.current_step = 0
        self.max_steps = len(self.df) - 1

    def reset(self, seed=None, options=None):
        """
        Réinitialise l’environnement.

        Le départ est choisi dans la première moitié du dataset afin
        de laisser suffisamment d’étapes à l’agent pour interagir.
        """
        super().reset(seed=seed)

        self.current_step = np.random.randint(0, len(self.df) // 2)

        return self._get_state(), {}

    def _get_state(self):
        """
        Retourne l’état RF courant sous forme de vecteur numérique.
        """
        return self.df.loc[
            self.current_step,
            self.features
        ].values.astype(np.float32)

    def step(self, action):
        """
        Applique une action et calcule la récompense associée.

        La récompense dépend du niveau de risque, de l’anticipation,
        de la qualité de service et du coût de l’action choisie.
        """

        row = self.df.loc[self.current_step]

        base_risk = float(row["RiskScore_RF"])
        anticipation = float(row["anticipation_score"])

        # Petite composante aléatoire pour simuler l’incertitude naturelle
        # d’un environnement RF réel.
        stochastic = np.random.beta(2, 2)

        # Risque combiné : score mesuré, anticipation et variabilité simulée.
        risk = (
            0.5 * base_risk +
            0.3 * anticipation +
            0.2 * stochastic
        )

        # Perturbation occasionnelle pour rendre l’environnement moins trivial.
        if np.random.rand() < 0.05:
            risk += 0.3

        risk = np.clip(risk, 0, 1)

        throughput = max(0, 1 - risk)
        packet_loss = risk

        # Coût associé à chaque action.
        # Les actions plus fortes sont volontairement plus coûteuses.
        action_costs = [0.05, 0.25, 0.5, 0.4]
        action_cost = action_costs[action]

        reward = 0.0

        # =========================
        # Logique de décision
        # =========================

        if risk < 0.3:
            # Risque faible : ignorer est généralement le bon choix.
            if action == 0:
                reward = 3.0
            elif action == 1:
                reward = 1.0
            elif action == 2:
                reward = -1.0
            elif action == 3:
                reward = -2.0

        elif 0.3 <= risk < 0.6:
            # Risque moyen : une action préventive devient utile.
            if action == 2:
                reward = 4.5
            elif action == 1:
                reward = 3.0
            elif action == 3:
                reward = 2.0
            elif action == 0:
                reward = 0.0

        else:
            # Risque élevé : l’inaction est fortement pénalisée.
            if action == 3:
                reward = 8.0
            elif action == 2:
                reward = 4.0
            elif action == 1:
                reward = 2.5
            elif action == 0:
                reward = -12.0

        # Bonus lorsque l’agent agit avant que l’attaque ne soit confirmée.
        if anticipation > 0.4:
            if action == 1:
                reward += 3.0
            elif action == 2:
                reward += 2.0

        # En situation critique, augmenter la puissance est récompensé.
        if risk > 0.7 and action == 3:
            reward += 5.0

        # À l’inverse, ignorer un risque critique est fortement pénalisé.
        if risk > 0.7 and action == 0:
            reward -= 8.0

        # Ajustement lié à la qualité de service.
        reward += 1.0 * throughput
        reward -= 0.5 * packet_loss
        reward -= action_cost

        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        next_state = (
            self._get_state()
            if not terminated
            else np.zeros(len(self.features), dtype=np.float32)
        )

        return next_state, reward, terminated, truncated, {
            "risk": risk,
            "throughput": throughput,
            "anticipation": anticipation
        }