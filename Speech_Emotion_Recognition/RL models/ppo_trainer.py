# =============================================================
# Step 3.1: Training and Exploration Module for Emotion RL Agents
# =============================================================

import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO, A2C, DQN, QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from gym.wrappers import ActionWrapper
from emotion_env import EmotionEnv  

# 3.1: PPO Agent Trainer
class PPOTrainer:
    def run(self, X_train, y_train, X_val, y_val, total_timesteps=10000):
        train_env = EmotionEnv(features=X_train, labels=y_train)
        val_env = EmotionEnv(features=X_val, labels=y_val)

        eval_callback = EvalCallback(val_env, best_model_save_path="./logs/PPO/",
                                     log_path="./logs/PPO/", eval_freq=5000,
                                     deterministic=False, render=False)

        base_model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003)
        base_model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        base_model.save("./models/PPO_base")

        print("\n--- PPO Base Agent Trained ---")

        print("\n--- PPO with Entropy Exploration ---")
        entropy_model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003, ent_coef=0.05)
        entropy_model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        entropy_model.save("./models/PPO_entropy")
