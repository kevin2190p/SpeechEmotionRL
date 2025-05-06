# ===============================================================
# Step 3.4: Training and Exploration Module for Emotion RL Agents
# ===============================================================

import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO, A2C, DQN, QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from gym.wrappers import ActionWrapper
from emotion_env import EmotionEnv  

# 3.4: QRDQN Agent Trainer
class QRDQNTrainer:
    def run(self, X_train, y_train, X_val, y_val, total_timesteps=10000):
        train_env = EmotionEnv(features=X_train, labels=y_train)
        val_env = EmotionEnv(features=X_val, labels=y_val)

        eval_callback = EvalCallback(val_env, best_model_save_path="./logs/QRDQN/",
                                     log_path="./logs/QRDQN/", eval_freq=5000,
                                     deterministic=False, render=False)

        base_model = QRDQN("MlpPolicy", train_env, verbose=1, learning_rate=0.0001, buffer_size=200000)
        base_model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        base_model.save("./models/QRDQN_base")

        print("\n--- QRDQN Base Agent Trained ---")

        print("\n--- QRDQN with Epsilon-Greedy ---")
        epsilon_model = QRDQN("MlpPolicy", train_env, verbose=1, learning_rate=0.0001,
                              exploration_fraction=0.2,
                              exploration_final_eps=0.01,
                              exploration_initial_eps=1.0)
        epsilon_model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        epsilon_model.save("./models/QRDQN_epsilon_greedy")

        print("\n--- QRDQN with Random Exploration ---")
        class RandomActionWrapper(ActionWrapper):
            def action(self, action):
                if random.random() < 0.1:
                    return self.env.action_space.sample()
                return action

            def reverse_action(self, action):
                return action

        train_env_random = RandomActionWrapper(train_env)

        random_explore_model = QRDQN("MlpPolicy", train_env_random, verbose=1, learning_rate=0.0003)
        random_explore_model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        random_explore_model.save("./models/QRDQN_random_explore")