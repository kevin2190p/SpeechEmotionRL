# ================================================
# Step 6: Evaluate the Best Models on the Test Set
# ================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import QRDQN

from model_validation import best_tuned_models
from system_hyperparameter_comparison import X_test, y_test

class TestSetEvaluator:
    """Class to evaluate best trained models on the test set."""

    def run_test_evaluation(self):
        print("\n--- Step 6: Evaluating the Best Models on the Test Set ---")

        test_env = EmotionEnv(features=X_test, labels=y_test)

        test_results = {}

        for agent_name, (model_path, best_params) in best_tuned_models.items():
            if os.path.exists(model_path):
                print(f"\nEvaluating the best {agent_name} model (from {model_path}) with params: {best_params} on the test set...")
                all_episode_rewards = []
                successful_episodes = 0
                predicted_emotions = []
                target_emotions = []
                episode_lengths = []

                n_test_episodes = 20

                if agent_name == "PPO":
                    best_model = PPO.load(model_path, env=test_env)
                elif agent_name == "A2C":
                    best_model = A2C.load(model_path, env=test_env)
                elif agent_name == "DQN":
                    best_model = DQN.load(model_path, env=test_env)
                elif agent_name == "QRDQN":
                    best_model = QRDQN.load(model_path, env=test_env)
                else:
                    print(f"Warning: Unknown algorithm '{agent_name}'. Skipping evaluation.")
                    continue

                for episode in range(n_test_episodes):
                    obs, _ = test_env.reset(seed=episode)
                    done = False
                    episode_reward = 0
                    steps_in_episode = 0

                    while not done:
                        obs_input = obs[np.newaxis, :]
                        action, _ = best_model.predict(obs_input, deterministic=False)
                        if isinstance(action, np.ndarray):
                            action = action.item()
                        obs, reward, done, _, _ = test_env.step(action)
                        episode_reward += reward
                        steps_in_episode += 1

                        target_emotion = test_env.get_target_emotion(obs)
                        predicted_emotion = test_env.action_to_emotion(action)

                        if target_emotion is not None and predicted_emotion is not None:
                            target_emotions.append(target_emotion)
                            predicted_emotions.append(predicted_emotion)

                    all_episode_rewards.append(episode_reward)
                    if test_env.is_successful_episode(episode_reward):
                        successful_episodes += 1
                    episode_lengths.append(steps_in_episode)

                test_results[agent_name] = all_episode_rewards
                mean_test_reward = np.mean(all_episode_rewards)
                std_test_reward = np.std(all_episode_rewards)
                success_rate = (successful_episodes / n_test_episodes) * 100 if n_test_episodes > 0 else 0

                if target_emotions:
                    accuracy = accuracy_score(target_emotions, predicted_emotions)
                    f1 = f1_score(target_emotions, predicted_emotions, average='weighted', zero_division=0)
                    precision = precision_score(target_emotions, predicted_emotions, average='weighted', zero_division=0)
                    recall = recall_score(target_emotions, predicted_emotions, average='weighted', zero_division=0)
                    conf_matrix = confusion_matrix(target_emotions, predicted_emotions)
                else:
                    accuracy = 0
                    f1 = 0
                    precision = 0
                    recall = 0
                    conf_matrix = np.array([])

                mean_episode_length = np.mean(episode_lengths)_
