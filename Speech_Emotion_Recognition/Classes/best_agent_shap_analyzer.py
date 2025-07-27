# =====================================================
# Step 7: Save the Best Agent and Perform SHAP Analysis
# =====================================================

import os
import json
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib import QRDQN

class BestAgentSHAPAnalyzer:
    def __init__(self, best_tuned_models, test_results, test_env, best_models, X_test):
        """
        Initialize with required variables.
        """
        self.best_tuned_models = best_tuned_models
        self.test_results = test_results
        self.test_env = test_env
        self.best_models = best_models
        self.X_test = X_test

    # ------------------------------------------------------
    # Step 8: Save the Best Agent and Corresponding Hyperparameters
    # ------------------------------------------------------
    def save_best_agent_and_params(self):
        print("\n--- Step 8: Save the Best Agent and Corresponding Hyperparameters ---")

        best_agent_name = None
        best_mean_reward = -np.inf
        best_model_path = None
        best_params = None

        for agent_name, rewards in self.test_results.items():
            mean_reward = np.mean(rewards)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_agent_name = agent_name
                best_model_path, best_params = self.best_tuned_models.get(agent_name, (None, None))

        if best_agent_name and best_model_path:
            best_model = None
            if "qrdqn" in best_agent_name.lower():
                best_model = QRDQN.load(best_model_path)
            elif "dqn" in best_agent_name.lower():
                best_model = DQN.load(best_model_path)
            elif "ppo" in best_agent_name.lower():
                best_model = PPO.load(best_model_path)
            elif "a2c" in best_agent_name.lower():
                best_model = A2C.load(best_model_path)
            else:
                print(f"Warning: Unknown algorithm '{best_agent_name}'. Cannot load.")

            if best_model:
                os.makedirs("saved_models", exist_ok=True)
                save_path = os.path.join("saved_models", f"best_emotion_agent_{best_agent_name}.zip")
                best_model.save(save_path)
                print(f"Best {best_agent_name} model saved to {save_path}")

                if best_params:
                    with open("saved_models/best_hyperparams.json", "w") as f:
                        json.dump({best_agent_name: best_params}, f, indent=4)
                    print(f"Best hyperparameters for {best_agent_name} saved to saved_models/best_hyperparams.json")
                else:
                    print(f"Warning: Best hyperparameters not found for {best_agent_name}. Skipping saving.")
            else:
                print(f"Error loading the best model for {best_agent_name} from {best_model_path}.")
        else:
            print("No best agent found with valid model path.")

    # ------------------------------------------------------
    # Step 9: Analyze Significant Patterns Using SHAP
    # ------------------------------------------------------
    def collect_data(self, agent, env, n_samples=500):
        states, actions = [], []
        obs, _ = env.reset()
        for _ in range(n_samples):
            action, _ = agent.predict(obs)
            states.append(obs)
            actions.append(action)
            obs, reward, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
        return np.array(states), np.array(actions)

    def shap_analysis(self, agent, agent_name, env, feature_names=None):
        print(f"\nRunning SHAP analysis for the best {agent_name} model...")
        states, actions = self.collect_data(agent, env, 500)

        rf = RandomForestRegressor()
        rf.fit(states, actions.flatten())

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(states)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(states.shape[1])]
        shap.summary_plot(shap_values, states, feature_names=feature_names, show=False)
        plt.title(f"SHAP Feature Impact - Best {agent_name} Model")
        plt.show()

    def run_shap_analysis_for_best_models(self):
        if 'PPO' in self.best_models and self.best_models['PPO'] is not None:
            ppo_model = PPO.load(self.best_models['PPO'], env=self.test_env)
            self.shap_analysis(ppo_model, "PPO", self.test_env, feature_names=[f"feature_{i}" for i in range(self.X_test.shape[1])])
        if 'A2C' in self.best_models and self.best_models['A2C'] is not None:
            a2c_model = A2C.load(self.best_models['A2C'], env=self.test_env)
            self.shap_analysis(a2c_model, "A2C", self.test_env, feature_names=[f"feature_{i}" for i in range(self.X_test.shape[1])])
        if 'DQN' in self.best_models and self.best_models['DQN'] is not None:
            dqn_model = DQN.load(self.best_models['DQN'], env=self.test_env)
            self.shap_analysis(dqn_model, "DQN", self.test_env, feature_names=[f"feature_{i}" for i in range(self.X_test.shape[1])])
        if 'QRDQN' in self.best_models and self.best_models['QRDQN'] is not None:
            qrdqn_model = QRDQN.load(self.best_models['QRDQN'], env=self.test_env)
            self.shap_analysis(qrdqn_model, "QRDQN", self.test_env, feature_names=[f"feature_{i}" for i in range(self.X_test.shape[1])])
