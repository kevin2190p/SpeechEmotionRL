# ========================
# Step 5: Model Validation
# ========================

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3 import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from EmotionEnv import EmotionEnv  

class ModelValidation:
    """
    Class to perform validation of reinforcement learning models across multiple seeds and visualize results.
    """

    def __init__(self, X_val, y_val, best_tuned_models, seeds=[0, 42, 100], n_episodes=10):
        """
        Initialize the ModelValidation class with the necessary data and models.

        Parameters:
        X_val (numpy array): Validation feature set
        y_val (numpy array): Validation labels
        best_tuned_models (dict): Dictionary of best-tuned models with paths and parameters
        seeds (list): List of seeds to validate the models across
        n_episodes (int): Number of episodes to run for each seed during validation
        """
        self.X_val = X_val
        self.y_val = y_val
        self.best_tuned_models = best_tuned_models
        self.seeds = seeds
        self.n_episodes = n_episodes

        # Create the validation environment
        self.val_env = EmotionEnv(X_val, y_val)  # Validation environment using validation data

    def evaluate_model(self, agent, env, n_episodes=10, seed=0):
        """
        Evaluate a given agent over multiple episodes with a specific seed.

        Parameters:
        agent (RL model): The reinforcement learning agent to evaluate
        env (Environment): The environment to evaluate the agent on
        n_episodes (int): Number of episodes to run for validation
        seed (int): Random seed for reproducibility

        Returns:
        (float, float): Mean reward and standard deviation of rewards across episodes
        """
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset(seed=seed)
            done = False
            total_reward = 0
            while not done:
                obs_input = obs[np.newaxis, :]  # Add a batch dimension for the agent's input
                action, _ = agent.predict(obs_input, deterministic=False) # Use deterministic for evaluation
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        return np.mean(rewards), np.std(rewards)

    def validate_models(self):
        """
        Validate each best agent across all seeds using the validation environment.
        This method prints results to the console and stores them in `final_validation_results`.
        """
        final_validation_results = {}

        print("\n--- Step 5: Validating Best Models Across Multiple Seeds ---")

        # Validate each best agent across all seeds
        for name, (model_path, params) in self.best_tuned_models.items():
            print(f"\nValidating best {name} agent (from {model_path}) across seeds...")
            seed_results = {}
            if os.path.exists(model_path):
                # Load the model based on its type
                if name == "PPO":
                    best_agent = PPO.load(model_path, env=self.val_env)
                elif name == "A2C":
                    best_agent = A2C.load(model_path, env=self.val_env)
                elif name == "DQN":
                    best_agent = DQN.load(model_path, env=self.val_env)
                elif name == "QRDQN":
                    best_agent = QRDQN.load(model_path, env=self.val_env)
                else:
                    print(f"Warning: Unknown algorithm '{name}'. Skipping validation.")
                    continue

                # Evaluate over multiple seeds
                for seed in self.seeds:
                    mean_reward, std_reward = self.evaluate_model(best_agent, self.val_env, n_episodes=10, seed=seed)
                    seed_results[f"Seed_{seed}"] = (mean_reward, std_reward)
                    final_validation_results[name] = seed_results
                    print(f"{name} | Seed {seed} -> Mean: {mean_reward:.2f}, Std: {std_reward:.2f}")
            else:
                print(f"Error: Best model not found at path: {model_path} for {name}. Skipping validation.")

        return final_validation_results

    def visualize_validation_results(self, final_validation_results):
        """
        Visualize the final validation results using a boxplot.
        
        Parameters:
        final_validation_results (dict): The validation results for all models across all seeds
        """
        # -----------------------------------------------------
        # Step 5.1: Boxplot visualization of final validation results
        # -----------------------------------------------------

        plt.figure(figsize=(12, 7))
        plt.title("Final Validation: Reward Distribution Across Seeds for Best Models")

        # Prepare data for boxplot
        boxplot_data = []
        boxplot_labels = []
        for agent_name, seed_results in final_validation_results.items():
            if seed_results:
                rewards = [result[0] for result in seed_results.values()]
                boxplot_data.append(rewards)
                boxplot_labels.append(agent_name)

        if boxplot_data:
            plt.boxplot(boxplot_data, labels=boxplot_labels)
            plt.ylabel("Average Reward")
            plt.xlabel("RL Agent")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo validation results to display.")
