# =========================
# Step 8: Ensembling Agents
# =========================

import os
import random
import numpy as np
from collections import Counter
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3 import QRDQN
import matplotlib.pyplot as plt

class EnsembleAgent:
    """
    Class to handle the creation, evaluation, and comparison of ensemble models and individual models.
    """

    def __init__(self, best_tuned_models, test_env):
        """
        Initialize the EnsembleAgent with best-tuned models and environment.

        Args:
            best_tuned_models (dict): Dictionary containing the best-tuned models.
            test_env (gym.Env): The environment in which models will be evaluated.
        """
        self.best_tuned_models = best_tuned_models
        self.test_env = test_env
        self.best_agents_all = {}
        self.ensemble_agents_list = []
        self._load_best_tuned_models()
        self._create_ensemble_agent()
        self.ensemble_evaluation_results = self.evaluate_agent(self.ensemble_agent, self.test_env, n_episodes=100)

    def _load_best_tuned_models(self):
        """
        Load the best-tuned models (PPO, A2C, DQN, QRDQN) from the provided dictionary and paths.
        """
        if 'PPO' in self.best_tuned_models:
            ppo_path, _ = self.best_tuned_models['PPO']
            if os.path.exists(ppo_path):
                self.best_agents_all['PPO'] = PPO.load(ppo_path, env=self.test_env)
            else:
                print(f"Warning: PPO model not found at {ppo_path}")
        if 'A2C' in self.best_tuned_models:
            a2c_path, _ = self.best_tuned_models['A2C']
            if os.path.exists(a2c_path):
                self.best_agents_all['A2C'] = A2C.load(a2c_path, env=self.test_env)
            else:
                print(f"Warning: A2C model not found at {a2c_path}")
        # Add similar loading for DQN and QRDQN if uncommented in Phase 4
        if not self.best_agents_all:
            raise ValueError("No best agents were loaded. Check the model paths in 'best_tuned_models'.")

    def _create_ensemble_agent(self):
        """
        Create an ensemble agent using a majority voting mechanism across the best-tuned models.
        """
        def ensemble_predict(state, agents):
            actions = []
            for agent in agents:
                action, _ = agent.predict(state[np.newaxis, :])
                actions.append(int(action))  # Ensure action is hashable
            counts = Counter(actions)
            max_count = max(counts.values())
            candidate_actions = [a for a, count in counts.items() if count == max_count]
            return random.choice(candidate_actions)

        class SimpleEnsembleAgent:
            """
            A simple ensemble agent that uses majority voting across models.
            """
            def __init__(self, models):
                self.models = models

            def predict(self, observation, deterministic=False):
                all_actions = [int(model.predict(observation, deterministic=deterministic)[0]) for model in self.models]
                most_common = Counter(all_actions).most_common(1)
                return most_common[0][0], None

        # Load individual models
        for algo_name, path_params in self.best_tuned_models.items():
            path, _ = path_params
            if os.path.exists(path):
                print(f"Loading best {algo_name} model for ensemble...")
                algo_name_lower = algo_name.lower()
                model = None
                if "qrdqn" in algo_name_lower:
                    model = QRDQN.load(path, env=self.test_env)
                elif "dqn" in algo_name_lower:
                    model = DQN.load(path, env=self.test_env)
                elif "ppo" in algo_name_lower:
                    model = PPO.load(path, env=self.test_env)
                elif "a2c" in algo_name_lower:
                    model = A2C.load(path, env=self.test_env)
                else:
                    print(f"Warning: Unknown algorithm '{algo_name}' for loading.")

                if model:
                    self.ensemble_agents_list.append(model)
                else:
                    print(f"Warning: Could not load {algo_name} model for ensemble.")
            else:
                print(f"Warning: Model not found at {path} for {algo_name}.")

        # Create the ensemble agent
        self.ensemble_agent = SimpleEnsembleAgent(self.ensemble_agents_list)

    def evaluate_agent(self, agent, env, n_episodes=100):
        """
        Evaluate the agent over a number of episodes.

        Args:
            agent: The agent to be evaluated.
            env (gym.Env): The environment in which the agent will be evaluated.
            n_episodes (int): The number of episodes to run for evaluation.

        Returns:
            dict: A dictionary containing mean reward and all rewards.
        """
        all_rewards = []
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=episode)
            done = False
            total_reward = 0
            while not done:
                action, _ = agent.predict(obs)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
            all_rewards.append(total_reward)
        mean_reward = np.mean(all_rewards)
        return {"mean_reward": mean_reward, "all_rewards": all_rewards}

    def compare_results(self, individual_results):
        """
        Compare the results of the ensemble agent with individual agents.

        Args:
            individual_results (dict): Results of individual agent evaluations.

        Returns:
            None
        """
        print("\n--- Comparison of Ensemble vs. Individual Best Agents ---")
        print(f"Ensemble Mean Reward (based on: {list(self.best_tuned_models.keys())}): {self.ensemble_evaluation_results['mean_reward']:.3f}")

        for algo, results in individual_results.items():
            print(f"{algo} Mean Reward: {results['mean_reward']:.3f}")

        # Plotting Results
        self.plot_results(individual_results)

    def plot_results(self, individual_results):
        """
        Plot comparison of the mean rewards and distribution of rewards.

        Args:
            individual_results (dict): Results of individual agent evaluations.

        Returns:
            None
        """
        # Bar Chart for Mean Rewards
        plt.figure(figsize=(12, 7))
        algo_names = list(individual_results.keys()) + ["Ensemble"]
        mean_rewards = [results["mean_reward"] for results in individual_results.values()] + [self.ensemble_evaluation_results["mean_reward"]]
        colors = ['blue', 'green', 'red', 'purple', 'orange'][:len(algo_names)]
        plt.bar(algo_names, mean_rewards, color=colors)
        plt.ylabel("Mean Reward")
        plt.title("Comparison of Mean Rewards: Ensemble vs. Individual Best Agents")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        # Box Plot for Reward Distributions
        plt.figure(figsize=(12, 7))
        all_data = [results["all_rewards"] for results in individual_results.values()] + [self.ensemble_evaluation_results["all_rewards"]]
        plt.boxplot(all_data, labels=algo_names)
        plt.ylabel("Reward")
        plt.title("Distribution of Rewards: Ensemble vs. Individual Best Agents")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()