# =============================================
# Step 4: Systematic Hyperparameter Comparison 
# =============================================

import os
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN, QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from sklearn.model_selection import train_test_split
from emotion_env import EmotionEnv  

class HyperparameterComparison:
    # Initialization with hyperparameters and environment setup
    def __init__(self, total_timesteps=5000, num_eval_episodes=5, eval_freq=1000):
        self.total_timesteps = total_timesteps
        self.num_eval_episodes = num_eval_episodes
        self.eval_freq = eval_freq
        self.algorithms = ["PPO", "A2C", "DQN", "QRDQN"]
        self.hyperparameter_spaces = {
            "PPO": {
                "learning_rate": [0.0001, 0.0005],
                "gamma": [0.9, 0.99],
            },
            "A2C": {
                "learning_rate": [0.0001, 0.0007],
                "gamma": [0.9, 0.99],
            },
            "DQN": {
                "learning_rate": [0.00005, 0.0005],
                "gamma": [0.9, 0.95],
                "exploration_fraction": [0.1, 0.2],
                "buffer_size": [100000, 200000],
            },
            "QRDQN": {
                "learning_rate": [0.00005, 0.0005],
                "gamma": [0.9, 0.95],
                "exploration_fraction": [0.1, 0.2],
                "buffer_size": [100000, 200000],
            },
        }
        self.results = {}
        self.best_models = {}
        self.best_models_dir = "./tuned_models"
        os.makedirs(self.best_models_dir, exist_ok=True)
        self.temp_log_base_dir = "./logs/hyperparameter_comparison"
        
        # Setting up the environment for training and evaluation
        temp_env = EmotionEnv()
        X = temp_env.X
        y = temp_env.y
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
        self.train_env = EmotionEnv(features=X_train, labels=y_train)
        self.val_env = EmotionEnv(features=X_val, labels=y_val)

    # Method to evaluate the model performance
    def evaluate_model(self, model, env, num_episodes=10):
        episode_rewards = []
        for _ in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                step_result = env.step(action)
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                total_reward += reward
            episode_rewards.append(total_reward)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward

    # Method to run hyperparameter comparison for all algorithms
    def run_comparison(self):
        for algorithm_name in self.algorithms:
            print(f"\n--- Hyperparameter Comparison for {algorithm_name} ---")
            best_mean_reward_for_algo = -np.inf
            best_hyperparameters_for_algo = None
            final_best_model_path_for_algo = None

            if algorithm_name not in self.hyperparameter_spaces:
                print(f"Skipping {algorithm_name}: No hyperparameters defined.")
                continue

            self.results[algorithm_name] = []
            param_names = list(self.hyperparameter_spaces[algorithm_name].keys())
            param_values_list = list(self.hyperparameter_spaces[algorithm_name].values())
            param_combinations = list(itertools.product(*param_values_list))

            for i, params in enumerate(param_combinations):
                hyperparameters = dict(zip(param_names, params))
                print(f"\nTraining {algorithm_name} Run {i+1}/{len(param_combinations)} with hyperparameters: {hyperparameters}")

                log_dir = os.path.join(self.temp_log_base_dir, algorithm_name, f"run_{i+1}")
                os.makedirs(log_dir, exist_ok=True)

                eval_callback = EvalCallback(
                    self.val_env,
                    best_model_save_path=log_dir,
                    log_path=log_dir,
                    eval_freq=self.eval_freq,
                    n_eval_episodes=self.num_eval_episodes,
                    deterministic=False,
                    render=False,
                    verbose=0,
                )

                model = None
                try:
                    # Instantiate the model for the algorithm being tested
                    if algorithm_name == "PPO":
                        model = PPO("MlpPolicy", self.train_env, verbose=0, **hyperparameters, tensorboard_log=log_dir)
                    elif algorithm_name == "A2C":
                        model = A2C("MlpPolicy", self.train_env, verbose=0, **hyperparameters, tensorboard_log=log_dir)
                    elif algorithm_name == "DQN":
                        model = DQN("MlpPolicy", self.train_env, verbose=0, **hyperparameters, tensorboard_log=log_dir)
                    elif algorithm_name == "QRDQN":
                        model = QRDQN("MlpPolicy", self.train_env, verbose=0, **hyperparameters, tensorboard_log=log_dir)
                    else:
                        raise ValueError(f"Algorithm {algorithm_name} not supported or has no defined hyperparameters.")

                    # Training the model
                    model.learn(total_timesteps=self.total_timesteps, callback=eval_callback)
                    print(f"{algorithm_name} Run {i+1} trained.")

                    current_run_best_model_path = os.path.join(log_dir, "best_model.zip")

                    if os.path.exists(current_run_best_model_path):
                        print(f"Loading best model from this run: {current_run_best_model_path}")
                        loaded_model = None
                        if algorithm_name == "PPO":
                            loaded_model = PPO.load(current_run_best_model_path, env=self.val_env)
                        elif algorithm_name == "A2C":
                            loaded_model = A2C.load(current_run_best_model_path, env=self.val_env)
                        elif algorithm_name == "DQN":
                            loaded_model = DQN.load(current_run_best_model_path, env=self.val_env)
                        elif algorithm_name == "QRDQN":
                            loaded_model = QRDQN.load(current_run_best_model_path, env=self.val_env)

                        if loaded_model:
                            mean_reward, std_reward = self.evaluate_model(loaded_model, self.val_env, self.num_eval_episodes)
                            print(f"Evaluation of Run {i+1} best model: Mean reward={mean_reward:.2f} +/- {std_reward:.2f}")

                            self.results[algorithm_name].append(
                                {
                                    "hyperparameters": hyperparameters,
                                    "mean_reward": mean_reward,
                                    "std_reward": std_reward,
                                    "temp_model_path": current_run_best_model_path,
                                }
                            )

                            if mean_reward > best_mean_reward_for_algo:
                                print(f"*** New best model found for {algorithm_name}! Mean reward: {mean_reward:.2f} > {best_mean_reward_for_algo:.2f} ***")
                                best_mean_reward_for_algo = mean_reward
                                best_hyperparameters_for_algo = hyperparameters

                                target_best_model_path = os.path.join(self.best_models_dir, f"best_model_{algorithm_name}_tuned.zip")

                                try:
                                    shutil.copy(current_run_best_model_path, target_best_model_path)
                                    final_best_model_path_for_algo = target_best_model_path
                                    print(f"Copied best model to: {final_best_model_path_for_algo}")
                                except Exception as e:
                                    print(f"ERROR copying best model from {current_run_best_model_path} to {target_best_model_path}: {e}")
                                    final_best_model_path_for_algo = None

                    else:
                        print(f"Warning: No 'best_model.zip' found in {log_dir}. EvalCallback might not have found a better model during training.")
                        self.results[algorithm_name].append(
                            {
                                "hyperparameters": hyperparameters,
                                "mean_reward": -np.inf,
                                "std_reward": np.nan,
                                "temp_model_path": None,
                            }
                        )

                except Exception as e:
                    print(f"ERROR during training or evaluation for {algorithm_name} Run {i+1} with params {hyperparameters}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.results[algorithm_name].append(
                        {
                            "hyperparameters": hyperparameters,
                            "mean_reward": -np.inf,
                            "std_reward": np.nan,
                            "temp_model_path": None,
                            "error": str(e)
                        }
                    )

                finally:
                    print(f"Cleaning up temporary log directory: {log_dir}")
                    shutil.rmtree(log_dir, ignore_errors=True)
                    if 'model' in locals() and model is not None:
                        del model
                    if 'loaded_model' in locals() and loaded_model is not None:
                        del loaded_model

            if final_best_model_path_for_algo and best_hyperparameters_for_algo:
                self.best_models[algorithm_name] = final_best_model_path_for_algo
                print(f"\n--- Best model for {algorithm_name} finalized ---")
                print(f"  Saved at: {final_best_model_path_for_algo}")
                print(f"  Best Hyperparameters: {best_hyperparameters_for_algo}")
                print(f"  Best Mean Validation Reward: {best_mean_reward_for_algo:.2f}")
            else:
                print(f"\n--- No best model was successfully saved for {algorithm_name} ---")

        print("\n--- Hyperparameter Comparison Results Summary ---")
        for algorithm_name, result_list in self.results.items():
            print(f"\nResults for {algorithm_name}:")
            if result_list:
                sorted_results = sorted(result_list, key=lambda x: x.get("mean_reward", -np.inf), reverse=True)
                for i, res in enumerate(sorted_results):
                    print(f"  Rank {i+1}:")
                    print(f"    Hyperparameters: {res['hyperparameters']}")
                    mean_rew = res.get('mean_reward', 'N/A')
                    std_rew = res.get('std_reward', 'N/A')
                    print(f"    Mean Reward: {mean_rew} | Std. Dev.: {std_rew}")
            else:
                print(f"  No valid runs for {algorithm_name}")
