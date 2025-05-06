# ======================================
# Step 4.1: Hyperparameter  Optimization
# ======================================

import os
import random
import shutil
import tempfile
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.envs import DummyVecEnv
import gym

# Step 4.1: Define the Hyperparameter Optimization Class
class HyperparameterOptimization:
    def __init__(self, train_env, val_env, total_timesteps=10000, n_random_samples=1, best_models_perm_dir="./tuned_random_search_models"):
        # Initial setup for training and evaluation environments
        self.train_env = train_env
        self.val_env = val_env
        self.total_timesteps = total_timesteps
        self.n_random_samples = n_random_samples
        self.best_models_perm_dir = best_models_perm_dir
        os.makedirs(self.best_models_perm_dir, exist_ok=True)
        self.best_tuned_models = {}

        # Define the random search space for each algorithm
        self.random_search_space = {
            "PPO": {
                "learning_rate": [0.00005, 0.001],
                "gamma": [0.9, 0.999],
                "n_steps": [256, 512, 1024, 2048],
                "ent_coef": [0.0, 0.02],
                "clip_range": [0.1, 0.3]
            },
            "A2C": {
                "learning_rate": [0.0001, 0.001],
                "gamma": [0.9, 0.99],
                "n_steps": [5, 16, 32],
                "ent_coef": [0.0, 0.05]
            },
            "DQN": {
                "learning_rate": [0.00005, 0.001],
                "gamma": [0.9, 0.995],
                "batch_size": [32, 64, 128],
                "buffer_size": [50000, 100000, 200000],
                "exploration_fraction": [0.05, 0.3],
                "exploration_final_eps": [0.01, 0.1]
            },
            "QRDQN": {
                "learning_rate": [0.00005, 0.001],
                "gamma": [0.9, 0.995],
                "batch_size": [32, 64, 128],
                "buffer_size": [50000, 100000, 200000],
                "exploration_fraction": [0.05, 0.3],
                "exploration_final_eps": [0.01, 0.1]
            }
        }

    def sample_params(self, space):
        """Method to sample random parameters from the search space."""
        params = {}
        for key, value in space.items():
            if isinstance(value, list):
                if len(value) == 2 and isinstance(value[0], float):
                    params[key] = random.uniform(value[0], value[1])
                elif len(value) == 2 and isinstance(value[0], int) and key in ["n_steps", "batch_size", "buffer_size"]:
                    params[key] = random.choice(value)
                elif all(isinstance(v, (int, float, str)) for v in value):
                    params[key] = random.choice(value)
                else:
                    print(f"Warning: Unsupported type for parameter '{key}'. Using default or skipping.")
            else:
                params[key] = value
        for int_key in ["n_steps", "batch_size", "buffer_size"]:
            if int_key in params:
                params[int_key] = int(params[int_key])
        return params

    def optimize(self):
        """Method to perform random search hyperparameter optimization."""
        print("\n--- Phase 4 (Refined): Advanced Hyperparameter Optimization ---")

        # Random Search
        for algorithm_name, search_space in self.random_search_space.items():
            print(f"\n--- Random Search for {algorithm_name} ---")
            best_reward_for_algo = -np.inf
            best_params_for_algo = None
            final_best_model_path_for_algo = None

            for i in range(self.n_random_samples):
                random_params = self.sample_params(search_space)
                print(f"\nTesting random parameters {i+1}/{self.n_random_samples} for {algorithm_name}: {random_params}")

                with tempfile.TemporaryDirectory() as log_dir:
                    callback = EvalCallback(
                        self.val_env,
                        best_model_save_path=log_dir,  # Save best model *temporarily* here
                        log_path=log_dir,
                        eval_freq=max(self.total_timesteps // 10, 500),  # Evaluate reasonably often
                        deterministic=False,
                        render=False,
                        n_eval_episodes=5,
                        verbose=0
                    )

                    model = None
                    try:
                        # Instantiate the model
                        model_class = globals()[algorithm_name]
                        model = model_class("MlpPolicy", self.train_env, verbose=0, **random_params)

                        # Train the model
                        model.learn(total_timesteps=self.total_timesteps, callback=callback)
                        print(f"  Run {i+1} trained.")

                        current_run_reward = callback.best_mean_reward

                        if current_run_reward > best_reward_for_algo:
                            print(f"  *** New best reward for {algorithm_name}: {current_run_reward:.3f} (previous: {best_reward_for_algo:.3f}) ***")
                            best_reward_for_algo = current_run_reward
                            best_params_for_algo = random_params

                            temp_best_model_path = os.path.join(log_dir, "best_model.zip")

                            if os.path.exists(temp_best_model_path):
                                perm_dest_path = os.path.join(self.best_models_perm_dir, f"best_tuned_model_{algorithm_name}.zip")

                                try:
                                    shutil.copy(temp_best_model_path, perm_dest_path)
                                    final_best_model_path_for_algo = perm_dest_path  # Update the pointer to the permanent path
                                    print(f"  Copied new best model to: {final_best_model_path_for_algo}")
                                except Exception as copy_e:
                                    print(f"  ERROR copying best model from {temp_best_model_path} to {perm_dest_path}: {copy_e}")
                                    final_best_model_path_for_algo = None  # Don't point to a failed copy
                            else:
                                print(f"  Warning: Callback reported best reward {current_run_reward:.3f}, but best_model.zip not found at {temp_best_model_path}")
                        else:
                            print(f"  Run reward ({current_run_reward:.3f}) did not improve best ({best_reward_for_algo:.3f})")

                    except Exception as e:
                        import traceback
                        print(f"  ERROR during training/evaluation for run {i+1} with params {random_params}:")
                        traceback.print_exc()  # Print full traceback

                    if model is not None:
                        del model

            if final_best_model_path_for_algo and best_params_for_algo:
                self.best_tuned_models[algorithm_name] = (final_best_model_path_for_algo, best_params_for_algo)
                print(f"\n---> Best overall parameters found for {algorithm_name}: {best_params_for_algo}")
                print(f"     Best validation reward: {best_reward_for_algo:.3f}")
                print(f"     Best model saved to: {final_best_model_path_for_algo}")
            else:
                print(f"\n---> No successful best model saved for {algorithm_name} after random search.")

        print("\n--- Advanced Hyperparameter Optimization Complete ---")
        print("\nFinal best models and parameters found:")
        if self.best_tuned_models:
            for algo, (path, params) in self.best_tuned_models.items():
                print(f"  Algorithm: {algo}")
                print(f"    Model Path: {path}")
                print(f"    Parameters: {params}")
        else:
            print("  No models were successfully optimized and saved.")
        
        print("\n--- Advanced Hyperparameter Optimization Complete ---")