import numpy as np
import pandas as pd
import time
import os
import re
import json
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from visualizer.plant_visualizer import get_health_state, load_crop_image


class CustomDQN(DQN):

    def __init__(
        self,
        *args,
        initial_eps: float = 1.0,
        final_eps: float = 0.05,
        exploration_steps: int = 10_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.exploration_steps = exploration_steps
        self.current_eps = initial_eps

    def _get_epsilon(self) -> float:
        """Compute current epsilon based on linear decay."""
        if self._n_calls >= self.exploration_steps:
            return self.final_eps
        fraction = self._n_calls / self.exploration_steps
        return self.initial_eps - fraction * (self.initial_eps - self.final_eps)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Custom prediction with ε‑greedy exploration."""
        # Update epsilon based on internal counter
        self.current_eps = self._get_epsilon()

        # Ensure observation is 2‑D array as expected by SB3
        observation = np.asarray(observation)
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]

        if not deterministic and np.random.random() < self.current_eps:
            # Random action (exploration)
            action = self.env.action_space.sample()
            action = np.array([action], dtype=np.int64)
        else:
            # Best action (exploitation)
            action, _ = super().predict(
                observation, state, episode_start, deterministic=True
            )
            action = np.asarray(action, dtype=np.int64)
            if action.ndim == 0:
                action = action[np.newaxis]

        return action, None  # state=None for SB3 API compatibility


class DQNTrainerSimplified:
    """Simplified DQN Trainer that focuses on core training functionality."""

    def __init__(self, model_version_manager=None, log_dir: str = "./logs/dqn"):
        self.model_manager = model_version_manager
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.model = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_number(value):
        """Pretty‑print numbers: scientific for big, fixed point otherwise."""
        if isinstance(value, (int, float)):
            if abs(value) > 1_000:
                return f"{value:.2e}"
            return f"{value:.2f}"
        return str(value)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        env,
        total_timesteps: int = 100_000,
        learning_rate: float = 1e-4,
        buffer_size: int = 10_000,
        exploration_steps: int = 10_000,
        initial_eps: float = 1.0,
        final_eps: float = 0.05,
        gamma: float = 0.99,
        use_adaptive_lr: bool = False,
        min_lr: float = 1e-6,
        lr_decay_factor: float = 0.95,
        lr_update_freq: int = 10_000,
        policy: str = "MlpPolicy",
        seed: int | None = 42,
        save_model: bool = True,
        model_metadata: dict | None = None,
        eval_freq: int = 5_000,
    ):
        start_time = time.time()

        model = CustomDQN(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            gamma=gamma,
            exploration_fraction=exploration_steps / total_timesteps,
            exploration_initial_eps=initial_eps,
            exploration_final_eps=final_eps,
            verbose=0,
            seed=seed,
            initial_eps=initial_eps,
            final_eps=final_eps,
            exploration_steps=exploration_steps,
        )

        callbacks: list[BaseCallback] = []

        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"{self.log_dir}/",
            log_path=f"{self.log_dir}/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        model_path = None
        if save_model:
            if self.model_manager is not None:
                if model_metadata is None:
                    model_metadata = {}
                model_metadata.update(
                    {
                        "learning_rate": learning_rate,
                        "final_lr": (learning_rate),
                        "buffer_size": buffer_size,
                        "exploration_steps": exploration_steps,
                        "initial_eps": initial_eps,
                        "final_eps": final_eps,
                        "gamma": gamma,
                        "training_time": training_time,
                        "total_timesteps": total_timesteps,
                    }
                )
                model_path = self.model_manager.save_model(model, "dqn", model_metadata)
            else:
                model_path = f"{self.log_dir}/final_model"
                model.save(model_path)
                print(f"Model saved to {model_path}")

        return model, training_time, model_path
    
    def _create_new_model(self, env, policy, learning_rate, buffer_size, 
                         exploration_steps, initial_eps, final_eps, gamma, seed):
        """Create a new DQN model with the specified parameters."""
        return CustomDQN(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            gamma=gamma,
            exploration_fraction=exploration_steps / 100000,  # Default total_timesteps
            exploration_initial_eps=initial_eps,
            exploration_final_eps=final_eps,
            verbose=0,
            seed=seed,
            initial_eps=initial_eps,
            final_eps=final_eps,
            exploration_steps=exploration_steps,
        )
    
    def _load_existing_model(self, model_path=None):
        """
        Load an existing model from the specified path or find the latest model.
        
        Parameters:
        - model_path: Path to a specific model to load. If None, will try to find the latest model.
        
        Returns:
        - Loaded model or None if no model is found
        """
        try:
            # If a specific model path is provided, try to load it
            if model_path is not None:
                if os.path.exists(model_path):
                    # Extract version information
                    version_match = re.search(r'v(\d+)', model_path)
                    version = version_match.group(1) if version_match else "Unknown"
                    print(f"Loading specified model: {model_path} (Version: v{version})")
                    return CustomDQN.load(model_path)
                else:
                    print(f"Model not found at {model_path}")
                    return None
            
            # Try to load the best model from the log directory
            best_model_path = f"{self.log_dir}/best_model.zip"
            if os.path.exists(best_model_path):
                print(f"Loading best model from {best_model_path} (Log directory)")
                return CustomDQN.load(best_model_path)
            
            # Try to load the final model from the log directory
            final_model_path = f"{self.log_dir}/final_model.zip"
            if os.path.exists(final_model_path):
                print(f"Loading final model from {final_model_path} (Log directory)")
                return CustomDQN.load(final_model_path)
            
            # If model manager is available, try to get the latest model
            if self.model_manager is not None:
                latest_version = self.model_manager.get_latest_version("dqn")
                if latest_version > 0:
                    latest_model_path = self.model_manager.get_model_path("dqn", latest_version)
                    # Ensure .zip extension is added
                    if not latest_model_path.endswith(".zip"):
                        latest_model_path += ".zip"
                    print(f"Loading latest model from model manager: {latest_model_path} (Version: v{latest_version})")
                    
                    # Try to load metadata to display more information
                    try:
                        metadata_path = latest_model_path.replace('.zip', '_metadata.json')
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                timestamp = metadata.get('timestamp', 'Unknown')
                                print(f"Model Info - Created: {timestamp}")
                    except Exception as e:
                        print(f"Error loading metadata: {e}")
                    
                    return CustomDQN.load(latest_model_path)
            
            # No model found
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # ------------------------------------------------------------------
    # Evaluation of the model (rewards but no visual one)
     # ------------------------------------------------------------------
    def evaluate(self, model, env, n_eval_episodes=10, deterministic=True):
        """Simplified evaluation method for a DQN model."""
        results = []

        for episode in range(n_eval_episodes):
            obs, _ = env.reset(seed=episode)  # Different seed per episode
            done = False
            total_reward = 0
            total_water = 0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                # Handle action format - ensure it's an integer
                if isinstance(action, np.ndarray):
                    action = int(action[0])

                obs, reward, done, _, info = env.step(action)
                total_reward += reward

                # Track water usage if available
                water_applied = (
                    env.water_amounts[action] if hasattr(env, "water_amounts") else 0
                )
                total_water += water_applied

            # Record episode results
            results.append(
                {
                    "episode": episode,
                    "total_reward": total_reward,
                    "final_yield": getattr(env, "yield_potential", 0),
                    "total_water_used": total_water,
                    "water_efficiency": getattr(env, "yield_potential", 0)
                    / (total_water + 1e-8),
                }
            )

        # Calculate averages
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        avg_yield = sum(r["final_yield"] for r in results) / len(results)
        avg_water = sum(r["total_water_used"] for r in results) / len(results)

        # Print summary
        print("\n===== Evaluation Results =====")
        print(f"Episodes: {n_eval_episodes}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Yield: {avg_yield:.2f}")
        print(f"Average Water Used: {avg_water:.2f} mm")

        return results

    # ------------------------------------------------------------------
    # Evaluation with Visualization
     # ------------------------------------------------------------------
    def evaluate_model(self, model, env, n_eval_episodes: int = 10):
        """Evaluate the trained model and return a DataFrame of metrics."""
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        print("\n💦 Irrigation Report 💦")
        print(f"🌟 Reward: {self._format_number(mean_reward)}")
        print(f"🎯 Variation: ±{self._format_number(std_reward)}")

        episode_metrics: list[dict] = []

        for episode in range(n_eval_episodes):
            obs, _ = env.reset(seed=episode)
            done = False
            total_reward = 0.0

            # Time‑series storage
            trajectory: dict[str, list] = {
                "soil_moisture": [],
                "water_used": [],
                "yield_potential": [],
                "rewards": [],
            }
            images: list[dict] = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action[0] if isinstance(action, np.ndarray) else action)

                obs, reward, done, _, _ = env.step(action)
                total_reward += reward

                # --- logging per‑step data ---
                # --- capture soil moisture with robust fallbacks ---
                if hasattr(env, "current_soil_moisture"):
                    sm_value = env.current_soil_moisture
                elif hasattr(env, "soil_moisture"):
                    sm_value = env.soil_moisture
                else:
                    # last‑chance: try first element of observation if scalar
                    try:
                        sm_value = float(obs[0])
                    except Exception:
                        sm_value = np.nan
                trajectory["soil_moisture"].append(sm_value)
                water_used_step = 0
                if hasattr(env, "water_amounts"):
                    water_amounts = getattr(env, "water_amounts")
                    if 0 <= action < len(water_amounts):
                        water_used_step = water_amounts[action]
                trajectory["water_used"].append(water_used_step)
                trajectory["yield_potential"].append(getattr(env, "yield_potential", 0))
                trajectory["rewards"].append(reward)

                # Capture images every 10 days or at end
                if getattr(env, "current_day", 0) % 10 == 0 or done:
                    images.append(
                        {
                            "day": getattr(env, "current_day", 0),
                            "yield_potential": getattr(env, "yield_potential", 0),
                        }
                    )

            # --------- aggregate episode metrics ---------
            total_water_used = (
                getattr(env, "total_water_used", float("nan"))
                if hasattr(env, "total_water_used")
                else float(np.sum(trajectory["water_used"]))
            )
            episode_metrics.append(
                {
                    "episode": episode,
                    "total_reward": total_reward,
                    "final_yield_potential": getattr(env, "yield_potential", 0),
                    "total_water_used": total_water_used,
                    "avg_soil_moisture": float(
                        np.nanmean(
                            [v for v in trajectory["soil_moisture"] if not np.isnan(v)]
                        )
                    ),
                }
            )

            # DQN Model Own Visualization using image grid :D

            if images:
                cols = min(3, len(images))
                rows = (len(images) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                axes = np.atleast_2d(axes)
                for idx, data in enumerate(images):
                    row, col = divmod(idx, cols)
                    ax = axes[row, col]
                    img = load_crop_image(data["yield_potential"])
                    if img is not None:
                        ax.imshow(img)
                    ax.axis("off")
                    health_state = get_health_state(
                        data["yield_potential"]
                    ).capitalize()
                    ax.set_title(
                        f"Day {data['day']}\n{health_state} (Yield: {data['yield_potential']:.2f})"
                    )
                # turn off unused axes
                for idx in range(len(images), rows * cols):
                    row, col = divmod(idx, cols)
                    axes[row, col].axis("off")
                plt.suptitle(f"Crop Health – Episode {episode + 1}")
                plt.tight_layout()
                plt.show()

        # ------------------- summary -------------------
        df = pd.DataFrame(episode_metrics)
        stats = df.describe()
        print("\n📊 Crop Report 📊")
        print(
            f"🌽 Yield: {self._format_number(stats.loc['mean', 'final_yield_potential'])} "
            f"(range: {self._format_number(stats.loc['min', 'final_yield_potential'])} – "
            f"{self._format_number(stats.loc['max', 'final_yield_potential'])})"
        )
        print(
            f"💧 Water Used: {self._format_number(stats.loc['mean', 'total_water_used'])} mm"
        )
        print(
            f"🌱 Moisture: {self._format_number(stats.loc['mean', 'avg_soil_moisture'])}% avg"
        )
        print(f"📈 Seasons: {int(stats.loc['count', 'total_reward'])}")
        return df

    def run_single_episode(
        self, model, env, render: bool = True, seed=None, visualize: bool = True
    ):
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        day = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action[0] if isinstance(action, np.ndarray) else action)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            day += 1
            if render and day % 10 == 0:
                stage_name = env.growth_stages[env.get_current_growth_stage()]["name"]
                print(
                    f"Day {day:3d} | Stage: {stage_name:9s} | Action: {action} "
                    f"({env.water_amounts[action]:2d}mm) | Soil Moisture: "
                    f"{getattr(env, 'soil_moisture', np.nan):4.1f}% | Yield: {env.yield_potential:.2f}"
                )

        if visualize:
            env.plot_episode()
        return (
            total_reward,
            env.yield_potential,
            getattr(env, "total_water_used", np.nan),
        )

    # ------------------------------------------------------------------
    # Plot the DQN Model Graph (In Blue Slightly Different from the others members)
     # ------------------------------------------------------------------
    def plot_simple_training_progress(self, log_path=None):
        """
        Plot training progress from evaluation logs.

        Parameters:
        - log_path (optional): Path to the evaluations.npz file directory.
                            Defaults to self.log_dir.
        """
        if log_path is None:
            log_path = self.log_dir

        try:
            results = np.load(f"{log_path}/evaluations.npz")
            timesteps = results["timesteps"]
            rewards = results["results"]

            mean_rewards = np.mean(rewards, axis=1)
            std_rewards = np.std(rewards, axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, mean_rewards, label="Mean Reward", color="blue")
            plt.fill_between(
                timesteps,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.2,
                color="blue",
                label="Reward Range",
            )
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title("Irrigation Agent Training Progress")
            plt.legend()
            plt.grid(True)
            plt.show()

        except FileNotFoundError:
            print("⚠️ Oops! No training logs found at", log_path)
