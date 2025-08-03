import numpy as np
import pandas as pd
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os
import pickle
import glob

from visualizer.plant_visualizer import get_health_state, load_crop_image



class DynaQAgent:
    """
    Improved Dyna-Q Agent for corn irrigation environment.
    Dyna-Q combines Q-learning with a learned model for planning.
    """

    def __init__(
        self,
        env,
        alpha=0.3,
        gamma=0.95,
        epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=0.995,
        planning_steps=30,
        bins=20,
        log_dir="./logs/dyna-q",
    ):
        """
        Initialize the Dyna-Q agent with parameters.

        Parameters:
        - env: The environment to learn from
        - alpha: Learning rate
        - gamma: Discount factor
        - epsilon: Initial exploration rate
        - min_epsilon: Minimum exploration rate
        - epsilon_decay: Rate of exploration decay
        - planning_steps: Number of planning steps after each real experience
        - bins: Number of bins for state discretization
        - log_dir: Directory to save logs
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.planning_steps = planning_steps
        self.bins = bins
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Initialize the environment and observation space
        obs_sample, _ = env.reset()
        self.obs_dim = len(obs_sample)
        self.n_actions = env.action_space.n

        # Initialize Q-table and model
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.model = dict()  # (state, action) -> (next_state, reward)

        # Initialize bins for discretization - improves generalization
        self.obs_bins = [
            np.linspace(low, high, bins - 1)
            for low, high in zip(env.observation_space.low, env.observation_space.high)
        ]

        # Tracking metrics
        self.episode_rewards = []
        self.episode_yields = []
        self.episode_water_usage = []
        self.episode_lengths = []

    def discretize(self, obs):
        """
        Convert continuous observation to discrete state representation.

        Parameters:
        - obs: Continuous observation from environment

        Returns:
        - Tuple representing the discretized state
        """
        return tuple(np.digitize(o, bins) for o, bins in zip(obs, self.obs_bins))

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Parameters:
        - state: Current state

        Returns:
        - Selected action (integer)
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        """
        Update Q-value using temporal difference learning.

        Parameters:
        - state: Current state
        - action: Action taken
        - reward: Reward received
        - next_state: New state
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def planning(self):
        """
        Perform planning steps using the learned model.
        This is the 'Dyna' part of Dyna-Q, using simulated experience.
        """
        if not self.model:
            return  # Skip if model is empty

        for _ in range(self.planning_steps):
            # Randomly select a state-action pair from experiences
            (state, action), (next_state, reward) = random.choice(
                list(self.model.items())
            )
            # Update Q-values using simulated experience
            self.update(state, action, reward, next_state)

    def train(
        self,
        episodes=500,
        max_steps=1000,
        render_interval=50,
        save_model=True,
        continue_training=True,
        model_path=None,
    ):
        """
        Train the agent for the specified number of episodes.

        Parameters:
        - episodes: Number of training episodes
        - max_steps: Maximum steps per episode
        - render_interval: Interval for printing progress
        - save_model: Whether to save Q-table at the end
        - continue_training: Whether to continue training from a previous model if available
        - model_path: Specific path to a model to load (if None, will try to find the latest model)

        Returns:
        - List of episode rewards
        """
        # Try to load existing model if continue_training is True
        if continue_training:
            loaded_model = self._load_existing_model(model_path)
            if loaded_model:
                self.q_table = defaultdict(
                    lambda: np.zeros(self.n_actions), loaded_model["q_table"]
                )
                self.model = loaded_model.get("model", {})
                self.epsilon = loaded_model.get("epsilon", self.epsilon)

                # Load training metrics if available
                if "episode_rewards" in loaded_model:
                    self.episode_rewards = loaded_model["episode_rewards"]
                if "episode_yields" in loaded_model:
                    self.episode_yields = loaded_model["episode_yields"]
                if "episode_water_usage" in loaded_model:
                    self.episode_water_usage = loaded_model["episode_water_usage"]
                if "episode_lengths" in loaded_model:
                    self.episode_lengths = loaded_model["episode_lengths"]

                print("\n===== Model Training Summary =====")
                print(f"Continuing training from existing model")
                print(f"Model path: {model_path if model_path else 'Auto-detected'}")
                print(f"Additional episodes: {episodes}")
                print(f"Current epsilon: {self.epsilon:.4f}")
                print("=======================\n")
            else:
                print("No existing model found. Starting training from scratch.")

        start_time = time.time()
        rewards = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.discretize(obs)
            total_reward = 0
            done = False
            step_count = 0

            # Run episode
            while not done and step_count < max_steps:
                # Choose and execute action
                action = self.choose_action(state)
                next_obs, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize(next_obs)
                total_reward += reward

                # Update Q-table with real experience
                self.update(state, action, reward, next_state)

                # Store transition in model for planning
                self.model[(state, action)] = (next_state, reward)

                # Perform planning
                self.planning()

                # Move to next state
                state = next_state
                step_count += 1

            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Track episode metrics
            rewards.append(total_reward)
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step_count)

            if hasattr(self.env, "yield_potential"):
                self.episode_yields.append(self.env.yield_potential)
            if hasattr(self.env, "total_water_used"):
                self.episode_water_usage.append(self.env.total_water_used)

            # Print progress
            if ep % render_interval == 0 or ep == episodes - 1:
                avg_reward = (
                    np.mean(rewards[-render_interval:])
                    if len(rewards) >= render_interval
                    else np.mean(rewards)
                )
                print(
                    f"Episode {ep+1}/{episodes}: Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}, Epsilon = {self.epsilon:.4f}"
                )

                if hasattr(self.env, "yield_potential") and hasattr(
                    self.env, "total_water_used"
                ):
                    print(
                        f"  Yield: {self.env.yield_potential:.2f}, Water Used: {self.env.total_water_used:.1f} mm"
                    )

        # Save Q-table if requested
        if save_model:
            self.save_model(f"{self.log_dir}/dyna_q_model.pkl")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Plot training progress
        self.plot_training_progress()

        return rewards

    def save_model(self, filepath):
        """Save the complete model to a file."""
        model_data = {
            "q_table": dict(self.q_table),  # Convert defaultdict to regular dict
            "model": self.model,
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_yields": self.episode_yields,
            "episode_water_usage": self.episode_water_usage,
            "episode_lengths": self.episode_lengths,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")
        return filepath

    def _load_existing_model(self, model_path=None):
        """
        Load an existing model from the specified path or find the latest model.

        Parameters:
        - model_path: Path to a specific model to load. If None, will try to find the latest model.

        Returns:
        - Loaded model data or None if no model is found
        """
        try:
            # If a specific model path is provided, try to load it
            if model_path is not None:
                if os.path.exists(model_path):
                    print(f"Loading specified model: {model_path}")
                    with open(model_path, "rb") as f:
                        model_data = pickle.load(f)
                    return model_data
                else:
                    print(f"Model not found at {model_path}")
                    return None

            # Try to find the latest model in the log directory
            model_files = glob.glob(f"{self.log_dir}/dyna_q_model*.pkl")
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                print(f"Loading latest model from: {latest_model}")
                with open(latest_model, "rb") as f:
                    model_data = pickle.load(f)
                return model_data

            # No model found
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def save_q_table(self, filepath):
        """Save the Q-table to a file (legacy method)."""
        # Convert defaultdict to regular dict for saving
        q_dict = dict(self.q_table)

        with open(filepath, "wb") as f:
            pickle.dump(q_dict, f)

        print(f"Q-table saved to: {filepath}")
        return filepath

    def load_q_table(self, filepath):
        """Load a Q-table from a file."""
        try:
            q_dict = np.load(filepath, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
            self.q_table.update(q_dict)
            print(f"Q-table loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            return False

    def evaluate(self, n_episodes=10, render=False):
        """
        Evaluate the trained agent without exploration.

        Parameters:
        - n_episodes: Number of evaluation episodes
        - render: Whether to print episode progress

        Returns:
        - Dictionary with evaluation metrics
        """
        eval_rewards = []
        eval_yields = []
        eval_water = []

        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=ep)  # Different seed for each episode
            state = self.discretize(obs)
            total_reward = 0
            done = False

            while not done:
                # Choose best action (no exploration)
                action = int(np.argmax(self.q_table[state]))

                # Execute action
                next_obs, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize(next_obs)
                total_reward += reward

                if render and (done or self.env.current_day % 10 == 0):
                    current_stage = self.env.get_current_growth_stage()
                    stage_name = self.env.growth_stages[current_stage]["name"]
                    water_applied = self.env.water_amounts[action]
                    print(
                        f"Day {self.env.current_day:3d} | Stage: {stage_name:9s} | "
                        f"Action: {action} ({water_applied:2d}mm) | "
                        f"Soil Moisture: {self.env.soil_moisture:4.1f}% | "
                        f"Yield: {self.env.yield_potential:.2f}"
                    )

                # Move to next state
                state = next_state

            # Record episode results
            eval_rewards.append(total_reward)

            if hasattr(self.env, "yield_potential"):
                eval_yields.append(self.env.yield_potential)
            if hasattr(self.env, "total_water_used"):
                eval_water.append(self.env.total_water_used)

            if render:
                print(f"\nEpisode {ep+1} Summary:")
                print(f"Total Reward: {total_reward:.2f}")
                if hasattr(self.env, "yield_potential"):
                    print(f"Final Yield: {self.env.yield_potential:.2f}")
                if hasattr(self.env, "total_water_used"):
                    print(f"Total Water Used: {self.env.total_water_used:.1f} mm")
                print("-" * 40)

        # Calculate averages
        avg_reward = np.mean(eval_rewards)
        avg_yield = np.mean(eval_yields) if eval_yields else None
        avg_water = np.mean(eval_water) if eval_water else None

        # Print summary
        print("\n===== Evaluation Results =====")
        print(f"Episodes: {n_episodes}")
        print(f"Average Reward: {avg_reward:.2f} ± {np.std(eval_rewards):.2f}")
        if avg_yield is not None:
            print(f"Average Yield: {avg_yield:.2f}")
        if avg_water is not None:
            print(f"Average Water Used: {avg_water:.2f} mm")

        # Return results
        results = {
            "rewards": eval_rewards,
            "mean_reward": avg_reward,
            "std_reward": np.std(eval_rewards),
        }

        if eval_yields:
            results["yields"] = eval_yields
            results["mean_yield"] = avg_yield

        if eval_water:
            results["water_usage"] = eval_water
            results["mean_water"] = avg_water

        return results
    
    def evaluate_model(self, n_eval_episodes: int = 10):
        """
        Evaluate the Dyna-Q agent using its learned Q-table and own visualization.
        Captures per-episode metrics and displays an image grid of crop health.
        Returns a pandas DataFrame of episode metrics.
        """
        episode_metrics = []

        for episode in range(n_eval_episodes):
            obs, _ = self.env.reset(seed=episode)
            state = self.discretize(obs)
            done = False
            total_reward = 0.0

            # Time-series storage
            trajectory = {
                "soil_moisture": [],
                "water_used": [],
                "yield_potential": [],
                "rewards": [],
            }
            images = []

            while not done:
                # Select best action
                action = int(np.argmax(self.q_table[state]))

                # Step environment
                obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                # Log trajectory data
                sm_value = getattr(self.env, 'soil_moisture', np.nan)
                trajectory['soil_moisture'].append(sm_value)
                water_amounts = getattr(self.env, 'water_amounts', [])
                water_used_step = water_amounts[action] if 0 <= action < len(water_amounts) else 0
                trajectory['water_used'].append(water_used_step)
                yield_val = getattr(self.env, 'yield_potential', 0)
                trajectory['yield_potential'].append(yield_val)
                trajectory['rewards'].append(reward)

                # Capture images every 10 days or at end
                day = getattr(self.env, 'current_day', 0)
                if day % 10 == 0 or done:
                    images.append({
                        'day': day,
                        'yield_potential': yield_val,
                    })

                # Move to next state
                state = self.discretize(obs)

            # Aggregate metrics
            total_water = getattr(self.env, 'total_water_used', float(np.sum(trajectory['water_used'])))
            avg_sm = float(np.nanmean([v for v in trajectory['soil_moisture'] if not np.isnan(v)]))
            final_yield = getattr(self.env, 'yield_potential', 0)

            episode_metrics.append({
                'episode': episode,
                'total_reward': total_reward,
                'final_yield_potential': final_yield,
                'total_water_used': total_water,
                'avg_soil_moisture': avg_sm,
            })

            # Visualization grid
            if images:
                cols = min(3, len(images))
                rows = (len(images) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
                axes = np.atleast_2d(axes)
                for idx, data in enumerate(images):
                    row, col = divmod(idx, cols)
                    ax = axes[row, col]
                    img = load_crop_image(data['yield_potential'])
                    if img is not None:
                        ax.imshow(img)
                    ax.axis('off')
                    health_state = get_health_state(data['yield_potential']).capitalize()
                    ax.set_title(f"Day {data['day']}\n{health_state} (Yield: {data['yield_potential']:.2f})")
                # Turn off unused axes
                for idx in range(len(images), rows * cols):
                    row, col = divmod(idx, cols)
                    axes[row, col].axis('off')
                plt.suptitle(f"Crop Health – Episode {episode + 1}")
                plt.tight_layout()
                plt.show()

        # Summary DataFrame
        df = pd.DataFrame(episode_metrics)
        stats = df.describe()
        print("\n📊 Crop Report 📊")
        print(f"🌽 Yield: {stats.loc['mean', 'final_yield_potential']:.2f} (range: {stats.loc['min', 'final_yield_potential']:.2f}–{stats.loc['max', 'final_yield_potential']:.2f})")
        print(f"💧 Water Used: {stats.loc['mean', 'total_water_used']:.2f} mm")
        print(f"🌱 Moisture: {stats.loc['mean', 'avg_soil_moisture']:.2f}% avg")
        print(f"📈 Episodes: {int(stats.loc['count', 'total_reward'])}")

        return df
    
    def plot_training_progress(self):
        """Plot training progress metrics."""
        if not self.episode_rewards:
            print("No training data to plot.")
            return

        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot rewards
        axs[0, 0].plot(self.episode_rewards)
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Total Reward")
        axs[0, 0].set_title("Episode Rewards")
        axs[0, 0].grid(True)

        # Plot moving average
        window = min(50, len(self.episode_rewards))
        if window > 0:
            moving_avg = np.convolve(
                self.episode_rewards, np.ones(window) / window, mode="valid"
            )
            axs[0, 1].plot(moving_avg)
            axs[0, 1].set_xlabel("Episode")
            axs[0, 1].set_ylabel("Average Reward")
            axs[0, 1].set_title(f"{window}-Episode Moving Average")
            axs[0, 1].grid(True)

        # Plot yields if available
        if self.episode_yields:
            axs[1, 0].plot(self.episode_yields)
            axs[1, 0].set_xlabel("Episode")
            axs[1, 0].set_ylabel("Yield Potential")
            axs[1, 0].set_title("Yield Progression")
            axs[1, 0].grid(True)

        # Plot water usage if available
        if self.episode_water_usage:
            axs[1, 1].plot(self.episode_water_usage)
            axs[1, 1].set_xlabel("Episode")
            axs[1, 1].set_ylabel("Water Used (mm)")
            axs[1, 1].set_title("Water Usage")
            axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_progress.png")
        plt.show()

    def run_single_episode(self, render=True, seed=None):
        """
        Run a single episode with the trained agent.

        Parameters:
        - render: Whether to print episode progress
        - seed: Random seed for reproducibility

        Returns:
        - Tuple of (total reward, yield, water used)
        """
        obs, _ = self.env.reset(seed=seed)
        state = self.discretize(obs)
        total_reward = 0
        done = False

        while not done:
            # Choose best action (no exploration)
            action = int(np.argmax(self.q_table[state]))

            # Execute action
            next_obs, reward, done, _, _ = self.env.step(action)
            next_state = self.discretize(next_obs)
            total_reward += reward

            if render and (done or self.env.current_day % 10 == 0):
                current_stage = self.env.get_current_growth_stage()
                stage_name = self.env.growth_stages[current_stage]["name"]
                water_applied = self.env.water_amounts[action]
                print(
                    f"Day {self.env.current_day:3d} | Stage: {stage_name:9s} | "
                    f"Action: {action} ({water_applied:2d}mm) | "
                    f"Soil Moisture: {self.env.soil_moisture:4.1f}% | "
                    f"Yield: {self.env.yield_potential:.2f}"
                )

            # Move to next state
            state = next_state

        if render:
            print(f"\nEpisode Summary:")
            print(f"Total Reward: {total_reward:.2f}")
            if hasattr(self.env, "yield_potential"):
                print(f"Final Yield: {self.env.yield_potential:.2f}")
            if hasattr(self.env, "total_water_used"):
                print(f"Total Water Used: {self.env.total_water_used:.1f} mm")

            # Optionally plot the episode results if environment supports it
            if hasattr(self.env, "plot_episode"):
                self.env.plot_episode()

        yield_val = getattr(self.env, "yield_potential", None)
        water_used = getattr(self.env, "total_water_used", None)

        return total_reward, yield_val,water_used