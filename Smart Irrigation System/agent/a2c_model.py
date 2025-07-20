from stable_baselines3 import A2C
import numpy as np
import time
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Union, Optional

class CustomA2C(A2C):
    """Custom A2C implementation optimized for corn irrigation environment."""
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Ensure proper observation formatting for the corn irrigation environment."""
        observation = np.array(observation)
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
            
        return super().predict(observation, state, episode_start, deterministic)


class LearningRateScheduler(BaseCallback):
    """
    Callback for adapting the learning rate during training.
    
    Parameters:
    -----------
    initial_lr: float
        Initial learning rate.
    min_lr: float
        Minimum learning rate.
    decay_factor: float
        Rate at which learning rate decays.
    update_freq: int
        Frequency (in timesteps) to update learning rate.
    """
    
    def __init__(self, initial_lr=0.0007, min_lr=0.00001, decay_factor=0.95, update_freq=10000):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.update_freq = update_freq
        self.current_lr = initial_lr
    
    def _on_step(self) -> bool:
        # Check if it's time to update the learning rate
        if self.num_timesteps % self.update_freq == 0 and self.num_timesteps > 0:
            # Calculate new learning rate with decay
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
            
            # Update the learning rate in the optimizer
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = self.current_lr
                
            # Optional: log the learning rate change
            print(f"Learning rate adjusted to {self.current_lr:.6f} at step {self.num_timesteps}")
        
        return True


class BasicTrainingProgressCallback(BaseCallback):
    """
    Simple callback for tracking and visualizing training progress.
    
    Parameters:
    -----------
    eval_env: gym.Env
        Environment used for evaluation.
    eval_freq: int
        Frequency (in timesteps) of evaluation.
    log_dir: str
        Directory to save logs.
    verbose: int
        Verbosity level.
    """
    
    def __init__(self, eval_env, eval_freq=5000, log_dir="./logs", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.evaluations = []
        self.timesteps = []
        self.rewards = []
        self.best_mean_reward = -float('inf')
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current model
            episode_rewards = []
            for _ in range(5):  # 5 evaluation episodes
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = int(action[0])
                    obs, reward, done, _, _ = self.eval_env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            mean_reward = np.mean(episode_rewards)
            self.rewards.append(episode_rewards)
            self.timesteps.append(self.num_timesteps)
            
            # Display progress
            print(f"\n----- Step {self.num_timesteps} -----")
            print(f"Mean reward: {mean_reward:.2f}")
            print(f"Episodes: 5")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"{self.log_dir}/best_model")
                print(f"New best model saved with reward: {mean_reward:.2f}")
            
            # Save evaluation results
            np.savez(
                f"{self.log_dir}/evaluations.npz",
                timesteps=np.array(self.timesteps),
                results=np.array(self.rewards)
            )
            
            # Simple plot of progress
            if len(self.timesteps) > 1:
                plt.figure(figsize=(10, 5))
                mean_rewards = [np.mean(r) for r in self.rewards]
                plt.plot(self.timesteps, mean_rewards)
                plt.xlabel("Timesteps")
                plt.ylabel("Mean Reward")
                plt.title("Training Progress")
                plt.savefig(f"{self.log_dir}/progress.png")
                plt.close()
        
        return True


class A2CTrainer:
    """Optimized A2C Trainer with simplified methods and adaptive learning rate."""
    
    def __init__(self, model_version_manager=None, log_dir="./logs/a2c"):
        """Initialize the A2C trainer with minimal components.
        
        Parameters:
        - model_version_manager: Optional ModelVersionManager instance for version control
        - log_dir: Directory for saving logs and models
        """
        # Initialize model version manager if provided
        self.model_manager = model_version_manager
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.model = None
    
    def train(
        self,
        env,
        total_timesteps=100000,
        learning_rate=0.0007,
        gamma=0.99,
        n_steps=8,
        ent_coef=0.01,
        vf_coef=0.5,
        use_adaptive_lr=True,
        min_lr=0.00001,
        lr_decay_factor=0.95,
        lr_update_freq=10000,
        policy="MlpPolicy",
        seed=42,
        save_model=True,
        model_metadata=None,
        eval_freq=5000,
        continue_training=True,
        model_path=None
    ):
        """
        Train an A2C model for the corn irrigation environment with adaptive learning rate.

        Parameters:
        - env: The environment instance
        - total_timesteps: Total timesteps to train for
        - learning_rate: Initial learning rate
        - gamma: Discount factor
        - n_steps: Number of steps for n-step return
        - ent_coef: Entropy coefficient
        - vf_coef: Value function coefficient
        - use_adaptive_lr: Whether to use adaptive learning rate
        - min_lr: Minimum learning rate (if adaptive)
        - lr_decay_factor: Rate of learning rate decay (if adaptive)
        - lr_update_freq: Frequency of learning rate updates (if adaptive)
        - policy: Policy network type
        - seed: Random seed
        - save_model: Whether to save the model using ModelVersionManager
        - model_metadata: Additional metadata to save with the model
        - eval_freq: Frequency of evaluation during training
        - continue_training: Whether to continue training from a previous model if available
        - model_path: Specific path to a model to load (if None, will try to find the latest model)

        Returns:
        - Trained A2C model, training time, and model path (if saved)
        """
        start_time = time.time()
        
        # Check if we should continue training from a previous model
        if continue_training:
            # Try to load a model from the specified path or find the latest model
            loaded_model = self._load_existing_model(model_path)
            # In the train method, after loading the model, add this code
            if loaded_model is not None:
                print("\n===== Model Training Summary =====")
                print(f"Continuing training from existing model")
                print(f"Model path: {model_path if model_path else 'Auto-detected'}")
                print(f"Additional training steps: {total_timesteps}")
                print(f"Learning rate: {learning_rate} (Adaptive: {use_adaptive_lr})")
                print("=======================\n")
                model = loaded_model
                # Set the model's environment to the current environment
                model.set_env(env)
            else:
                # Initialize a new model if no existing model is found
                print("No existing model found. Creating a new model.")
                model = self._create_new_model(env, policy, learning_rate, gamma, n_steps, ent_coef, vf_coef, seed)
        else:
            # Always create a new model if continue_training is False
            print("Creating a new model (continue_training=False).")
            model = self._create_new_model(env, policy, learning_rate, gamma, n_steps, ent_coef, vf_coef, seed)
        
        # Store the model for later use
        self.model = model
        
        # Create callbacks
        callbacks = []
        
        # Add learning rate scheduler if enabled
        if use_adaptive_lr:
            lr_scheduler = LearningRateScheduler(
                initial_lr=learning_rate,
                min_lr=min_lr,
                decay_factor=lr_decay_factor,
                update_freq=lr_update_freq
            )
            callbacks.append(lr_scheduler)
        
        # Add progress tracking callback
        progress_callback = BasicTrainingProgressCallback(
            eval_env=env,
            eval_freq=eval_freq,
            log_dir=self.log_dir
        )
        callbacks.append(progress_callback)
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not continue_training  # Don't reset timesteps if continuing training
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model if requested and if model manager is available
        model_path = None
        if save_model and self.model_manager is not None:
            # Prepare metadata
            if model_metadata is None:
                model_metadata = {}
                
            # Add training parameters to metadata
            model_metadata.update({
                "learning_rate": learning_rate,
                "final_lr": lr_scheduler.current_lr if use_adaptive_lr else learning_rate,
                "gamma": gamma,
                "n_steps": n_steps,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "seed": seed,
                "training_time": training_time,
                "total_timesteps": total_timesteps,
                "best_reward": progress_callback.best_mean_reward
            })
            
            # Save using model version manager
            model_path = self.model_manager.save_model(model, "a2c", model_metadata)
        elif save_model:
            # Save without version manager
            model_path = f"{self.log_dir}/final_model"
            model.save(model_path)
            print(f"Model saved to {model_path}")

        return model, training_time, model_path
    
    def _create_new_model(self, env, policy, learning_rate, gamma, n_steps, ent_coef, vf_coef, seed):
        """Create a new A2C model with the specified parameters."""
        return CustomA2C(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            verbose=0,
            seed=seed
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
                    return CustomA2C.load(model_path)
                else:
                    print(f"Model not found at {model_path}")
                    return None
            
            # Try to load the best model from the log directory
            best_model_path = f"{self.log_dir}/best_model.zip"
            if os.path.exists(best_model_path):
                print(f"Loading best model from {best_model_path} (Log directory)")
                return CustomA2C.load(best_model_path)
            
            # Try to load the final model from the log directory
            final_model_path = f"{self.log_dir}/final_model.zip"
            if os.path.exists(final_model_path):
                print(f"Loading final model from {final_model_path} (Log directory)")
                return CustomA2C.load(final_model_path)
            
            # If model manager is available, try to get the latest model
            if self.model_manager is not None:
                latest_version = self.model_manager.get_latest_version("a2c")
                if latest_version > 0:
                    latest_model_path = self.model_manager.get_model_path("a2c", latest_version)
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
                                best_reward = metadata.get('best_reward', 'Unknown')
                                print(f"Model Info - Created: {timestamp}, Best Reward: {best_reward}")
                    except Exception as e:
                        print(f"Error loading metadata: {e}")
                    
                    return CustomA2C.load(latest_model_path)
            
            # No model found
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def evaluate(self, model, env, n_eval_episodes=10, deterministic=True):
        """
        Simplified evaluation method for an A2C model.
        
        Parameters:
        - model: The trained model
        - env: Environment to evaluate in
        - n_eval_episodes: Number of episodes to evaluate
        - deterministic: Whether to use deterministic actions
        
        Returns:
        - DataFrame with evaluation results
        """
        results = []
        
        for episode in range(n_eval_episodes):
            obs, _ = env.reset(seed=episode)  # Different seed per episode
            done = False
            total_reward = 0
            total_water = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                if isinstance(action, np.ndarray):
                    action = int(action[0])
                    
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                
                # Track water usage if available
                water_applied = getattr(env, 'daily_water_usage', [0])[-1]
                total_water += water_applied
            
            # Record episode results
            results.append({
                "episode": episode,
                "total_reward": total_reward,
                "final_yield": getattr(env, 'yield_potential', 0),
                "total_water_used": total_water,
                "water_efficiency": getattr(env, 'yield_potential', 0) / (total_water + 1e-8)
            })
        
        # Convert to DataFrame and print summary
        results_df = pd.DataFrame(results)
        print("\n===== Evaluation Results =====")
        print(f"Episodes: {n_eval_episodes}")
        print(f"Mean Reward: {results_df['total_reward'].mean():.2f} ± {results_df['total_reward'].std():.2f}")
        print(f"Mean Yield: {results_df['final_yield'].mean():.2f}")
        print(f"Mean Water Used: {results_df['total_water_used'].mean():.2f}")
        print(f"Mean Water Efficiency: {results_df['water_efficiency'].mean():.4f}")
        
        # Simple visualization of results
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_eval_episodes), results_df['total_reward'])
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Evaluation Results")
        plt.savefig(f"{self.log_dir}/evaluation_results.png")
        plt.close()
        
        return results_df
    
    def run_single_episode(self, env, render=True, seed=None):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(np.squeeze(action))  # Convert action to scalar
            
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            if render:
                env.render()
        
        print(f"\nEpisode Summary:")
        print(f"Final Yield: {env.yield_potential:.2f}")
        print(f"Total Water Used: {env.total_water_used:.1f} mm")
        print(f"Total Reward: {total_reward:.2f}")
    
        if render:
            env.plot_episode()
        
        return total_reward, env.yield_potential, env.total_water_used
    
    def plot_simple_training_progress(self):
        """
        Simple method to plot training progress from saved logs.
        """
        try:
            results = np.load(f"{self.log_dir}/evaluations.npz")
            timesteps = results["timesteps"]
            rewards = results["results"]
            
            plt.figure(figsize=(10, 6))
            mean_rewards = np.mean(rewards, axis=1)
            std_rewards = np.std(rewards, axis=1)
            
            plt.plot(timesteps, mean_rewards, label="Mean Reward")
            plt.fill_between(
                timesteps,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.2,
                label="Standard Deviation"
            )
            
            plt.xlabel("Training Steps")
            plt.ylabel("Reward")
            plt.title("A2C Training Progress")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{self.log_dir}/final_training_progress.png")
            plt.show()
            
        except FileNotFoundError:
            print("No evaluation logs found. Please train the model first.")