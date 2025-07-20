import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import re
import json
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from visualizer.plant_visualizer import get_health_state, load_crop_image
class LearningRateScheduler(BaseCallback):
    """
    Callback for adapting the learning rate during PPO training.
    """
    
    def __init__(self, initial_lr=0.0003, min_lr=0.00001, decay_factor=0.95, update_freq=10000):
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


class PPOTrainerSimplified:
    """
    Simplified PPO Trainer focused on core training functionality.
    """
    
    def __init__(self, model_version_manager=None, log_dir="./logs/ppo"):
        """Initialize the PPO trainer with minimal components."""
        self.model_manager = model_version_manager
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.model = None
    
    def train(
        self,
        env,
        total_timesteps=100000,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
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
        Train a PPO model with simplified parameters and adaptive learning rate.
        
        Parameters:
        - env: Environment to train in
        - total_timesteps: Total timesteps to train for
        - learning_rate: Initial learning rate
        - n_steps: Number of steps to run for each environment per update
        - batch_size: Minibatch size
        - n_epochs: Number of epoch when optimizing the surrogate loss
        - gamma: Discount factor
        - gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimation
        - clip_range: Clipping parameter for PPO
        - clip_range_vf: Clipping parameter for value function (if None, same as clip_range)
        - ent_coef: Entropy coefficient for exploration
        - use_adaptive_lr: Whether to use adaptive learning rate
        - min_lr: Minimum learning rate (if adaptive)
        - lr_decay_factor: Rate of learning rate decay (if adaptive)
        - lr_update_freq: Frequency of learning rate updates (if adaptive)
        - policy: Policy network type
        - seed: Random seed
        - save_model: Whether to save the model
        - model_metadata: Additional metadata to save with the model
        - eval_freq: Evaluation frequency during training
        - continue_training: Whether to continue training from a previous model if available
        - model_path: Specific path to a model to load (if None, will try to find the latest model)
        
        Returns:
        - Trained PPO model, training time, and model path (if saved)
        """
        start_time = time.time()
        
        # Check if we should continue training from a previous model
        if continue_training:
            # Try to load a model from the specified path or find the latest model
            loaded_model = self._load_existing_model(model_path)
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
                model = self._create_new_model(env, policy, learning_rate, n_steps, batch_size, 
                                              n_epochs, gamma, gae_lambda, clip_range, 
                                              clip_range_vf, ent_coef, seed)
        else:
            # Always create a new model if continue_training is False
            print("Creating a new model (continue_training=False).")
            model = self._create_new_model(env, policy, learning_rate, n_steps, batch_size, 
                                          n_epochs, gamma, gae_lambda, clip_range, 
                                          clip_range_vf, ent_coef, seed)
        
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
        
        # Add evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"{self.log_dir}/",
            log_path=f"{self.log_dir}/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not continue_training  # Don't reset timesteps if continuing training
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model if requested
        model_path = None
        if save_model and self.model_manager is not None:
            # Prepare metadata
            if model_metadata is None:
                model_metadata = {}
                
            # Add training parameters to metadata
            model_metadata.update({
                "learning_rate": learning_rate,
                "final_lr": lr_scheduler.current_lr if use_adaptive_lr else learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "training_time": training_time,
                "total_timesteps": total_timesteps
            })
            
            # Save using model version manager
            model_path = self.model_manager.save_model(model, "ppo", model_metadata)
        elif save_model:
            # Save without version manager
            model_path = f"{self.log_dir}/final_model"
            model.save(model_path)
            print(f"Model saved to {model_path}")

        return model, training_time, model_path
    
    def _create_new_model(self, env, policy, learning_rate, n_steps, batch_size, 
                         n_epochs, gamma, gae_lambda, clip_range, 
                         clip_range_vf, ent_coef, seed):
        """Create a new PPO model with the specified parameters."""
        return PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
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
                    return PPO.load(model_path)
                else:
                    print(f"Model not found at {model_path}")
                    return None
            
            # Try to load the best model from the log directory
            best_model_path = f"{self.log_dir}/best_model.zip"
            if os.path.exists(best_model_path):
                print(f"Loading best model from {best_model_path} (Log directory)")
                return PPO.load(best_model_path)
            
            # Try to load the final model from the log directory
            final_model_path = f"{self.log_dir}/final_model.zip"
            if os.path.exists(final_model_path):
                print(f"Loading final model from {final_model_path} (Log directory)")
                return PPO.load(final_model_path)
            
            # If model manager is available, try to get the latest model
            if self.model_manager is not None:
                latest_version = self.model_manager.get_latest_version("ppo")
                if latest_version > 0:
                    latest_model_path = self.model_manager.get_model_path("ppo", latest_version)
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
                    
                    return PPO.load(latest_model_path)
            
            # No model found
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def evaluate(self, model, env, n_eval_episodes=10, deterministic=True):
        results = []
        
        for episode in range(n_eval_episodes):
            obs, _ = env.reset(seed=episode)
            done = False
            total_reward = 0
            
            while not done:
                # Predict action
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Ensure action is a scalar integer
                if isinstance(action, np.ndarray):
                    if action.ndim == 0:
                        action = int(action.item())
                    else:
                        raise ValueError(f"Expected scalar action, got array with shape {action.shape}")
                else:
                    action = int(action)
                
                # Step the environment
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
            
            # Record results
            results.append({"episode": episode, "total_reward": total_reward})
        
        # Calculate and print average reward
        avg_reward = sum(r["total_reward"] for r in results) / len(results)
        print(f"\nEvaluation over {n_eval_episodes} episodes: Average Reward = {avg_reward:.2f}")
        
        return results

    def evaluate_model(self, model, env, n_eval_episodes: int = 10):
        """
        Evaluate PPO model over episodes, log metrics, and show crop-health image grid.
        Returns a pandas DataFrame of metrics.
        """
        episode_metrics = []
        for episode in range(n_eval_episodes):
            obs, _ = env.reset(seed=episode)
            done = False
            total_reward = 0.0
            trajectory = {"soil_moisture": [], "water_used": [], "yield_potential": [], "rewards": []}
            images = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # scalar
                if isinstance(action, np.ndarray):
                    action = int(action.flatten()[0])
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                # log
                sm = getattr(env, 'soil_moisture', np.nan)
                trajectory['soil_moisture'].append(sm)
                wa = getattr(env, 'water_amounts', [])
                wstep = wa[action] if 0 <= action < len(wa) else 0
                trajectory['water_used'].append(wstep)
                yval = getattr(env, 'yield_potential', 0)
                trajectory['yield_potential'].append(yval)
                trajectory['rewards'].append(reward)
                day = getattr(env, 'current_day', 0)
                if day % 10 == 0 or done:
                    images.append({'day': day, 'yield_potential': yval})

            total_water = getattr(env, 'total_water_used', float(np.sum(trajectory['water_used'])))
            avg_sm = float(np.nanmean([v for v in trajectory['soil_moisture'] if not np.isnan(v)]))
            final_yield = getattr(env, 'yield_potential', 0)

            episode_metrics.append({
                'episode': episode,
                'total_reward': total_reward,
                'final_yield_potential': final_yield,
                'total_water_used': total_water,
                'avg_soil_moisture': avg_sm,
            })

            # Plot grid
            if images:
                cols = min(3, len(images))
                rows = (len(images) + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
                axes = np.atleast_2d(axes)
                for idx, data in enumerate(images):
                    r, c = divmod(idx, cols)
                    ax = axes[r, c]
                    img = load_crop_image(data['yield_potential'])
                    if img is not None:
                        ax.imshow(img)
                    ax.axis('off')
                    state = get_health_state(data['yield_potential']).capitalize()
                    ax.set_title(f"Day {data['day']}\n{state} (Yield: {data['yield_potential']:.2f})")
                # disable unused
                for idx in range(len(images), rows*cols):
                    r, c = divmod(idx, cols)
                    axes[r, c].axis('off')
                plt.suptitle(f"Crop Health – Episode {episode+1}")
                plt.tight_layout()
                plt.show()

        df = pd.DataFrame(episode_metrics)
        stats = df.describe()
        print("\n📊 Crop Report 📊")
        print(f"🌽 Yield: {stats.loc['mean','final_yield_potential']:.2f} (range: {stats.loc['min','final_yield_potential']:.2f}–{stats.loc['max','final_yield_potential']:.2f})")
        print(f"💧 Water Used: {stats.loc['mean','total_water_used']:.2f} mm")
        print(f"🌱 Moisture: {stats.loc['mean','avg_soil_moisture']:.2f}% avg")
        print(f"📈 Episodes: {int(stats.loc['count','total_reward'])}")
        return df
    
    def continue_training(
        self,
        model_path,
        env,
        total_timesteps=50000,
        use_adaptive_lr=True,
        **kwargs
    ):
        """
        Continue training a previously saved PPO model.
        
        Parameters:
        - model_path: Path to the saved model
        - env: Environment to train in
        - total_timesteps: Additional timesteps to train for
        - use_adaptive_lr: Whether to use adaptive learning rate
        - **kwargs: Additional arguments for model.learn()
        
        Returns:
        - Updated model, training time, and save path (if applicable)
        """
        # Load model
        model = PPO.load(model_path)
        model.set_env(env)
        
        # Create learning rate scheduler if requested
        callbacks = []
        if use_adaptive_lr:
            # Try to get current learning rate from model
            try:
                current_lr = model.learning_rate
            except AttributeError:
                current_lr = 0.0003  # Use default if not available
                
            lr_scheduler = LearningRateScheduler(
                initial_lr=current_lr,
                min_lr=current_lr * 0.1,  # 10% of current as minimum
                decay_factor=0.95,
                update_freq=10000
            )
            callbacks.append(lr_scheduler)
        
        # Add evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f"{self.log_dir}/continued/",
            log_path=f"{self.log_dir}/continued/",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Add callbacks to kwargs
        if 'callback' not in kwargs:
            kwargs['callback'] = callbacks
        
        # Continue training
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, **kwargs)
        training_time = time.time() - start_time
        
        # Save updated model
        save_path = f"{self.log_dir}/continued_model"
        model.save(save_path)
        
        print(f"Continued training completed in {training_time:.2f} seconds")
        print(f"Updated model saved to {save_path}")
        
        return model, training_time, save_path
    
    def run_single_episode(self, model, env, render=True, seed=None):
        """
        Run a single episode with the trained model and optionally render it.
        
        Parameters:
        - model: Trained PPO model
        - env: Environment to run in
        - render: Whether to print episode progress
        - seed: Random seed for reproducibility
        
        Returns:
        - Tuple of (total reward, yield, water used)
        """
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        day = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            # Ensure action is a scalar integer
            if isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action = int(action.item())
                else:
                    raise ValueError(f"Expected scalar action, got array with shape {action.shape}")
            else:
                action = int(action)
            
            # Execute action
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            day += 1
            
            # Render environment if requested
            if render and (day % 10 == 0 or day == 1 or done):
                current_stage = env.get_current_growth_stage()
                stage_name = env.growth_stages[current_stage]["name"]
                print(f"Day {day:3d} | Stage: {stage_name:9s} | Action: {action} ({env.water_amounts[action]:2d}mm) | "
                      f"Soil Moisture: {env.soil_moisture:4.1f}% | Yield: {env.yield_potential:.2f}")
        
        print(f"\nEpisode Summary:")
        print(f"Final Yield: {env.yield_potential:.2f}")
        print(f"Total Water Used: {env.total_water_used:.1f} mm")
        print(f"Total Reward: {total_reward:.2f}")
        
        # Optionally plot the episode results if environment supports it
        if render and hasattr(env, 'plot_episode'):
            env.plot_episode()
        
        return total_reward, env.yield_potential, env.total_water_used