import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from IPython.display import clear_output

class SimpleTrainingVisualizer:
    """
    A simplified training visualization tool for reinforcement learning.
    Replaces the complex TrainingVisualization.py with cleaner implementation.
    """
    
    def __init__(self, log_dir="./logs"):
        """
        Initialize the visualizer.
        
        Parameters:
        - log_dir: Root directory for logs
        """
        self.log_dir = log_dir
        
    def find_model_logs(self, model_type=None):
        """
        Find evaluation logs for the specified model type.
        
        Parameters:
        - model_type: Model type to find logs for (e.g., "a2c", "dqn")
        
        Returns:
        - Path to the evaluation log file, or None if not found
        """
        if model_type:
            log_path = f"{self.log_dir}/{model_type}/evaluations.npz"
            return log_path if os.path.exists(log_path) else None
        else:
            # Try to find any evaluations.npz file
            for subdir in os.listdir(self.log_dir):
                log_path = f"{self.log_dir}/{subdir}/evaluations.npz"
                if os.path.exists(log_path):
                    return log_path
        return None
    
    def plot_training_progress(self, model_type=None):
        """
        Plot the training progress for a model.
        
        Parameters:
        - model_type: Type of model to visualize (e.g., "a2c", "dqn")
        """
        log_path = self.find_model_logs(model_type)
        
        if not log_path:
            print(f"No training logs found for {model_type or 'any model'}.")
            return
        
        try:
            results = np.load(log_path)
            timesteps = results["timesteps"]
            rewards_data = results["results"]
            
            if len(timesteps) < 2:
                print("Not enough data points for visualization.")
                return
                
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Training progress
            mean_rewards = rewards_data.mean(axis=1)
            std_rewards = rewards_data.std(axis=1)
            
            ax1.plot(timesteps, mean_rewards, 'b-', label="Mean Reward")
            ax1.fill_between(
                timesteps, 
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.2, 
                color='b', 
                label="Standard Deviation"
            )
            
            ax1.set_xlabel("Training Steps")
            ax1.set_ylabel("Reward")
            ax1.set_title(f"{model_type.upper() if model_type else 'Model'} Training Progress")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Reward distribution
            all_rewards = rewards_data.flatten()
            ax2.hist(all_rewards, bins=20, alpha=0.7, color='green')
            ax2.axvline(all_rewards.mean(), color='r', linestyle='--', 
                        label=f'Mean: {all_rewards.mean():.2f}')
            ax2.set_xlabel("Reward Value")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Reward Distribution")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Print summary statistics
            print(f"\nTraining Summary for {model_type.upper() if model_type else 'Model'}")
            print(f"Total training steps: {timesteps[-1]:,}")
            print(f"Initial mean reward: {mean_rewards[0]:.2f}")
            print(f"Final mean reward: {mean_rewards[-1]:.2f}")
            print(f"Improvement: {mean_rewards[-1] - mean_rewards[0]:.2f} ({(mean_rewards[-1] - mean_rewards[0]) / abs(mean_rewards[0]) * 100:.1f}%)")
            print(f"Best mean reward: {mean_rewards.max():.2f} (at step {timesteps[mean_rewards.argmax()]:,})")
            
        except Exception as e:
            print(f"Error plotting training progress: {e}")
    
    def live_progress(self, model_type=None, update_interval=5, max_updates=20):
        """
        Display live training progress.
        
        Parameters:
        - model_type: Type of model to monitor (e.g., "a2c", "dqn")
        - update_interval: Seconds between updates
        - max_updates: Maximum number of updates before stopping
        """
        update_count = 0
        
        try:
            while update_count < max_updates:
                clear_output(wait=True)
                
                # Find log file
                log_path = self.find_model_logs(model_type)
                
                if not log_path:
                    print(f"Waiting for training logs... ({update_count + 1}/{max_updates})")
                else:
                    results = np.load(log_path)
                    timesteps = results["timesteps"]
                    rewards_data = results["results"]
                    
                    if len(timesteps) < 1:
                        print(f"Waiting for evaluation data... ({update_count + 1}/{max_updates})")
                    else:
                        # Plot current progress
                        plt.figure(figsize=(10, 6))
                        mean_rewards = rewards_data.mean(axis=1)
                        std_rewards = rewards_data.std(axis=1)
                        
                        plt.plot(timesteps, mean_rewards, 'b-')
                        plt.fill_between(
                            timesteps,
                            mean_rewards - std_rewards,
                            mean_rewards + std_rewards,
                            alpha=0.2,
                            color='b'
                        )
                        
                        plt.xlabel("Training Steps")
                        plt.ylabel("Mean Reward")
                        plt.title(f"Live Training Progress ({model_type.upper() if model_type else 'Model'})")
                        plt.grid(True, alpha=0.3)
                        
                        # Add latest statistics
                        latest_mean = mean_rewards[-1]
                        latest_step = timesteps[-1]
                        plt.annotate(f"{latest_mean:.2f}", 
                                    xy=(latest_step, latest_mean),
                                    xytext=(5, 0), 
                                    textcoords="offset points")
                        
                        plt.tight_layout()
                        plt.show()
                        
                        # Print current stats
                        print(f"Current progress - Step: {latest_step:,}, Mean reward: {latest_mean:.2f}")
                        print(f"Best reward so far: {mean_rewards.max():.2f}")
                        print(f"Update {update_count + 1}/{max_updates}")
                
                update_count += 1
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("Live monitoring stopped by user.")
        except Exception as e:
            print(f"Error in live progress monitoring: {e}")
    
    def compare_models(self, model_logs):
        """
        Compare multiple model training runs.
        
        Parameters:
        - model_logs: Dictionary mapping model names to log file paths
        """
        if not model_logs:
            print("No model logs provided for comparison.")
            return
            
        plt.figure(figsize=(12, 6))
        
        for model_name, log_path in model_logs.items():
            if not os.path.exists(log_path):
                print(f"Log file not found: {log_path}")
                continue
                
            try:
                results = np.load(log_path)
                timesteps = results["timesteps"]
                rewards = results["results"].mean(axis=1)
                
                plt.plot(timesteps, rewards, label=model_name)
                
                # Add final reward annotation
                plt.annotate(f"{rewards[-1]:.2f}", 
                            xy=(timesteps[-1], rewards[-1]),
                            xytext=(5, 0), 
                            textcoords="offset points")
                
            except Exception as e:
                print(f"Error loading {model_name} logs: {e}")
        
        plt.xlabel("Training Steps")
        plt.ylabel("Mean Reward")
        plt.title("Model Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    visualizer = SimpleTrainingVisualizer()
    
    # Plot training progress for A2C
    visualizer.plot_training_progress("a2c")
    
    # Compare different models
    # visualizer.compare_models({
    #     "A2C": "./logs/a2c/evaluations.npz",
    #     "DQN": "./logs/dqn/evaluations.npz"
    # })