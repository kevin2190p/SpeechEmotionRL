import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, PPO, DQN
import pickle
from tqdm import tqdm
import json
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

class ModelComparison:
    """
    A class for comparing different reinforcement learning models for the irrigation system.
    """
    
    def __init__(self, env, log_dir="./logs", models_dir="./models"):
        """
        Initialize the model comparison tool.
        
        Parameters:
        - env: The environment instance for evaluation
        - log_dir: Directory containing training logs
        - models_dir: Directory containing saved models
        """
        self.env = env
        self.log_dir = log_dir
        self.models_dir = models_dir
        self.models = {}
        self.results = {}
        self.console = Console()
        
    def load_models(self, model_paths=None):
        """
        Load models for comparison.
        
        Parameters:
        - model_paths: Dictionary mapping model names to file paths
                      If None, will try to load best models from standard locations
        
        Returns:
        - Dictionary of loaded models
        """
        if model_paths is None:
            # Try to load best models from standard locations
            model_paths = {
                "A2C": os.path.join(self.log_dir, "a2c", "best_model.zip"),
                "PPO": os.path.join(self.log_dir, "ppo", "best_model.zip"),
                "DQN": os.path.join(self.log_dir, "dqn", "best_model.zip"),
                "Dyna-Q": os.path.join(self.log_dir, "dyna-q", "dyna_q_model.pkl")
            }
        
        # Create a table for model loading status
        table = Table(title="Model Loading Status", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Path", style="green")
        table.add_column("Status", style="yellow")
        
        # Load each model if the file exists
        for name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    if name == "A2C":
                        self.models[name] = A2C.load(path)
                        status = "[bold green]Success[/bold green]"
                    elif name == "PPO":
                        self.models[name] = PPO.load(path)
                        status = "[bold green]Success[/bold green]"
                    elif name == "DQN":
                        self.models[name] = DQN.load(path)
                        status = "[bold green]Success[/bold green]"
                    elif name == "Dyna-Q":
                        with open(path, 'rb') as f:
                            self.models[name] = pickle.load(f)
                        status = "[bold green]Success[/bold green]"
                except Exception as e:
                    status = f"[bold red]Error: {str(e)}[/bold red]"
            else:
                status = "[bold red]File not found[/bold red]"
            
            table.add_row(name, path, status)
        
        # Display the table
        self.console.print(table)
        
        return self.models
    
    def evaluate_model(self, model_name, n_episodes=10, max_steps=1000):
        """
        Evaluate a single model.
        
        Parameters:
        - model_name: Name of the model to evaluate
        - n_episodes: Number of evaluation episodes
        - max_steps: Maximum steps per episode
        
        Returns:
        - Dictionary of evaluation metrics
        """
        if model_name not in self.models:
            self.console.print(f"[bold red]Model {model_name} not loaded[/bold red]")
            return None
        
        model = self.models[model_name]
        
        # Prepare metrics
        rewards = []
        episode_lengths = []
        yields = []
        water_usage = []
        
        # Create progress display
        with self.console.status(f"[bold green]Evaluating {model_name}...[/bold green]"):
            # Run evaluation episodes
            for episode in range(n_episodes):
                self.console.print(f"Running episode {episode+1}/{n_episodes} for {model_name}")
                obs, _ = self.env.reset()
                done = False
                total_reward = 0
                step_count = 0
                
                while not done and step_count < max_steps:
                    # Get action based on model type
                    if model_name == "Dyna-Q":
                        state = self._discretize_state(obs, model)
                        action = self._get_best_action(model, state)
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    
                    # Execute action
                    next_obs, reward, done, _, _ = self.env.step(action)
                    total_reward += reward
                    obs = next_obs
                    step_count += 1
                
                # Record metrics
                rewards.append(total_reward)
                episode_lengths.append(step_count)
                
                # Record environment-specific metrics if available
                if hasattr(self.env, 'yield_potential'):
                    yields.append(self.env.yield_potential)
                if hasattr(self.env, 'total_water_used'):
                    water_usage.append(self.env.total_water_used)
        
        # Compute statistics
        results = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "rewards": rewards,
            "episode_lengths": episode_lengths
        }
        
        if yields:
            results["mean_yield"] = np.mean(yields)
            results["yields"] = yields
        
        if water_usage:
            results["mean_water_usage"] = np.mean(water_usage)
            results["water_usage"] = water_usage
            
            # Calculate water efficiency (yield per unit of water)
            if yields:
                water_efficiency = [y/w if w > 0 else 0 for y, w in zip(yields, water_usage)]
                results["mean_water_efficiency"] = np.mean(water_efficiency)
                results["water_efficiency"] = water_efficiency
        
        self.results[model_name] = results
        
        # Display results for this model
        self._display_model_results(model_name, results)
        
        return results
    
    def _display_model_results(self, model_name, results):
        """Display results for a single model using rich."""
        table = Table(title=f"{model_name} Evaluation Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add basic metrics
        table.add_row("Mean Reward", f"{results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        table.add_row("Min Reward", f"{results['min_reward']:.2f}")
        table.add_row("Max Reward", f"{results['max_reward']:.2f}")
        table.add_row("Mean Episode Length", f"{results['mean_episode_length']:.2f}")
        
        # Add environment-specific metrics if available
        if "mean_yield" in results:
            table.add_row("Mean Yield", f"{results['mean_yield']:.2f}")
        if "mean_water_usage" in results:
            table.add_row("Mean Water Usage", f"{results['mean_water_usage']:.2f} mm")
        if "mean_water_efficiency" in results:
            table.add_row("Mean Water Efficiency", f"{results['mean_water_efficiency']:.4f} yield/mm")
        
        self.console.print(table)
    
    def compare_all_models(self, n_episodes=10, max_steps=1000):
        """
        Evaluate and compare all loaded models.
        
        Parameters:
        - n_episodes: Number of evaluation episodes per model
        - max_steps: Maximum steps per episode
        
        Returns:
        - Dictionary of evaluation results for all models
        """
        for model_name in self.models.keys():
            self.evaluate_model(model_name, n_episodes, max_steps)
        
        # After all models are evaluated, display comparison table
        self.display_comparison_table()
        
        # Recommend the best model
        self.recommend_best_model()
        
        return self.results
    
    def display_comparison_table(self):
        """Display a comparison table of all evaluated models."""
        if not self.results:
            self.console.print("[bold red]No evaluation results available. Run compare_all_models first.[/bold red]")
            return
        
        # Create comparison table
        table = Table(title="Model Comparison", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        
        # Add a column for each model
        for model_name in self.results.keys():
            table.add_column(model_name, style="green")
        
        # Common metrics to display
        metrics = [
            ("Mean Reward", "mean_reward", "{:.2f}"),
            ("Reward Std", "std_reward", "{:.2f}"),
            ("Mean Episode Length", "mean_episode_length", "{:.2f}")
        ]
        
        # Add environment-specific metrics if available
        if all("mean_yield" in results for results in self.results.values()):
            metrics.append(("Mean Yield", "mean_yield", "{:.2f}"))
        
        if all("mean_water_usage" in results for results in self.results.values()):
            metrics.append(("Mean Water Usage (mm)", "mean_water_usage", "{:.2f}"))
        
        if all("mean_water_efficiency" in results for results in self.results.values()):
            metrics.append(("Water Efficiency (yield/mm)", "mean_water_efficiency", "{:.4f}"))
        
        # Add rows for each metric
        for metric_name, metric_key, format_str in metrics:
            row = [metric_name]
            for model_name, results in self.results.items():
                if metric_key in results:
                    row.append(format_str.format(results[metric_key]))
                else:
                    row.append("N/A")
            table.add_row(*row)
        
        self.console.print(table)
    
    def recommend_best_model(self):
        """Analyze results and recommend the best model."""
        if not self.results:
            self.console.print("[bold red]No evaluation results available. Run compare_all_models first.[/bold red]")
            return
        
        # Define criteria and weights for model selection
        criteria = {
            "mean_reward": {"weight": 0.4, "higher_is_better": True},
            "mean_water_efficiency": {"weight": 0.3, "higher_is_better": True},
            "mean_yield": {"weight": 0.2, "higher_is_better": True},
            "mean_water_usage": {"weight": 0.1, "higher_is_better": False}
        }
        
        # Filter criteria that are available for all models
        available_criteria = {}
        for key, value in criteria.items():
            if all(key in results for results in self.results.values()):
                available_criteria[key] = value
        
        if not available_criteria:
            self.console.print("[bold yellow]Cannot recommend best model: no common criteria available.[/bold yellow]")
            return
        
        # Normalize weights
        total_weight = sum(c["weight"] for c in available_criteria.values())
        normalized_criteria = {k: {**v, "weight": v["weight"] / total_weight} 
                              for k, v in available_criteria.items()}
        
        # Calculate scores for each model
        scores = {}
        for model_name, results in self.results.items():
            score = 0
            for criterion, config in normalized_criteria.items():
                if criterion in results:
                    # Normalize the value relative to other models
                    values = [r[criterion] for r in self.results.values() if criterion in r]
                    min_val = min(values)
                    max_val = max(values)
                    
                    if max_val == min_val:  # Avoid division by zero
                        normalized_value = 1.0
                    else:
                        if config["higher_is_better"]:
                            normalized_value = (results[criterion] - min_val) / (max_val - min_val)
                        else:
                            normalized_value = (max_val - results[criterion]) / (max_val - min_val)
                    
                    score += normalized_value * config["weight"]
            
            scores[model_name] = score
        
        # Find the best model
        best_model = max(scores.items(), key=lambda x: x[1])
        
        # Create a table to display scores
        table = Table(title="Model Scores", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Rank", style="yellow")
        
        # Sort models by score
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Add rows for each model
        for i, (model_name, score) in enumerate(sorted_models):
            rank = i + 1
            row_style = "bold" if model_name == best_model[0] else ""
            table.add_row(
                f"[{row_style}]{model_name}[/{row_style}]" if row_style else model_name,
                f"[{row_style}]{score:.4f}[/{row_style}]" if row_style else f"{score:.4f}",
                f"[{row_style}]{rank}[/{row_style}]" if row_style else f"{rank}"
            )
        
        self.console.print(table)
        
        # Display recommendation
        panel = Panel(
            Text.from_markup(
                f"[bold green]Recommendation model: {best_model[0]}[/bold green]\n\n"
                f"Score: {best_model[1]:.4f}\n\n"
                "This recommendation is based on the following weighted assessment:\n"
                + "\n".join([f"- {k} (Weight: {v['weight']:.2f})" for k, v in normalized_criteria.items()])
            ),
            title="Model Recommended",
            border_style="green",
            box=box.ROUNDED
        )
        self.console.print(panel)
    
    def plot_comparison(self, metrics=None, save_path=None):
        """
        Plot comparison of model performance.
        
        Parameters:
        - metrics: List of metrics to plot (if None, will plot standard metrics)
        - save_path: Path to save the plot (if None, will display the plot)
        """
        if not self.results:
            self.console.print("[bold red]No evaluation results available. Run compare_all_models first.[/bold red]")
            return
        
        if metrics is None:
            metrics = ["mean_reward", "mean_yield", "mean_water_usage", "mean_water_efficiency"]
        
        # Filter metrics that are available for all models
        available_metrics = []
        for metric in metrics:
            if all(metric in self.results[model] for model in self.results):
                available_metrics.append(metric)
        
        if not available_metrics:
            self.console.print("[bold red]No common metrics available for all models[/bold red]")
            return
        
        # Create a DataFrame for plotting
        data = []
        for model_name, results in self.results.items():
            for metric in available_metrics:
                data.append({
                    "Model": model_name,
                    "Metric": metric,
                    "Value": results[metric]
                })
        
        df = pd.DataFrame(data)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Create a grouped bar chart
        chart = sns.barplot(x="Metric", y="Value", hue="Model", data=df)
        
        plt.title("Model Performance Comparison", fontsize=16)
        plt.xlabel("Metric", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(title="Model", fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for p in chart.patches:
            chart.annotate(f"{p.get_height():.2f}", 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha = 'center', va = 'bottom', 
                          fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.console.print(f"[green]图表已保存至 {save_path}[/green]")
        else:
            plt.show()
    
    def plot_reward_distributions(self, save_path=None):
        """
        Plot reward distributions for all models.
        
        Parameters:
        - save_path: Path to save the plot (if None, will display the plot)
        """
        if not self.results:
            self.console.print("[bold red]No evaluation results available. Run compare_all_models first.[/bold red]")
            return
        
        plt.figure(figsize=(12, 8))
        
        for model_name, results in self.results.items():
            if "rewards" in results:
                sns.kdeplot(results["rewards"], label=model_name)
        
        plt.title("Reward Distributions", fontsize=16)
        plt.xlabel("Reward", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend(title="Model", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.console.print(f"[green]图表已保存至 {save_path}[/green]")
        else:
            plt.show()
    
    def save_results(self, filepath):
        """
        Save evaluation results to a JSON file.
        
        Parameters:
        - filepath: Path to save the results
        """
        if not self.results:
            self.console.print("[bold red]No evaluation results available. Run compare_all_models first.[/bold red]")
            return
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                    serializable_results[model_name][key] = [v.tolist() for v in value]
                else:
                    serializable_results[model_name][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        self.console.print(f"[green]结果已保存至 {filepath}[/green]")
    
    def _discretize_state(self, obs, model):
        """Helper method to discretize state for Dyna-Q model."""
        if hasattr(model, 'discretize'):
            return model['discretize'](obs)
        
        # Default discretization if model doesn't provide one
        discretized = tuple(map(int, obs * 10))
        return discretized
    
    def _get_best_action(self, model, state):
        """Helper method to get best action from Dyna-Q model."""
        q_table = model['q_table']
        
        # If state not in Q-table, return random action
        if state not in q_table:
            return np.random.randint(0, 2)  # Assuming binary action space
        
        # Get action with highest Q-value
        q_values = q_table[state]
        return max(q_values, key=q_values.get)