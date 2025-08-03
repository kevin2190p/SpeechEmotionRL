import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

class RLTrainer:
    """Basic Reinforcement Learning Trainer with shared functionality of A2C and DQN trainers"""
    
    @staticmethod
    def _format_number(value):
        """Formatting numbers: scientific notation for large values, normal format for small values"""
        if isinstance(value, (int, float)):
            if abs(value) > 1000:
                return f"{value:.2e}"
            return f"{value:.2f}"
        return str(value)
    
    @classmethod
    def _create_training_callback(cls, env, model_name="model", log_dir="./logs"):
        """Create callbacks for training progress tracking
        
        Parameters:
        - env
        - model_name
        - log_dir
        
        Returns:
        - EvalCallback instance
        """
    
        os.makedirs(f"{log_dir}/{model_name}", exist_ok=True)
        
        class CustomCallback(EvalCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_mean_reward = None

            def _on_step(self):
                super()._on_step()
                if self.n_calls % self.eval_freq == 0 and self.evaluations_results:
                    mean_reward = np.mean(self.evaluations_results[-1])
                    timesteps = self.model.num_timesteps
                    
                    # Get epsilon (if DQN model)
                    epsilon = getattr(self.model, "current_eps", None)
                    season_days = 120
                    
                    # Define the progress description
                    progress = "still learning!"
                    if self.last_mean_reward is not None:
                        if mean_reward > self.last_mean_reward:
                            progress = "getting better!"
                        elif mean_reward < self.last_mean_reward:
                            progress = "corn's looking thirsty!"
                    self.last_mean_reward = mean_reward
                   
                    print(f"\n🌽 Cornfield Update (Step {timesteps}) 🌽")
                    print(f"💧 Reward: {cls._format_number(mean_reward)} ({progress})")
                    if epsilon is not None:
                        print(f"🚜 Trying new tricks: {cls._format_number(epsilon*100)}%")
                    print(f"⏳ Season Days: {season_days}")
                return True
        
     
        eval_callback = CustomCallback(
            env,
            best_model_save_path=f"{log_dir}/{model_name}/",
            log_path=f"{log_dir}/{model_name}/",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        return eval_callback
    
    @classmethod
    def _print_training_summary(cls, total_timesteps, log_path="./logs/model/"):
    
        try:
            results = np.load(f"{log_path}evaluations.npz")
            final_mean_reward = results["results"][-1].mean()
            final_std_reward = results["results"][-1].std()
        except FileNotFoundError:
            final_mean_reward, final_std_reward = 0.0, 0.0
        
        print(f"\n🌾 Harvest Time! Training Done 🌾")
        print(f"💧 Final Reward: {cls._format_number(final_mean_reward)}")
        print(f"🎯 Variation: ±{cls._format_number(final_std_reward)}")
        print(f"⏳ Total Steps: {total_timesteps}")
        print("Time to check the crops!")
    
    @classmethod
    def evaluate_model(cls, model, env, n_eval_episodes=10, plot_results=True):
       
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
        print(f"\n💦 Irrigation Report 💦")
        print(f"🌟 Reward: {cls._format_number(mean_reward)} (solid effort!)")
        print(f"🎯 Variation: ±{cls._format_number(std_reward)}")
        
        results = []
        for episode in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            episode_data = {
                "soil_moisture": [],
                "water_used": [],
                "yield_potential": [],
                "rewards": []
            }
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                # Handling different types of action formats
                if isinstance(action, np.ndarray):
                    if action.ndim == 0:  # scalar array
                        action = int(action.item())
                    else:
                        action = int(action[0])  # multidimensional array
                else:
                    action = int(action)  # non-array object
                
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
        
                episode_data["soil_moisture"].append(env.current_soil_moisture)
                episode_data["water_used"].append(env.daily_water_usage[-1])
                episode_data["yield_potential"].append(env.yield_potential)
                episode_data["rewards"].append(reward)
            
            # Add Episodes Results
            results.append({
                "total_reward": total_reward,
                "final_yield_potential": env.yield_potential,
                "total_water_used": env.total_water_used,
                "avg_soil_moisture": np.mean(episode_data["soil_moisture"])
            })
            
            # Optional plotting of round results
            if plot_results:
                plt.figure(figsize=(10, 5))
                plt.plot(episode_data["soil_moisture"], label=f"Episode {episode + 1}", color="green")
                plt.xlabel("Day")
                plt.ylabel("Soil Moisture (%)")
                plt.title(f"🌱 Soil Moisture Journey - Episode {episode + 1}")
                plt.legend()
                plt.grid(True)
                plt.show()
        
        results_df = pd.DataFrame(results)
        stats = results_df.describe()
        print(f"\n📊 Crop Report 📊")
        print(f"🌽 Yield: {cls._format_number(stats.loc['mean', 'final_yield_potential'])} (range: {cls._format_number(stats.loc['min', 'final_yield_potential'])} to {cls._format_number(stats.loc['max', 'final_yield_potential'])})")
        print(f"💧 Water Used: {cls._format_number(stats.loc['mean', 'total_water_used'])} mm")
        print(f"🌱 Moisture: {cls._format_number(stats.loc['mean', 'avg_soil_moisture'])}% avg")
        print(f"📈 Seasons: {int(stats.loc['count', 'total_reward'])}")
        
        return results_df
    
    @staticmethod
    def plot_training_logs(log_path="./logs/", model_name=None):
        
        full_path = log_path
        if model_name:
            full_path = f"{log_path}/{model_name}/"
        
        try:
            results = np.load(f"{full_path}evaluations.npz")
            timesteps = results["timesteps"]
            rewards = results["results"]
            
            plt.figure(figsize=(10, 5))
            plt.plot(timesteps, rewards.mean(axis=1), label="Mean Reward", color="blue")
            plt.fill_between(
                timesteps,
                rewards.mean(axis=1) - rewards.std(axis=1),
                rewards.mean(axis=1) + rewards.std(axis=1),
                alpha=0.2,
                color="blue",
                label="Reward Range"
            )
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title("Irrigation Agent Progress")
            plt.legend()
            plt.grid(True)
            plt.show()
        except FileNotFoundError:
            print(f"⚠️ Oops! No training logs found at {full_path}")
            
    def compare_models(self, env, models_to_compare, include_baselines=True, n_eval_episodes=5):
        """Compare multiple reinforcement learning models on the same environment.
        
        Parameters:
        - env: Environment to test on
        - models_to_compare: List of tuples (model_type, version, display_name)
        where version can be None for latest, or dict of already loaded models
        - include_baselines: Whether to include baseline policies in the comparison
        - n_eval_episodes: Number of episodes for evaluation
        
        Returns:
        - DataFrame with comparison results
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        from stable_baselines3 import A2C, DQN, PPO  # Import common RL algorithms
        
        # Mapping of model type strings to their respective classes
        model_classes = {
            "a2c": A2C,
            "dqn": DQN,
            "ppo": PPO,
        }
        
        policies = {}
        
        # Process models to compare
        if isinstance(models_to_compare, dict):
            # If a dictionary of pre-loaded models is provided
            policies = {name: lambda obs, model=model: model.predict(obs, deterministic=True)[0] 
                    for name, model in models_to_compare.items()}
        else:
            # Load models based on type and version
            for model_type, version, display_name in models_to_compare:
                # Get the model path
                if version is None:
                    version = self.model_manager.get_latest_version(model_type)
                    
                model_path = self.model_manager.get_model_path(model_type, version)
                
                # Determine the model class and load the model
                model_class = model_classes.get(model_type.lower())
                if model_class is None:
                    print(f"Warning: Unknown model type '{model_type}'. Skipping.")
                    continue
                    
                try:
                    model = model_class.load(model_path)
                    policies[f"{display_name} v{version}"] = lambda obs, m=model: m.predict(obs, deterministic=True)[0]
                except Exception as e:
                    print(f"Error loading model {model_type} v{version}: {e}")
        
        # Add baseline policies if requested
        if include_baselines:
            policies["Random"] = lambda obs: env.action_space.sample()
            policies["Conservative"] = lambda obs: self._conservative_policy(env)
            policies["Aggressive"] = lambda obs: self._aggressive_policy(env)
        
        # Compare all policies
        return self._compare_policies(env, policies, n_eval_episodes)
            
    @staticmethod
    def _conservative_policy(env):
        """Conservative baseline policy - only water when soil moisture is very low."""
        current_stage = env.get_current_growth_stage()
        optimal_moisture = env.growth_stages[current_stage]["optimal_moisture"]
        soil_moisture = env.current_soil_moisture
        
        if soil_moisture < optimal_moisture * 0.7:
            return 2  # Medium irrigation (20mm)
        else:
            return 0  # No irrigation

    @staticmethod
    def _aggressive_policy(env):
        """Aggressive baseline policy - maintain high moisture levels."""
        current_stage = env.get_current_growth_stage()
        optimal_moisture = env.growth_stages[current_stage]["optimal_moisture"]
        soil_moisture = env.current_soil_moisture
        
        if soil_moisture < optimal_moisture - 10:
            return 3  # Heavy irrigation (30mm)
        elif soil_moisture < optimal_moisture - 5:
            return 2  # Medium irrigation (20mm)
        elif soil_moisture < optimal_moisture:
            return 1  # Light irrigation (10mm)
        else:
            return 0  # No irrigation
        
        
    def _compare_policies(self, env, policies, n_eval_episodes=5):
        """
        Helper method to compare different policies.
        
        Parameters:
        - env: Environment to test on
        - policies: Dictionary mapping policy names to policy functions
        - n_eval_episodes: Number of episodes for evaluation
        
        Returns:
        - DataFrame with comparison results
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        
        results = {}
        
        for policy_name, policy_fn in policies.items():
            policy_results = []
            
            for episode in range(n_eval_episodes):
                obs, _ = env.reset(seed=episode)  # Same seed for fair comparison
                done = False
                total_reward = 0
                
                while not done:
                    action = policy_fn(obs)
                    if isinstance(action, np.ndarray):
                        action = int(action[0])
                    obs, reward, done, _, _ = env.step(action)
                    total_reward += reward
                
                policy_results.append({
                    "policy": policy_name,
                    "episode": episode,
                    "total_reward": total_reward,
                    "yield_potential": env.yield_potential,
                    "total_water_used": env.total_water_used
                })
            
            results[policy_name] = pd.DataFrame(policy_results)
        
        # Combine all results
        all_results = pd.concat(results.values())
        
        # Create summary stats
        summary = all_results.groupby("policy").agg({
            "total_reward": ["mean", "std"],
            "yield_potential": ["mean", "min", "max"],
            "total_water_used": ["mean", "std"]
        })
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        
        for policy_name in policies.keys():
            policy_data = all_results[all_results["policy"] == policy_name]
            ax.bar(
                policy_name, 
                policy_data["total_reward"].mean(),
                yerr=policy_data["total_reward"].std(),
                capsize=10,
                alpha=0.7
            )
        
        plt.ylabel("Average Reward")
        plt.title("Irrigation Strategy Comparison")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("\n🏆 Performance Ranking:")
        for i, (policy, data) in enumerate(summary["total_reward"]["mean"].sort_values(ascending=False).items()):
            print(f"{i+1}. {policy}: {self._format_number(data)} reward, {self._format_number(summary.loc[policy, ('yield_potential', 'mean')])} yield")
        
        return summary