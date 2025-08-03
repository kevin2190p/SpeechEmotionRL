from customEnv.SimpleCornIrrigationEnv import SimpleCornIrrigationEnv
from utils.ModelComparison import ModelComparison
import os
import matplotlib.pyplot as plt

def main():
    """
    Main function to run model comparison with rich text output.
    """
    os.makedirs("./comparison_results", exist_ok=True)
    
    env = SimpleCornIrrigationEnv()
    
    comparator = ModelComparison(env)
    
    loaded_models = comparator.load_models()
    
    if not loaded_models:
        print("No models are loaded.Please check the model path")
        return
    
    results = comparator.compare_all_models(n_episodes=10)
    
    comparator.plot_comparison(save_path="./comparison_results/model_comparison.png")
    comparator.plot_reward_distributions(save_path="./comparison_results/reward_distributions.png")
    
    comparator.save_results("./comparison_results/comparison_results.json")
    
    print("\nComparative completion.The results have been saved to ./comparison_results/")

if __name__ == "__main__":
    main()