# ==================================
# Step 2: Custom Emotion Environment
# ==================================

import numpy as np
import gymnasium as gym


class EmotionEnv(gym.Env):
    def __init__(self, features=None, labels=None, features_path="extracted_features.npy", labels_path="extracted_labels.npy", stage_difficulty=1):
        super(EmotionEnv, self).__init__()

        if features is not None and labels is not None:
            self.X = features
            self.y = labels
            print("Using provided feature and label arrays.")
        else:
            try:
                self.X = np.load(features_path)
                self.y = np.load(labels_path)
                print("Loaded extracted features and labels from files.")
            except FileNotFoundError:
                print("Error: Feature or label file not found.")
                exit()

        self.stage_difficulty = stage_difficulty
        self.current_idx = 0
        self.max_steps = len(self.X)  # Maximum steps is the number of samples
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.X.shape[1],), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(np.unique(self.y)))
        self.emotions = {i: label for i, label in enumerate(np.unique(self.y))}  # Map numerical labels back to their string repr.

    # Reset the environment state
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_idx = 0
        self.episode_reward = 0  # Reset episode reward
        return self.X[self.current_idx], {}

    # Perform one step in the environment
    def step(self, action, confidence=1.0):
        true_label = self.y[self.current_idx]

        # Basic reward
        reward = 1.0 if action == true_label else -0.5

        # Confidence bonus (only add if correct)
        if action == true_label:
            reward += 0.2 * confidence  # scale bonus by confidence (e.g., +0.2 maximum)

        self.episode_reward += reward  # accumulate reward

        self.current_idx += 1
        done = self.current_idx >= self.max_steps
        next_state = self.X[self.current_idx] if not done else np.zeros_like(self.X[0])

        return next_state, reward, done, False, {}

    # Placeholder render method
    def render(self, mode="human"):
        pass

    # Get the target emotion for the current observation
    def get_target_emotion(self, obs):
        return self.emotions[self.y[self.current_idx - 1]]  # Get label of previous step

    # Convert action integer to emotion string
    def action_to_emotion(self, action):
        return self.emotions[action]

    # Determine if the episode is successful based on reward
    def is_successful(self):
        return self.episode_reward > 0

    # Check if a given episode reward qualifies as successful
    def is_successful_episode(self, episode_reward):
        return episode_reward > 0