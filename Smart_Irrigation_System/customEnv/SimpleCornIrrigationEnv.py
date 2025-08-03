import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class SimpleCornIrrigationEnv(gym.Env):
    """
    A simplified corn irrigation environment focusing on core irrigation decisions.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 season_length=120,
                 render_mode=None,
                 difficulty="normal",
                 region_type="temperate",
                 seed=None):
        """
        Initialize the simplified irrigation environment.
        
        Parameters:
        - season_length: Length of growing season in days
        - render_mode: Visualization mode ('human' or 'rgb_array')
        - difficulty: Scenario difficulty ('easy', 'normal', 'hard')
        - region_type: Climate region ('arid', 'temperate', 'tropical')
        - seed: Random seed for reproducibility
        """
        super(SimpleCornIrrigationEnv, self).__init__()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Basic environment parameters
        self.season_length = season_length
        self.render_mode = render_mode
        self.current_day = 0
        self.difficulty = difficulty
        self.region_type = region_type
        
        # Define action space: irrigation amounts
        # 0: No irrigation (0mm)
        # 1: Light irrigation (10mm)
        # 2: Medium irrigation (20mm)
        # 3: Heavy irrigation (30mm)
        self.action_space = spaces.Discrete(4)
        
        if self.region_type == "arid":
            self.water_amounts = [0, 8, 16, 24]  # Drought-stricken regions may need more water
        elif self.region_type == "tropical":
            self.water_amounts = [0, 3, 6, 12]  # Tropical areas may require less irrigation
        else:  # temperate
            self.water_amounts = [0, 5, 10, 15]  # Temperate regions to maintain the original program
        
        # Define simplified observation space (6 dimensions)
        # - Current soil moisture (0-100%)
        # - Day in season (0-season_length)
        # - Growth stage (0-4)
        # - Temperature (0-40°C)
        # - Recent rainfall (0-50mm)
        # - Next day rain probability (0-100%)
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([100, self.season_length, 4, 40, 50, 100]),
            dtype=np.float32
        )
        
        # Growth stages (simplified)
        self.growth_stages = [
            {"name": "Seedling", "days": 15, "optimal_moisture": 70, "importance": 1.0},
            {"name": "Jointing", "days": 25, "optimal_moisture": 75, "importance": 1.2},
            {"name": "Staminate", "days": 15, "optimal_moisture": 80, "importance": 1.5},
            {"name": "Filling", "days": 30, "optimal_moisture": 75, "importance": 1.3},
            {"name": "Maturity", "days": 35, "optimal_moisture": 65, "importance": 1.0}
        ]
        
        # Weather patterns (simplified)
        self.weather_patterns = self._initialize_regional_patterns()
        
        # Environment state tracking
        self.total_water_used = 0
        self.yield_potential = 1.0
        self.current_step = 0
        self.soil_moisture = 50.0  # Initial soil moisture
        
        # Simple logging
        self.water_usage_history = []
        self.rainfall_history = []
        self.soil_moisture_history = [self.soil_moisture]
        self.yield_history = [self.yield_potential]
        self.reward_history = []
        
        # Generate synthetic weather data
        self.weather_data = self._generate_season_data()
    
    def _initialize_regional_patterns(self):
        """Define simplified weather patterns for different regions"""
        patterns = {
            "arid": {
                "temp_base": 30,
                "temp_variation": 15,
                "rain_probability": 0.1,
                "rain_amount_min": 0, # mm
                "rain_amount_max": 5, # mm
            },
            "temperate": {
                "temp_base": 25,
                "temp_variation": 10,
                "rain_probability": 0.3,
                "rain_amount_min": 5, # mm
                "rain_amount_max": 15,
            },
            "tropical": {
                "temp_base": 32,
                "temp_variation": 5,
                "rain_probability": 0.6,
                "rain_amount_min": 10, 
                "rain_amount_max": 40,
            }
        }
        return patterns[self.region_type]
    
    def _generate_season_data(self):
        """Generate simplified synthetic weather data for the season"""
        data = []
        pattern = self.weather_patterns
        
        # Adjust difficulty
        if self.difficulty == "easy":
            rain_modifier = 1.2
            variation_modifier = 0.8
        elif self.difficulty == "hard":
            rain_modifier = 0.5
            variation_modifier = 1.2
        else:  # normal
            rain_modifier = 1.0
            variation_modifier = 1.0
        
        # Generate daily weather
        prev_temp = pattern["temp_base"]
        
        for day in range(self.season_length):
            # Season progress factor (0-1)
            season_progress = day / self.season_length
            
            # Temperature calculation (seasonal curve + randomness)
            temp_trend = pattern["temp_base"] * (0.8 + 0.4 * np.sin(np.pi * season_progress))
            temperature = 0.7 * prev_temp + 0.3 * temp_trend + np.random.normal(0, pattern["temp_variation"] * 0.2 * variation_modifier)
            temperature = min(max(temperature, 5), 40)  # Keep within limits
            
            # Rainfall calculation
            rain_chance = pattern["rain_probability"] * rain_modifier
            if np.random.random() < rain_chance:
                rainfall = pattern["rain_amount_min"] + np.random.exponential((pattern["rain_amount_max"] - pattern["rain_amount_min"]) / 3)
                rainfall = min(rainfall, pattern["rain_amount_max"])
            else:
                rainfall = 0
            
            # Next day rain probability
            next_day_rain_prob = rain_chance * 100 * (0.8 + 0.4 * np.random.random())
            
            # Store data
            data.append({
                'Temperature': temperature,
                'Rainfall': rainfall,
                'NextDayRainProb': next_day_rain_prob
            })
            
            prev_temp = temperature
            
        return data
    
    def get_current_growth_stage(self):
        """Get the current growth stage index (0-4)"""
        days_elapsed = 0
        for i, stage in enumerate(self.growth_stages):
            days_elapsed += stage["days"]
            if self.current_day < days_elapsed:
                return i
        return 4  # Default to final stage if beyond all stages
    
    def calculate_evapotranspiration(self, temperature, growth_stage_idx):
        """Calculate a simplified daily evapotranspiration"""
        # Base ET based on temperature
        base_et = 0.2 * temperature - 1
        
        # Crop coefficient by growth stage
        crop_coefficients = [0.3, 0.7, 1.2, 1.0, 0.5]
        kc = crop_coefficients[growth_stage_idx]
        
        daily_et = base_et * kc
        daily_et = max(1, min(15, daily_et))  # Keep within reasonable limits
        
        return daily_et
    
    def _get_observation(self):
        """Get the simplified state observation"""
        # Get current data
        weather = self.weather_data[self.current_step % self.season_length]
        
        # Get current growth stage
        growth_stage_idx = self.get_current_growth_stage()
        
        # Build observation vector
        observation = np.array([
            self.soil_moisture,           # Current soil moisture
            self.current_day,             # Current day in season
            growth_stage_idx,             # Growth stage (0-4)
            weather['Temperature'],       # Temperature
            weather['Rainfall'],          # Rainfall
            weather['NextDayRainProb']    # Rain probability for next day
        ], dtype=np.float32)
        
        return observation
    
    def calculate_reward(self, action):
        """Calculate a simplified reward"""
        # Get current weather and growth stage
        weather = self.weather_data[self.current_step % self.season_length]
        growth_stage_idx = self.get_current_growth_stage()
        growth_stage = self.growth_stages[growth_stage_idx]
        
        # Get current environmental conditions
        temperature = weather['Temperature']
        rainfall = weather['Rainfall']
        optimal_moisture = growth_stage["optimal_moisture"]
        stage_importance = growth_stage["importance"]
        
        # Calculate evapotranspiration
        daily_et = self.calculate_evapotranspiration(temperature, growth_stage_idx)
        
        water_applied = self.water_amounts[action]
        
        # Update total water used
        self.total_water_used += water_applied
        
        # Calculate water balance and update soil moisture
        effective_rainfall = rainfall * 0.8  # Not all rainfall is effective
        water_balance = water_applied + effective_rainfall - daily_et
        
        # Update soil moisture (simple model)
        new_soil_moisture = self.soil_moisture + water_balance
        new_soil_moisture = max(0, min(100, new_soil_moisture))  # Bound within 0-100%
        
        # Calculate reward components
        
        # 1. Moisture management reward
        moisture_error = abs(new_soil_moisture - optimal_moisture)
        if moisture_error < 5:
            moisture_reward = 10  # Excellent
        elif moisture_error < 10:
            moisture_reward = 5   # Good
        elif moisture_error < 15:
            moisture_reward = 0   # Acceptable
        elif moisture_error < 20:
            moisture_reward = -5  # Poor
        else:
            moisture_reward = -10 # Very poor
        
        # 2. Water conservation reward
        if rainfall > daily_et * 1.2: # Rainfall far exceeds demand
            if water_applied == 0:
                conservation_reward = 7 # No irrigation at all, bigger rewards
            else:
                conservation_reward = -7 # Unnecessary irrigation, greater penalties
        elif rainfall > daily_et: # Rainfall just enough to meet demand
            if water_applied == 0:
                conservation_reward = 5
            else:
                conservation_reward = -5
        elif rainfall > daily_et * 0.7: # Rainfall close to meeting demand
            if water_applied <= 5: # Minimum irrigation
                conservation_reward = 3  # Incentives for moderate irrigation
            else:
                conservation_reward = -2 # Small penalties for over-irrigation
        else:
            conservation_reward = 0 # Needs normal irrigation, neutral
        
        # 3. Yield impact
        old_yield = self.yield_potential
        
        # Calculate yield impact based on moisture
        if moisture_error < 7:
            # Excellent moisture management
            self.yield_potential += 0.01 * stage_importance
        elif moisture_error < 15:
            # Acceptable moisture management
            self.yield_potential += 0.003
        else:
            # Poor moisture management
            severity = (moisture_error - 15) / 10
            self.yield_potential -= 0.02 * severity * stage_importance
        
        # Bound yield potential
        self.yield_potential = max(0.1, min(2.0, self.yield_potential))
        
        # Calculate total reward 
        yield_change = self.yield_potential - old_yield
        yield_reward = 10 * yield_change
        
        total_reward = (moisture_reward + conservation_reward) * stage_importance + yield_reward + 2.0  # Base bonus
        
        # Update state
        self.soil_moisture = new_soil_moisture
        
        # Log data
        self.water_usage_history.append(water_applied)
        self.rainfall_history.append(rainfall)
        self.soil_moisture_history.append(new_soil_moisture)
        self.yield_history.append(self.yield_potential)
        self.reward_history.append(total_reward)
        
        return total_reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Set new random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset day counter
        self.current_day = 0
        
        # Choose a random starting point in the season
        self.current_step = np.random.randint(0, self.season_length // 4)
        
        # Reset tracking variables
        self.total_water_used = 0
        self.yield_potential = 1.0
        
        # Initialize soil moisture based on region
        if self.region_type == "arid":
            self.soil_moisture = np.random.uniform(30, 50)
        elif self.region_type == "tropical":
            self.soil_moisture = np.random.uniform(50, 70)
        else:
            self.soil_moisture = np.random.uniform(40, 60)
        
        # Reset logs
        self.water_usage_history = []
        self.rainfall_history = []
        self.soil_moisture_history = [self.soil_moisture]
        self.yield_history = [self.yield_potential]
        self.reward_history = []
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """Take a step in the environment"""
        # Calculate reward
        reward = self.calculate_reward(action)
        
        # Move to next day
        self.current_day += 1
        
        # Move to next weather data point
        self.current_step = (self.current_step + 1) % self.season_length
        
        # Check if season is over
        done = self.current_day >= self.season_length
        
        # Get new observation
        observation = self._get_observation()
        
        # Add final reward for season completion if done
        if done:
            # Simple final yield bonus
            final_yield_bonus = (self.yield_potential - 1.0) * 50
            reward += final_yield_bonus
            
            info = {
                "final_yield": self.yield_potential,
                "total_water_used": self.total_water_used,
                "water_efficiency": self.yield_potential / (self.total_water_used + 1)
            }
        else:
            info = {}
        
        return observation, reward, done, False, info
    
    def render(self):
        """Render the current state of the environment"""
        if self.render_mode is None:
            return
        
        if self.render_mode == "human":
            growth_stage_idx = self.get_current_growth_stage()
            stage_name = self.growth_stages[growth_stage_idx]["name"]
            
            print(f"\n--- Day {self.current_day} | {stage_name} Stage ---")
            print(f"Soil Moisture: {self.soil_moisture:.1f}%")
            weather = self.weather_data[self.current_step % self.season_length]
            print(f"Weather: Temp {weather['Temperature']:.1f}°C | Rain {weather['Rainfall']:.1f}mm")
            print(f"Yield Potential: {self.yield_potential:.2f}")
            
            if len(self.water_usage_history) > 0:
                print(f"Last Action: {self.water_usage_history[-1]}mm irrigation")
                print(f"Last Reward: {self.reward_history[-1]:.2f}")
        
        elif self.render_mode == "rgb_array":
            # Return a dummy array for visualization
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def plot_episode(self):
        """Plot the results of an episode"""
        if len(self.soil_moisture_history) <= 1:
            print("No episode data to plot.")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        days = range(len(self.soil_moisture_history))
        
        # Plot soil moisture
        ax1.plot(days, self.soil_moisture_history, 'b-', label='Soil Moisture (%)')
        
        # Highlight growth stages
        days_elapsed = 0
        for i, stage in enumerate(self.growth_stages):
            start_day = days_elapsed
            days_elapsed += stage["days"]
            end_day = min(days_elapsed, self.season_length)
            
            if start_day < len(days) and end_day > 0:
                ax1.axvspan(start_day, end_day, alpha=0.2, color=f'C{i}')
                ax1.text((start_day + end_day) / 2, 90, stage["name"], 
                         horizontalalignment='center', fontsize=9)
                
                # Plot optimal moisture line for this stage
                if start_day < len(days) and end_day < len(days):
                    optimal = stage["optimal_moisture"]
                    ax1.plot([start_day, end_day], [optimal, optimal], '--', color=f'C{i}', alpha=0.7)
        
        # Plot water applications and rainfall
        ax2.bar(days[1:], self.water_usage_history, width=0.4, align='edge', 
               label='Irrigation (mm)', alpha=0.6, color='blue')
        ax2.bar(days[1:], self.rainfall_history, width=-0.4, align='edge', 
               label='Rainfall (mm)', alpha=0.6, color='skyblue')
        ax2.set_ylabel('Water (mm)')
        ax2.legend()
        
        # Plot yield progression
        ax3.plot(days, self.yield_history, 'g-', label='Yield Potential')
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Yield Potential')
        
        # Add secondary axis for rewards
        ax3_twin = ax3.twinx()
        if self.reward_history:
            ax3_twin.plot(days[1:], self.reward_history, 'r--', alpha=0.6, label='Reward')
            ax3_twin.set_ylabel('Reward')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines3_twin, labels3_twin = ax3_twin.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc='upper right')
        ax3.legend(lines3 + lines3_twin, labels3 + labels3_twin, loc='lower right')
        
        # Add titles
        ax1.set_title('Soil Moisture Over Time')
        ax2.set_title('Water Input')
        ax3.set_title('Yield Progression and Rewards')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n--- Season Summary ---")
        print(f"Final Yield: {self.yield_potential:.2f}")
        print(f"Total Water Used: {self.total_water_used:.1f} mm")
        print(f"Total Rainfall: {sum(self.rainfall_history):.1f} mm")
        if self.total_water_used > 0:
            print(f"Water Efficiency: {self.yield_potential / (self.total_water_used / 100):.3f}")