import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, A2C, PPO
from agent.dyna_Q_model import DynaQAgent
from customEnv.SimpleCornIrrigationEnv import SimpleCornIrrigationEnv
from visualizer.plant_visualizer import load_growth_stage_image, get_health_state
from PIL import Image
import io


# —————— ENVIRONMENT WRAPPER ——————
class CustomEnvWrapper(gym.Wrapper):
    """Ensure obs space matches the trained models, clip & cast to float32."""
    def __init__(self, env):
        super().__init__(env)
        # Mirror the underlying env space exactly
        self.observation_space = spaces.Box(
            low=env.observation_space.low.astype(np.float32),
            high=env.observation_space.high.astype(np.float32),
            dtype=np.float32
        )
        self.action_space = env.action_space

    def _process(self, obs):
        return np.clip(obs.astype(np.float32),
                       self.observation_space.low,
                       self.observation_space.high)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._process(obs), reward, done, truncated, info

# —————— FACTORY & DATA EXTRACTOR ——————
def create_environment(season_length, difficulty, region_type, randomness):
    """randomness → seed for reproducibility (0.0–1.0 scaled)."""
    seed = int(randomness * 10000)
    env = SimpleCornIrrigationEnv(
        season_length=season_length,
        difficulty=difficulty,
        region_type=region_type,
        seed=seed
    )
    return CustomEnvWrapper(env)

def get_env_data(env):
    """
    Pull out the six‐dim observation plus metadata.
    SimpleCornIrrigationEnv._get_observation returns:
      [soil_moisture, day, stage_idx, temperature, rainfall, next_day_rain_prob]
    """
    raw = env.unwrapped
    obs = raw._get_observation()
    return {
        'Day': int(obs[1]),
        'Growth Stage': raw.growth_stages[int(obs[2])]['name'],
        'Soil Moisture (%)': obs[0],
        'Temperature (°C)': obs[3],
        'Rainfall (mm)': obs[4],
        'Next-Day Rain Prob (%)': obs[5],
        'Yield Potential': raw.yield_potential
    }

# Helper function to convert PIL Image to Streamlit-compatible format
def pil_to_streamlit(pil_img):
    if pil_img is None:
        return None
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# —————— EPISODE RUNNER ——————
def run_episode(model, env, model_type):
    data = []
    obs, _ = env.reset()
    done = False

    while not done:
        if model_type == "Dyna-Q":
            state = model.discretize(obs)
            action = model.choose_action(state)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        entry = get_env_data(env)
        entry.update({'Action': action, 'Reward': reward})
        data.append(entry)

    return pd.DataFrame(data)

# —————— MODEL LOADER ——————
def load_model(model_type, env, params=None):
    if model_type == "Dyna-Q":
        agent = DynaQAgent(env=env.unwrapped, **(params or {}))
        found = agent._load_existing_model()
        if found:
            st.sidebar.success("Dyna-Q loaded")
        else:
            st.sidebar.warning("No Dyna-Q checkpoint, using fresh agent")
        return agent
    else:
        cls = {"DQN": DQN, "A2C": A2C, "PPO": PPO}[model_type]
        path = f"./logs/{model_type.lower()}/best_model.zip"
        try:
            m = cls.load(path, env=env)
            st.sidebar.success(f"{model_type} loaded")
            return m
        except Exception as e:
            st.sidebar.error(f"Could not load {model_type}: {e}")
            return None

# —————— STREAMLIT UI ——————
def main():
    st.set_page_config("Corn Irrigation Dashboard", layout="wide")
    st.title("🌽 Corn Irrigation System Dashboard")

    # — Sidebar: environment & randomness —
    st.sidebar.header("Environment Settings")
    season = st.sidebar.slider("Season Length (days)", 30, 180, 120)
    diff   = st.sidebar.selectbox("Difficulty", ["easy","normal","hard"])
    region = st.sidebar.selectbox("Region Type", ["temperate","tropical","arid"])
    rand   = st.sidebar.slider("Randomness (seed)", 0.0, 1.0, 0.3)

    # Persist in session
    if 'env' not in st.session_state:
        st.session_state.env = None
        st.session_state.history = None

    tabs = st.tabs(["Environment Monitoring","Model Evaluation","Simulation Results"])

    # ——— Tab 1: Environment Monitoring ———
    with tabs[0]:
        st.header("Environment Monitoring")
        if st.button("Initialize / Reset"):
            st.session_state.env = create_environment(season, diff, region, rand)
            st.session_state.history = [get_env_data(st.session_state.env)]
            st.success("Environment ready!")

        if st.session_state.env:
            cols = st.columns(4)
            latest = st.session_state.history[-1]
            for col,key in zip(cols, ["Temperature (°C)","Rainfall (mm)","Soil Moisture (%)","Growth Stage"]):
                col.metric(key, latest[key])
            
            # Display growth stage image
            current_stage = latest["Growth Stage"]
            stage_img = load_growth_stage_image(current_stage)
            if stage_img:
                st.image(pil_to_streamlit(stage_img), caption=f"Current Growth Stage: {current_stage}", width=300)
            
            # Display health state based on yield potential
            health_state = get_health_state(latest["Yield Potential"]).capitalize()
            st.write(f"Plant Health: {health_state} (Yield Potential: {latest['Yield Potential']:.2f})")

            if st.button("Take Random Action"):
                a = st.session_state.env.action_space.sample()
                obs, r, done, *_ = st.session_state.env.step(a)
                entry = get_env_data(st.session_state.env)
                entry.update({'Action':a,'Reward':r})
                st.session_state.history.append(entry)
                if done:
                    st.warning("Season ended—reset to start over.")

            if len(st.session_state.history)>1:
                df = pd.DataFrame(st.session_state.history)
                fig = go.Figure()
                for col in ["Temperature (°C)","Rainfall (mm)","Soil Moisture (%)"]:
                    fig.add_trace(go.Scatter(x=df["Day"], y=df[col], name=col))
                fig.update_layout(xaxis_title="Day", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
                
                # # Add growth stage visualization
                # st.subheader("Growth Stage Progression")
                # stage_df = df[["Day", "Growth Stage"]].drop_duplicates()
                # if not stage_df.empty:
                #     stage_cols = st.columns(len(stage_df))
                #     for i, (_, row) in enumerate(stage_df.iterrows()):
                #         stage_img = load_growth_stage_image(row["Growth Stage"])
                #         if stage_img and i < len(stage_cols):
                #             stage_cols[i].image(pil_to_streamlit(stage_img), caption=f"Day {row['Day']}: {row['Growth Stage']}", width=150)

    # ——— Tab 2: Model Evaluation ———
    with tabs[1]:
        st.header("Model Evaluation")
        mt = st.selectbox("Model Type", ["DQN","A2C","PPO","Dyna-Q"])
        n_ep = st.slider("Episodes", 1, 20, 10)
        if st.button("Load & Evaluate"):
            env = create_environment(season, diff, region, rand)
            model = load_model(mt, env)
            if model:
                dfs = [ run_episode(model, env, mt) for _ in range(n_ep) ]
                stats = pd.DataFrame([{'Total Reward':df['Reward'].sum(),
                                       'Final Yield':df['Yield Potential'].iloc[-1],
                                       'Total Water':df['Action'].sum()} for df in dfs])
                st.dataframe(stats.describe().loc[['mean','std','min','max']])
                fig = px.histogram(stats, x="Total Reward", nbins=20, title="Reward Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # # Display growth stages from the first episode as an example
                # if dfs and len(dfs[0]) > 0:
                #     st.subheader("Sample Growth Stage Progression (First Episode)")
                #     sample_df = dfs[0][["Day", "Growth Stage"]].drop_duplicates()
                #     stage_cols = st.columns(min(5, len(sample_df)))
                #     for i, (_, row) in enumerate(sample_df.iterrows()):
                #         if i < 5:  # Limit to 5 stages to avoid overcrowding
                #             stage_img = load_growth_stage_image(row["Growth Stage"])
                #             if stage_img:
                #                 stage_cols[i].image(pil_to_streamlit(stage_img), caption=f"Day {row['Day']}: {row['Growth Stage']}", width=150)

    # ——— Tab 3: Simulation Results ———
    with tabs[2]:
        st.header("Simulation Results")
        models = st.multiselect("Compare Models", ["DQN","A2C","PPO","Dyna-Q"], default=["DQN"])
        nrun   = st.slider("Episodes per Model", 1, 10, 3)
        if st.button("Run Comparison"):
            allr = []
            all_dfs = []  # Store all episode dataframes
            for m in models:
                env = create_environment(season, diff, region, rand)
                mdl = load_model(m, env)
                if not mdl: continue
                for i in range(nrun):
                    df = run_episode(mdl, env, m)
                    df['Model'] = m
                    df['Episode'] = i+1
                    all_dfs.append(df)
                    allr.append({'Model':m,
                                 'Episode':i+1,
                                 'Reward':df['Reward'].sum(),
                                 'Yield':df['Yield Potential'].iloc[-1],
                                 'Water':df['Action'].sum()})
            
            cmp_df = pd.DataFrame(allr)
            st.dataframe(cmp_df.groupby('Model').agg(['mean','std']))
            fig = px.box(cmp_df, x='Model', y=['Reward','Yield','Water'],
                         title="Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
            # # Show growth stage progression for best episode of each model
            # if all_dfs:
            #     st.subheader("Growth Stage Progression (Best Episode per Model)")
            #     # Find best episode for each model based on yield
            #     best_episodes = cmp_df.loc[cmp_df.groupby('Model')['Yield'].idxmax()]
                
            #     for _, row in best_episodes.iterrows():
            #         model_name = row['Model']
            #         episode_num = row['Episode']
                    
            #         # Find the corresponding dataframe
            #         episode_df = next((df for df in all_dfs if df['Model'].iloc[0] == model_name and df['Episode'].iloc[0] == episode_num), None)
                    
            #         if episode_df is not None:
            #             st.write(f"**{model_name}** - Episode {episode_num} (Yield: {row['Yield']:.2f})")
                        
            #             # Get unique growth stages
            #             stage_df = episode_df[["Day", "Growth Stage"]].drop_duplicates()
            #             stage_cols = st.columns(min(5, len(stage_df)))
                        
            #             for i, (_, stage_row) in enumerate(stage_df.iterrows()):
            #                 if i < 5:  # Limit to 5 stages
            #                     stage_img = load_growth_stage_image(stage_row["Growth Stage"])
            #                     if stage_img:
            #                         stage_cols[i].image(pil_to_streamlit(stage_img), 
            #                                            caption=f"Day {stage_row['Day']}: {stage_row['Growth Stage']}", 
            #                                            width=120)

if __name__=="__main__":
    main()
