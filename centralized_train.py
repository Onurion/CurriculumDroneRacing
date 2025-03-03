import os
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from drone_race_centralized import DroneRaceCentralizedEnv 

# =============================================================================
# Define Evaluation Parameters for Centralized Training
# =============================================================================
total_timesteps = int(1e7)        # Total number of training timesteps.
eval_freq = 5000                  # Evaluation frequency (in timesteps).
eval_episodes = 10                # Number of episodes to use in each evaluation.
n_gates = 5                       # Number of gates in the environment.
algorithm = "ppo"                 # Choice of algorithm: "ppo" or "sac".
date_str = datetime.now().strftime("%d%B_%H%M")
main_folder = f"{date_str}_centralized_2drones_gate_{n_gates}_{algorithm}"

# =============================================================================
# Environment Creation Functions for Centralized Training
# =============================================================================

def make_env():
    # Import your centralized environment.
    # Ensure that DroneRaceCentralizedEnv returns one joint observation,
    # a scalar reward (e.g., combined per-drone rewards), and a single done flag.
    base_env = DroneRaceCentralizedEnv(n_gates=n_gates)
    monitor_file = os.path.join(main_folder, "train_monitor.csv")
    env = Monitor(base_env, filename=monitor_file, allow_early_resets=True)
    return env

def make_eval_env():
    base_env = DroneRaceCentralizedEnv(n_gates=n_gates)
    monitor_file = os.path.join(main_folder, "eval_monitor.csv")
    env = Monitor(base_env, filename=monitor_file, allow_early_resets=True)
    return env

num_envs = 4  # Number of parallel environments.

# Create vectorized environments for training and evaluation.
vec_train_env = DummyVecEnv([make_env for _ in range(num_envs)])
vec_eval_env = DummyVecEnv([make_eval_env for _ in range(num_envs)])

# =============================================================================
# Initialize the Training Model
# =============================================================================
if algorithm == "ppo":
    model = PPO("MlpPolicy",
                vec_train_env,
                tensorboard_log=os.path.join(main_folder, "tensorboard/"),
                device="cpu",
                verbose=1)
elif algorithm == "sac":
    model = SAC("MlpPolicy",
                vec_train_env,
                tensorboard_log=os.path.join(main_folder, "tensorboard/"),
                device="cpu",
                verbose=1)

# Create a standalone evaluation environment.
eval_env = make_eval_env()

# =============================================================================
# Create Callbacks:
#
# 1. EvalCallback: Built into SB3 to save the best model based on the scalar reward.
#
# (Note: The custom frozen model/self-play callback from your previous file is removed,
#  because in a centralized setting the agent controls both drones.)
# =============================================================================
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(main_folder, "best_model/"),
    log_path=os.path.join(main_folder, "logs"),
    eval_freq=eval_freq,
    n_eval_episodes=eval_episodes,
    deterministic=True,
    render=False,
)

callback_list = CallbackList([eval_callback])

# =============================================================================
# Start Training
# =============================================================================
model.learn(total_timesteps=total_timesteps, callback=callback_list)