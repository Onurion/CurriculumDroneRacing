import os
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from envs.drone_race_centralized_v8 import DroneRaceCentralizedMultiEnv 
from utils import *


# =============================================================================
# Define Evaluation Parameters for Centralized Training
# =============================================================================
total_timesteps = 2e7              # Total number of timesteps for training.
eval_freq = 4000                  # Evaluation frequency (in timesteps).
eval_episodes = 10                # Number of episodes to use in each evaluation.
win_rate_threshold = 0.6          # Win rate to trigger frozen model update.
n_gates = 5                       # Number of gates in the environment.
n_agents = 2
radius = 10.0
action_coefficient= 1.92
distance_exp_decay= 2.00
w_distance= 0.52
w_distance_change= 1.51
w_deviation= 0.17
w_inactivity= 0.20
w_collision_penalty = 0.2
reward_type= 1
observation_type= 1
is_buffer_obs= False
buffer_size= 5
algorithm = "ppo"  # Replace with your algorithm name, e.g., "ppo" or "sac".
max_steps = 2000
env_class =  DroneRaceCentralizedMultiEnv
date_str = datetime.now().strftime("%d%B_%H%M")
folder = f"{date_str}_centralized_{n_agents}drones_v3"
env_args = {"n_agents": n_agents, "n_gates": n_gates, "radius":radius, "action_coefficient":action_coefficient, "distance_exp_decay":distance_exp_decay, "w_distance":w_distance,
            "w_distance_change":w_distance_change, "w_deviation":w_deviation, "w_inactivity":w_inactivity, "w_collision_penalty":w_collision_penalty , "reward_type": reward_type, 
            "observation_type": observation_type, "is_buffer_obs": is_buffer_obs, "buffer_size": buffer_size, "max_steps": max_steps}

num_envs = 4  # Number of parallel environments
level = 0
N_iteration = 3

curriculum_stages = [
    {
        "name": "Stage 1 - Basics",
        "timesteps": 2e7,  # Number of timesteps for this stage.
        "env_params": {
            "minimum_velocity": 1.0,     # Low required speed.
            "action_coefficient": 2.0,
            "enable_collision": False,  # Collisions are not penalized at first.
            "terminate_on_collision": False,
            "collision_penalty": 0.0,
            "gate_passing_tolerance": 0.5,
            "takeover_reward": 0.0,
        },
    },
    {
        "name": "Stage 2 - Intermediate",
        "timesteps": 2e7,  # Number of timesteps for this stage.
        "env_params": {
            "minimum_velocity": 3.0,     # Increase minimum speed.
            "action_coefficient": 3.0,
            "enable_collision": True,  # Begin penalizing collisions.
            "terminate_on_collision": False,
            "collision_penalty": 0.25,
            "gate_passing_tolerance": 0.3,
            "takeover_reward": 0.1,
        },
    },
    {
        "name": "Stage 3 - Advanced",
        "timesteps": 2e7,  # Number of timesteps for this stage.
        "env_params": {
            "minimum_velocity": 5.0,     # Further increase minimum speed.
            "action_coefficient": 4.0,
            "enable_collision": True,
            "terminate_on_collision": False,    # Now terminate episode on collisions.
            "collision_penalty": 0.5,
            "gate_passing_tolerance": 0.25,
            "takeover_reward": 0.2,
        },
    },
    {
        "name": "Stage 4 - Advanced II",
        "timesteps": 2e7,  # Number of timesteps for this stage.
        "env_params": {
            "minimum_velocity": 7.0,     # Further increase minimum speed.
            "action_coefficient": 6.0,
            "enable_collision": True,
            "terminate_on_collision": False,    # Now terminate episode on collisions.
            "collision_penalty": 0.6,
            "gate_passing_tolerance": 0.2,
            "takeover_reward": 0.2,
        },
    },
    {
        "name": "Stage 5 - Advanced III",
        "timesteps": 2e7,  # Number of timesteps for this stage.
        "env_params": {
            "minimum_velocity": 10.0,     # Further increase minimum speed.
            "action_coefficient": 7.5,
            "enable_collision": True,
            "terminate_on_collision": False,    # Now terminate episode on collisions.
            "collision_penalty": 0.7,
            "gate_passing_tolerance": 0.2,
            "takeover_reward": 0.2,
        },
    },
]

# =============================================================================
# Environment Creation Functions for Centralized Training
# =============================================================================

def reinitialize_envs(env_args, folder, num_envs):

    def make_env():
        env = env_class(**env_args)
        monitor_file = os.path.join(folder, "train_monitor.csv")
        env = Monitor(env, filename=monitor_file, allow_early_resets=True)
        return env

    vec_train_env = DummyVecEnv([make_env for _ in range(num_envs)])

    def make_eval_env():
        env_args["random_init"] = False
        env = env_class(**env_args)
        monitor_file = os.path.join(folder, "eval_monitor.csv")
        env = Monitor(env, filename=monitor_file, allow_early_resets=True)
        return env

    eval_env = make_eval_env()

    return vec_train_env, eval_env

def train(main_folder):
    stage = curriculum_stages[level]  # Stages 4 and 5, assuming 0-indexed stages.
    stage_number = level + 1  # stage 4 for j = 0, stage 5 for j = 1
    main_folder_stage = main_folder + f"_stage_{stage_number}"
    os.makedirs(main_folder_stage, exist_ok=True)
    stage_name = stage.get("name", f"stage_{stage_number}")
    new_params = stage.get("env_params", {})
    # Get the timesteps for this stage, or default if not specified.
    stage_timesteps = stage.get("timesteps", total_timesteps)

    current_env_args = dict(env_args)  # Make a copy of your initial parameters.

    print(f"\n=== Starting Centralized Training for Stage: {stage_name} ===")

    # Update the env_args dictionary with new curriculum parameters.
    current_env_args.update(new_params)

    print ("Current env args: ", current_env_args)
    vec_train_env, eval_env = reinitialize_envs(current_env_args, main_folder_stage, num_envs)

    # Create vectorized environments for training and frozen model.

    write_env_parameters(main_folder_stage, current_env_args, stage=stage_name, algorithm=algorithm,
                        total_timesteps=stage_timesteps, eval_freq=eval_freq,
                        eval_episodes=eval_episodes,win_rate_threshold=win_rate_threshold)

    # =============================================================================
    # Initialize the Training Model
    # =============================================================================
    if algorithm == "ppo":
        model = PPO("MlpPolicy",
                    vec_train_env,
                    tensorboard_log=os.path.join(main_folder_stage, "tensorboard/"),
                    device="cpu",
                    verbose=1)
    elif algorithm == "sac":
        model = SAC("MlpPolicy",
                    vec_train_env,
                    tensorboard_log=os.path.join(main_folder_stage, "tensorboard/"),
                    device="cpu",
                    verbose=1)

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
        best_model_save_path=os.path.join(main_folder_stage, "best_model/"),
        log_path=os.path.join(main_folder_stage, "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    callback_list = CallbackList([eval_callback])

    # =============================================================================
    # Start Training
    # =============================================================================
    model.learn(total_timesteps=stage_timesteps, callback=callback_list)


if __name__ == "__main__":

    for i in range(N_iteration):
        main_folder = f"{folder}_iter_{i}"
        train(main_folder=main_folder)