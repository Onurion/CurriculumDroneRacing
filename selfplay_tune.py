import os
import numpy as np
import pandas as pd
from datetime import datetime
import gym
import optuna

# Import your RL library and custom modules.
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from drone_race_selfplay import *
from test import evaluate_model


# Make sure you have the following imported or defined in your project:
# from your_module import DroneRaceSelfPlayEnv, SelfPlayWrapper, FrozenOpponentPolicy, FrozenModelUpdateCallback, evaluate_model

# -------------------------------
# Global settings and constants
# -------------------------------

total_timesteps_full = int(2e6)         # Full training timesteps (for later full training)
tuning_timesteps = int(5e5)             # Limited timesteps for hyperparameter tuning
eval_freq = 1000                      # Evaluation frequency (timesteps)
eval_episodes = 10                    # Number of episodes used in evaluation
win_rate_threshold = 0.6              # Frozen model update threshold
n_gates = 5                           # Number of gates in the environment
algorithm = "ppo"                     # Algorithm to use: "ppo" or "sac"
num_envs = 4                        # Number of parallel environments for vectorized env.
tuning_trials = 100

# -------------------------------
# Define the objective function
# -------------------------------

def objective(trial):
    """
    Objective function for hyperparameter tuning using Optuna.
    The function constructs the training/evaluation environments and trains the agent,
    then returns the mean total reward from evaluation episodes.
    """
    # Suggest hyperparameters (coefficients) for the environment.
    w_distance = trial.suggest_float("w_distance", 0.1, 1.0)
    w_distance_change = trial.suggest_float("w_distance_change", 0.5, 2.0)
    w_alignment = trial.suggest_float("w_alignment", 0.5, 2.0)
    w_inactivity = trial.suggest_float("w_inactivity", 0.9, 3.0)
    # action_coeff = trial.suggest_float("action_coeff", 0.5, 3.0)
    # w_deviation = trial.suggest_float("w_deviation", 0.1, 3.0)


    # Set up environment arguments using the hyperparameters from this trial.
    env_args = {
        "n_gates": n_gates,
        "radius": 5,
        "min_vel": 0.25,
        "action_coeff": 1.0,
        "distance_exp_decay": 2.0,
        "w_distance": w_distance,
        "w_distance_change": w_distance_change,
        "w_alignment": w_alignment,
        "w_inactivity": w_inactivity
        # "w_deviation": w_deviation,

    }

    # Create a unique folder for logging this trial (useful for Tensorboard, Monitor logs, etc.)
    date_str = datetime.now().strftime("%d%B_%H%M")
    main_folder_trial = f"tuning_radius5_updated/{date_str}_trial_{trial.number}"
    os.makedirs(main_folder_trial, exist_ok=True)

    # Log the current parameter set in a text file.
    params_file = os.path.join(main_folder_trial, "parameters.txt")
    with open(params_file, "a") as f:
        f.write(f"Trial {trial.number} parameters:\n")
        # f.write(f"  action_coeff: {action_coeff:.2f}\n")
        f.write(f"  w_distance: {w_distance:.2f}\n")
        f.write(f"  w_distance_change: {w_distance_change:.2f}\n")
        f.write(f"  w_alignment: {w_alignment:.2f}\n")
        # f.write(f"  w_deviation: {w_deviation:.2f}\n")
        f.write(f"  w_inactivity: {w_inactivity:.2f}\n\n")

    # ------------------------
    # Define environment maker functions
    # ------------------------

    def make_env():
        base_env = DroneRaceSelfPlayEnv(**env_args)
        # Create a dummy opponent policy based on a zero action in the opponent's action space.
        dummy_opponent_policy = lambda obs: np.zeros_like(base_env.action_space["drone1"].sample())
        env = SelfPlayWrapper(base_env, dummy_opponent_policy)
        monitor_file = os.path.join(main_folder_trial, "train_monitor.csv")
        env = Monitor(env, filename=monitor_file, allow_early_resets=True)
        return env

    def make_eval_env():
        base_env = DroneRaceSelfPlayEnv(**env_args)
        dummy_opponent_policy = lambda obs: np.zeros_like(base_env.action_space["drone1"].sample())
        env = SelfPlayWrapper(base_env, dummy_opponent_policy)
        monitor_file = os.path.join(main_folder_trial, "eval_monitor.csv")
        env = Monitor(env, filename=monitor_file, allow_early_resets=True)
        return env

    # ------------------------
    # Build vectorized environments
    # ------------------------
    vec_train_env = DummyVecEnv([make_env for _ in range(num_envs)])
    vec_frozen_env = DummyVecEnv([make_env for _ in range(num_envs)])
    eval_env = make_eval_env()

    # ------------------------
    # Initialize models (train and frozen opponent)
    # ------------------------
    if algorithm == "ppo":
        model = PPO("MlpPolicy", vec_train_env,
                    tensorboard_log= os.path.join(main_folder_trial, "tensorboard"),
                    device="cpu", verbose=0)
        frozen_model = PPO("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)
    elif algorithm == "sac":
        model = SAC("MlpPolicy", vec_train_env,
                    tensorboard_log= os.path.join(main_folder_trial, "tensorboard"),
                    device="cpu", verbose=0)
        frozen_model = SAC("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)
    else:
        raise ValueError("Unsupported algorithm.")

    # Copy the parameters from the training model to the frozen model.
    frozen_model.set_parameters(model.get_parameters())

    # Create the frozen opponent policy using the frozen model.
    frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)

    # Update the opponent policy for the vectorized training environments.
    for env in vec_train_env.envs:
        env.frozen_opponent_policy = frozen_opponent_policy
    eval_env.frozen_opponent_policy = frozen_opponent_policy

    # ------------------------
    # Set up callbacks
    # ------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path= os.path.join(main_folder_trial, "best_model"),
        log_path= os.path.join(main_folder_trial, "logs"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1
    )

    frozen_update_callback = FrozenModelUpdateCallback(
        eval_env=eval_env,
        frozen_model=frozen_model,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        win_rate_threshold=win_rate_threshold,
        verbose=0,
    )

    callback_list = CallbackList([eval_callback, frozen_update_callback])

    # ------------------------
    # Train the model (for tuning we use only a fraction of total timesteps)
    # ------------------------
    model.learn(total_timesteps=tuning_timesteps, callback=callback_list)

    # ------------------------
    # Evaluate the model performance after training
    # ------------------------
    mean_targets_reached, mean_speed, mean_reward = evaluate_model(model, eval_env, n_episodes=eval_episodes)

    # Report also to Tensorboard or logging if desired.
    print(f"Trial {trial.number}: mean_targets_reached = {mean_targets_reached:.2f} mean_speed = {mean_speed:.2f} mean_reward = {mean_reward:.2f} | "
          f"Parms: w_distance={w_distance:.2f}, "
          f"w_alignment={w_alignment:.2f}, "
        #   f"w_distance_change={w_distance_change:.2f}, w_deviation={w_deviation:.2f}, "
          f"w_inactivity={w_inactivity:.2f}")
        # action_coeff={action_coeff:.2f},


    # Append the evaluation result into the same parameters.txt file.
    params_file = os.path.join(main_folder_trial, "parameters.txt")
    with open(params_file, "a") as f:
        f.write("Evaluation results:\n")
        f.write(f"  Mean Reward: {mean_reward:.2f}\n")
        f.write(f"  Mean Targets Reached: {mean_targets_reached:.2f}\n")
        f.write(f"  Mean Speed: {mean_speed:.2f}\n")


    return mean_targets_reached

# -------------------------------
# Run the Hyperparameter Optimization
# -------------------------------

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=tuning_trials)

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Best hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Optionally, after tuning, you can retrain on the full dataset with the best hyperparameters:
# best_env_args = {
#     "n_gates": n_gates,
#     "radius": 10,
#     "action_coeff": trial.params["action_coeff"],
#     "distance_exp_decay": 2.0,
#     "w_distance": trial.params["w_distance"],
#     "w_distance_change": trial.params["w_distance_change"],
#     "w_deviation": trial.params["w_deviation"],
#     "w_inactivity": trial.params["w_inactivity"]
# }
# Then rebuild your environments and model using 'total_timesteps_full'.