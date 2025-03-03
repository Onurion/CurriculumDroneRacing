import os
import optuna
import numpy as np

from stable_baselines3 import PPO, SAC
from utils import *
from curriculum_train import reinitialize_envs
from test_multiple import evaluate_model
from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnNoModelImprovement
from datetime import datetime

# -------------------------------
# Global configuration and variables
# -------------------------------

num_envs = 4
algorithm = "ppo"  # or "sac"
total_timesteps = int(1e6)  # For the sake of testing; use 1e7 in production runs
eval_freq = 2000
eval_episodes = 5
max_no_improvement_evals = 10
win_rate_threshold = 0.6
n_gates = 5
radius = 10.0
action_coefficient= 1.92
distance_exp_decay= 2.00
w_distance= 0.52
w_distance_change= 1.51
w_deviation= 0.17
w_inactivity= 0.20
reward_type= 1
observation_type= 1
is_buffer_obs= False
buffer_size= 5
max_steps = 2000
algorithm = "ppo"  # Replace with your algorithm name, e.g., "ppo" or "sac".
date_str = datetime.now().strftime("%d%B_%H%M")
main_folder = f"{date_str}_automated_curriculum_2_params"
env_args = {"n_gates": n_gates, "radius":radius, "action_coefficient":action_coefficient, "distance_exp_decay":distance_exp_decay, "w_distance":w_distance,
            "w_distance_change":w_distance_change, "w_deviation":w_deviation, "w_inactivity":w_inactivity, "reward_type": reward_type, "observation_type": observation_type,
            "is_buffer_obs": is_buffer_obs, "buffer_size": buffer_size, "max_steps": max_steps}


# Global variables to hold the best model info across trials
best_model_path = None
best_score = -np.inf  # set as minus infinity to maximize properly

# For all trials
def log_trial_results(trial_number, score, mean_targets_reached, std_targets_reached,
                      mean_speed, std_speed, mean_collisions, std_collisions):
    log_line = (f"Trial {trial_number} Score: {score:.2f}, "
                f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} "
                f"Mean Speed={mean_speed:.2f}/±{std_speed:.2f} "
                f"Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}\n")

    with open(os.path.join(main_folder, "all_trials.txt"), "a") as f:
        f.write(log_line)

def objective(trial):
    """
    Optuna objective function.

    For each trial, this function:
      - Suggests hyperparameters (minimum_velocity, action_coefficient).
      - Sets up the training and evaluation environments accordingly.
      - Uses transfer learning: if a globally best model exists, it is loaded to initialize the agent.
      - Trains the model and evaluates its performance.
      - Updates the best model (globally) if the current trial outperforms previous ones.
    """
    global best_model_path, best_score

    # Suggest hyperparameters for environment difficulty
    minimum_velocity = trial.suggest_float("minimum_velocity", 1.0, 10.0)
    action_coefficient = trial.suggest_float("action_coefficient", 1.0, 10.0)
    collision_penalty= 0.50
    gate_passing_tolerance= 0.25
    takeover_reward= 0.20

    # collision_penalty = trial.suggest_float("collision_penalty", 0.0, 1.0)
    # gate_passing_tolerance = trial.suggest_float("gate_passing_tolerance", 0.1, 0.5)
    # takeover_reward = trial.suggest_float("takeover_reward", 0.0, 1.0)
    # w_distance = trial.suggest_float("w_distance", 0.1, 0.75)
    # w_distance_change = trial.suggest_float("w_distance_change", 0.1, 2.0)
    # w_deviation = trial.suggest_float("w_deviation", 0.1, 0.75)
    # w_inactivity = trial.suggest_float("w_inactivity", 0.1, 0.5)

    # Define remaining parameters (fixed in this example)
    env_params = {
        "minimum_velocity": minimum_velocity,
        "action_coefficient": action_coefficient,
        "enable_collision": True,
        "terminate_on_collision": False,
        "collision_penalty": collision_penalty,
        "gate_passing_tolerance": gate_passing_tolerance,
        "takeover_reward": takeover_reward,
        "w_distance": w_distance,
        "w_distance_change": w_distance_change,
        "w_deviation": w_deviation,
        "w_inactivity": w_inactivity
    }

    # Update environments' common arguments with new curriculum parameters
    current_env_args = dict(env_args)
    current_env_args.update(env_params)
    trial_no =  f"trial_{trial.number}"

    # Create a folder specific to the current trial for logging and saving models
    trial_folder = os.path.join(main_folder, trial_no)
    os.makedirs(trial_folder, exist_ok=True)

    # (Re)initialize the environments for training, evaluation, and frozen model setup.
    vec_train_env, vec_frozen_env, eval_env = reinitialize_envs(current_env_args, trial_folder, num_envs)

    write_env_parameters(trial_folder, current_env_args, trial=trial_no, algorithm=algorithm,
                        total_timesteps=total_timesteps, eval_freq=eval_freq,
                        eval_episodes=eval_episodes,win_rate_threshold=win_rate_threshold)

    # -------------------------------
    # Transfer Learning: Load the best model if it exists.
    # -------------------------------
    if best_model_path is not None and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for trial {trial.number}")
        if algorithm == "ppo":
            model = PPO.load(best_model_path, env=vec_train_env, device="cpu", tensorboard_log=os.path.join(trial_folder, "tensorboard"))
        elif algorithm == "sac":
            model = SAC.load(best_model_path, env=vec_train_env, device="cpu", tensorboard_log=os.path.join(trial_folder, "tensorboard"))
        model.set_env(vec_train_env)
    else:
        # If no best model is available, initialize the model from scratch.
        if algorithm == "ppo":
            model = PPO("MlpPolicy", vec_train_env,
                        tensorboard_log=os.path.join(trial_folder, "tensorboard"),
                        device="cpu", verbose=1)
        elif algorithm == "sac":
            model = SAC("MlpPolicy", vec_train_env,
                        tensorboard_log=os.path.join(trial_folder, "tensorboard"),
                        device="cpu", verbose=1)

    # Create and set up the frozen model for self-play.
    if algorithm == "ppo":
        frozen_model = PPO("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)
    elif algorithm == "sac":
        frozen_model = SAC("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)
    frozen_model.set_parameters(model.get_parameters())

    # Configure the frozen opponent policy.
    frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)
    for env in vec_train_env.envs:
        env.frozen_opponent_policy = frozen_opponent_policy
    eval_env.frozen_opponent_policy = frozen_opponent_policy

    # -------------------------------
    # Set up callbacks for evaluation and model updates.
    # -------------------------------
    improvement_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=max_no_improvement_evals)

    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(trial_folder, "best_model"),
        log_path=os.path.join(trial_folder, "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=improvement_callback
    )
    frozen_update_callback = FrozenModelUpdateCallback(
        eval_env=eval_env,
        frozen_model=frozen_model,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        win_rate_threshold=win_rate_threshold,
        verbose=1,
    )

    callback_list = CallbackList([eval_callback, frozen_update_callback])

    # -------------------------------
    # Model Training
    # -------------------------------
    model.learn(total_timesteps=total_timesteps, callback=callback_list)

    # -------------------------------
    # Evaluation: Measure performance after training.
    # -------------------------------
    # performance = evaluate_stage(eval_env)
    targets_reached_list, avg_speed_list, total_reward_list, collisions_list = evaluate_model(model, env, n_episodes=eval_episodes)
    mean_reward = np.mean(total_reward_list)
    mean_speed = np.mean(avg_speed_list)
    std_speed = np.std(avg_speed_list)
    mean_targets_reached = np.mean(targets_reached_list)
    std_targets_reached = np.std(targets_reached_list)
    mean_collisions = np.mean(collisions_list)
    std_collisions = np.std(collisions_list)

    score = (mean_targets_reached - std_targets_reached / 2.0) / 50.0 + \
            (mean_speed - std_speed / 2.0) / 10.0 - \
            (mean_collisions + std_collisions / 2.0)

    print(f"\nEvaluation Parameters: minimum_velocity {minimum_velocity:.2f} action_coefficient: {action_coefficient:.2f}")
    print(f"Results: Mean Reward={mean_reward:.2f}, Mean Speed={mean_speed:.2f}/±{std_speed:.2f}, "
          f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}")

    print(f"Trial {trial.number} finished with score: {score:.2f}")

    # Use these functions in your code
    log_trial_results(trial.number, score, mean_targets_reached, std_targets_reached,
                      mean_speed, std_speed, mean_collisions, std_collisions)

    # -------------------------------
    # Transfer Learning: Save the model if it improved the global best score.
    # -------------------------------
    if score > best_score:
        print(f"New best score achieved: {score:.2f} (old best: {best_score:.2f}). Saving new best model.")

        # Log the new best score
        log_line = (f"Trial {trial.number} Score: {score:.2f}, "
                    f"Mean Targets Reached={mean_targets_reached:.2f}/±{std_targets_reached:.2f} "
                    f"Mean Speed={mean_speed:.2f}/±{std_speed:.2f} "
                    f"Mean Collisions={mean_collisions:.2f}/±{std_collisions:.2f}\n")

        with open(os.path.join(main_folder, "best_scores.txt"), "a") as f:
            f.write(log_line)

        best_score = score
        best_model_folder = os.path.join(trial_folder, "best_model")
        os.makedirs(best_model_folder, exist_ok=True)
        best_model_path = os.path.join(best_model_folder, "best_model.zip")
        model.save(best_model_path)
    else:
        print(f"Trial {trial.number} did not beat the best score: {best_score:.2f}.")
        # Optionally, you could reload the globally best model for consistency, though here it
        # simply indicates that the current trial did not improve.

    # Return the score so that Optuna can track the performance.
    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    print("Optimization completed.")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Score: {best_trial.value:.2f}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")