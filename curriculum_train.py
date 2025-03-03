from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_race_selfplay import *
from drone_race_curriculum import *
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import numpy as np


# =============================================================================
# Define Evaluation and Frozen Model Update Parameters
# =============================================================================
total_timesteps = 1e6         # Total number of training timesteps.
eval_freq = 4000                  # Evaluation frequency (in timesteps).
eval_episodes = 10                # Number of episodes to use in each evaluation.
win_rate_threshold = 0.6          # Win rate to trigger frozen model update.
n_gates = 5                       # Number of gates in the environment.
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
algorithm = "ppo"  # Replace with your algorithm name, e.g., "ppo" or "sac".
date_str = datetime.now().strftime("%d%B_%H%M")
main_folder = f"{date_str}_curriculum_2drones_v3"
env_args = {"n_gates": n_gates, "radius":radius, "action_coefficient":action_coefficient, "distance_exp_decay":distance_exp_decay, "w_distance":w_distance,
            "w_distance_change":w_distance_change, "w_deviation":w_deviation, "w_inactivity":w_inactivity, "reward_type": reward_type, "observation_type": observation_type,
            "is_buffer_obs": is_buffer_obs, "buffer_size": buffer_size}

num_envs = 4  # Number of parallel environments



curriculum_stages = [
    {
        "name": "Stage 1 - Basics",
        "timesteps": 1e6,  # Number of timesteps for this stage.
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
        "timesteps": 5e6,  # Number of timesteps for this stage.
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
        "timesteps": 1e7,  # Number of timesteps for this stage.
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
        "timesteps": 5e7,  # Number of timesteps for this stage.
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


def reinitialize_envs(env_args, main_folder, num_envs):

    def make_env():
        base_env = DroneRaceCurriculumEnv(**env_args)
        dummy_opponent_policy = lambda obs: np.zeros_like(base_env.action_space["drone1"].sample())
        env = SelfPlayWrapper(base_env, dummy_opponent_policy)
        monitor_file = os.path.join(main_folder, "train_monitor.csv")
        env = Monitor(env, filename=monitor_file, allow_early_resets=True)
        return env

    vec_train_env = DummyVecEnv([make_env for _ in range(num_envs)])
    vec_frozen_env = DummyVecEnv([make_env for _ in range(num_envs)])

    def make_eval_env():
        env_args["random_init"] = False
        base_env = DroneRaceCurriculumEnv(**env_args)
        dummy_opponent_policy = lambda obs: np.zeros_like(base_env.action_space["drone1"].sample())
        env = SelfPlayWrapper(base_env, dummy_opponent_policy)
        monitor_file = os.path.join(main_folder, "eval_monitor.csv")
        env = Monitor(env, filename=monitor_file, allow_early_resets=True)
        return env

    eval_env = make_eval_env()

    return vec_train_env, vec_frozen_env, eval_env


def train():
    level_start = 0
    previous_main_folder = None #"Results_21Feb_2025/21February_1412_curriculum_2drones_v3_stage_3"

    new_stages = curriculum_stages[level_start:]  # Stages 4 and 5, assuming 0-indexed stages.
    for i, stage in enumerate(new_stages):
        stage_number = i + level_start + 1  # stage 4 for j = 0, stage 5 for j = 1
        main_folder_stage = main_folder + f"_stage_{stage_number}"
        os.makedirs(main_folder_stage, exist_ok=True)
        stage_name = stage.get("name", f"stage_{stage_number}")
        new_params = stage.get("env_params", {})
        # Get the timesteps for this stage, or default if not specified.
        stage_timesteps = stage.get("timesteps", total_timesteps)

        current_env_args = dict(env_args)  # Make a copy of your initial parameters.

        print(f"\n=== Starting Curriculum Stage: {stage_name} ===")

        # Update the env_args dictionary with new curriculum parameters.
        current_env_args.update(new_params)

        print ("Current env args: ", current_env_args)
        # Create vectorized environments for training and frozen model.
        vec_train_env, vec_frozen_env, eval_env = reinitialize_envs(current_env_args, main_folder_stage, num_envs)

        write_env_parameters(main_folder_stage, current_env_args, stage=stage_name, algorithm=algorithm,
                            total_timesteps=total_timesteps, eval_freq=eval_freq,
                            eval_episodes=eval_episodes,win_rate_threshold=win_rate_threshold)

        # Initialize the training model using the vectorized training environment.
        # Initialize the frozen model using the vectorized frozen environment and copy initial parameters.
        if algorithm == "ppo":
            model = PPO("MlpPolicy", vec_train_env, tensorboard_log= main_folder_stage + "/tensorboard/", device="cpu", verbose=1)
            frozen_model = PPO("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)
        elif algorithm == "sac":
            model = SAC("MlpPolicy", vec_train_env, tensorboard_log= main_folder_stage + "/tensorboard/", device="cpu", verbose=1)
            frozen_model = SAC("MlpPolicy", vec_frozen_env, device="cpu", verbose=0)


        frozen_model.set_parameters(model.get_parameters())

        # Create the frozen opponent policy using the frozen model.
        frozen_opponent_policy = FrozenOpponentPolicy(frozen_model)

        # Update each sub-environment in the vectorized training env to use the frozen opponent policy.
        for env in vec_train_env.envs:
            env.frozen_opponent_policy = frozen_opponent_policy

        # Create a separate single-instance evaluation environment.
        eval_env.frozen_opponent_policy = frozen_opponent_policy  # Ensure it uses the same frozen policy.

        # Transfer Learning: if not the first stage, try to load the best model from the previous stage.
        if previous_main_folder is not None:
            best_model_path = os.path.join(previous_main_folder, "best_model", "best_model.zip")
            if os.path.exists(best_model_path):
                print(f"Loading best model from previous stage: {best_model_path}")
                model = model.load(best_model_path, env=vec_train_env, tensorboard_log= main_folder_stage + "/tensorboard/", device='cpu')
                model.set_env(vec_train_env)
            else:
                print("No best model found from the previous stage; proceeding with current model weights.")


        # =============================================================================
        # Create Callbacks:
        # 1. EvalCallback (built-in) for saving the best model based on reward.
        # 2. FrozenModelUpdateCallback (custom) for updating the frozen opponent.
        # =============================================================================

        # Built-in EvalCallback for saving the best model during training.
        eval_callback = CustomEvalCallback(
            eval_env,
            best_model_save_path= main_folder_stage + "/best_model/",
            log_path= main_folder_stage + "/logs",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

        # Custom callback for self-play frozen model update based on win rate.
        frozen_update_callback = FrozenModelUpdateCallback(
            eval_env=eval_env,
            frozen_model=frozen_model,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
            win_rate_threshold=win_rate_threshold,
            verbose=1,
        )

        # Combine both callbacks into a CallbackList.
        callback_list = CallbackList([eval_callback, frozen_update_callback])


        model.learn(total_timesteps=stage_timesteps, callback=callback_list)

        previous_main_folder = main_folder_stage

        print(f"Finished Curriculum Stage: {stage_name}")


if __name__ == "__main__":
    train()


# Create a separate evaluation environment (non-vectorized)
# eval_env = make_env()  # Using the same make_env() as before
# obs = eval_env.reset()
# for episode in range(3):
#     done = False
#     print(f"\n--- Evaluation Episode: {episode} ---")
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = eval_env.step(action)
#         print(f"Reward: {reward}")
#         eval_env.render()
#     obs = eval_env.reset()




